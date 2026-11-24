# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FreeToken expert offload for the BF16 Qwen3.6-35B-A3B single-GPU path."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache

    from vllm.config import VllmConfig
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

logger = init_logger(__name__)

NUM_LAYERS = 40
NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512
EXPERT_BYTES = 3 * HIDDEN_SIZE * INTERMEDIATE_SIZE * 2
GIB = 1024**3


@triton.jit
def _prepare_decode_routes(
    ids_ptr,
    weights_ptr,
    safe_ids_ptr,
    safe_weights_ptr,
    no_valid_route_ptr,
    top_k: tl.constexpr,
    num_experts: tl.constexpr,
):
    offsets = tl.arange(0, top_k)
    ids = tl.load(ids_ptr + offsets)
    weights = tl.load(weights_ptr + offsets)
    valid = (ids >= 0) & (ids < num_experts)
    fallback = tl.min(tl.where(valid, ids, num_experts), 0)
    no_valid_route = fallback == num_experts
    tl.store(
        safe_ids_ptr + offsets,
        tl.where(valid, ids, tl.where(no_valid_route, 0, fallback)),
    )
    tl.store(safe_weights_ptr + offsets, tl.where(valid, weights, 0.0))
    tl.store(no_valid_route_ptr, no_valid_route)


def _get_cpu_bank(
    weight: torch.nn.Parameter, shape: tuple[int, ...], name: str
) -> torch.Tensor:
    source = getattr(weight, "_vllm_uva_cpu_data", None)
    if source is None or source[1] != weight.data_ptr():
        raise RuntimeError(f"FreeToken needs the current UVA CPU bank for {name}")
    bank = source[0]
    if (
        bank.device.type != "cpu"
        or not bank.is_pinned()
        or bank.dtype != torch.bfloat16
        or bank.shape != shape
        or not bank.is_contiguous()
    ):
        raise RuntimeError(f"FreeToken needs a contiguous pinned BF16 bank for {name}")
    return bank


class Qwen36FreeTokenRuntime:
    """Own one cache per model and initialize it after weight loading."""

    def __init__(self, vllm_config: VllmConfig):
        model = vllm_config.model_config
        config = model.hf_text_config
        if (
            config.num_hidden_layers,
            config.num_experts,
            config.num_experts_per_tok,
            config.hidden_size,
            config.moe_intermediate_size,
        ) != (NUM_LAYERS, NUM_EXPERTS, TOP_K, HIDDEN_SIZE, INTERMEDIATE_SIZE):
            raise ValueError("FreeToken requires Qwen3.6-35B-A3B expert geometry")
        if model.dtype != torch.bfloat16 or vllm_config.quant_config is not None:
            raise ValueError("FreeToken requires unquantized BF16 model weights")
        parallel = vllm_config.parallel_config
        if (
            not current_platform.is_cuda()
            or parallel.world_size_across_dp != 1
            or parallel.enable_expert_parallel
            or parallel.use_ubatching
            or vllm_config.scheduler_config.max_num_seqs != 1
            or vllm_config.speculative_config is not None
            or vllm_config.lora_config is not None
            or model.enable_sleep_mode
        ):
            raise ValueError(
                "FreeToken requires one CUDA GPU and one sequence, with "
                "speculation, LoRA, sleep mode and microbatching disabled"
            )
        compilation = vllm_config.compilation_config
        if (
            compilation.mode != CompilationMode.NONE
            or compilation.cudagraph_mode
            not in (
                CUDAGraphMode.NONE,
                CUDAGraphMode.FULL_DECODE_ONLY,
            )
        ):
            raise ValueError("FreeToken requires mode=0 and NONE or FULL_DECODE_ONLY")
        offload = vllm_config.offload_config
        min_offload_gib = NUM_LAYERS * NUM_EXPERTS * EXPERT_BYTES / GIB
        if (
            offload.offload_backend != "uva"
            or offload.uva.cpu_offload_gb < min_offload_gib
            or envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA
            or envs.VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY
        ):
            raise ValueError(
                f"FreeToken requires at least {min_offload_gib:g} GiB of pinned UVA "
                "offload for the routed expert weights"
            )
        cache_gib = envs.VLLM_QWEN36_CACHE_GB
        if (
            not math.isfinite(cache_gib)
            or cache_gib * GIB < 2 * NUM_EXPERTS * EXPERT_BYTES
        ):
            raise ValueError("FreeToken GPU cache must fit two complete expert layers")
        self.cache_slots = int(cache_gib * GIB) // EXPERT_BYTES
        self.layers: list[Qwen36FreeTokenRoutedExperts] = []
        self.cache: OffloadMoeCache | None = None
        self.decode_layers: list[OffloadMoELayer] = []

    def initialize(self, device: torch.device) -> OffloadMoeCache:
        try:
            from freetoken.distributed.info import set_tp_info, try_get_tp_info
            from freetoken.layers.moe import OffloadMoELayer
            from freetoken.moe.offload_cache import OffloadMoeCache
        except ImportError as exc:
            raise RuntimeError(
                "Install the FreeToken CUDA runtime before enabling "
                "VLLM_QWEN36_FREETOKEN"
            ) from exc

        if len(self.layers) != NUM_LAYERS:
            raise RuntimeError("FreeToken requires all routed expert layers")

        gate_up = [
            _get_cpu_bank(
                layer.w13_weight,
                (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
                f"{layer.layer_name}.w13_weight",
            )
            for layer in self.layers
        ]
        down = [
            _get_cpu_bank(
                layer.w2_weight,
                (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
                f"{layer.layer_name}.w2_weight",
            )
            for layer in self.layers
        ]
        tp_info = try_get_tp_info()
        if tp_info is None:
            set_tp_info(rank=0, size=1)
        elif (tp_info.rank, tp_info.size) != (0, 1):
            raise RuntimeError("FreeToken requires TP rank 0 of 1")

        cache = OffloadMoeCache(
            num_layers=NUM_LAYERS,
            num_experts=NUM_EXPERTS,
            cache_size=self.cache_slots,
            device=device,
            cache_policy="lru",
            prefill_overlap=True,
            prefill_hit_d2d=False,
            quant_format="bf16",
            decode_target="gpu",
        )
        cache.set_bank_sources({"gate_up": gate_up, "down": down})
        cache.collect_stats = False
        decode_layers = [
            OffloadMoELayer(
                layer_id=layer_id,
                num_experts=NUM_EXPERTS,
                top_k=TOP_K,
                hidden_size=HIDDEN_SIZE,
                intermediate_size=INTERMEDIATE_SIZE,
                renormalize=True,
                activation="silu",
                apply_router_weight_on_input=False,
            )
            for layer_id in range(NUM_LAYERS)
        ]
        for layer in decode_layers:
            layer.offload_cache = cache
        self.decode_layers = decode_layers
        self.cache = cache
        self.layers.clear()
        logger.info(
            "Qwen FreeToken enabled: %d GPU slots, double-buffered prefill, GPU decode",
            self.cache_slots,
        )
        return cache


class Qwen36FreeTokenRoutedExperts(RoutedExperts):
    """Keep vLLM routing and shared experts, replacing routed expert execution."""

    def __init__(self, *args, runtime: Qwen36FreeTokenRuntime, layer_id: int, **kwargs):
        super().__init__(*args, **kwargs)
        method = self.quant_method
        if (
            not isinstance(method, UnquantizedFusedMoEMethod)
            or method.unquantized_backend != UnquantizedMoeBackend.TRITON
            or method.is_monolithic
            or self.activation != MoEActivation.SILU
            or self.moe_config.has_bias
            or self.apply_router_weight_on_input
        ):
            raise ValueError("FreeToken requires the modular BF16 Triton MoE backend")
        if layer_id != len(runtime.layers):
            raise RuntimeError("FreeToken expert layers must be registered in order")
        self._freetoken = runtime
        self._freetoken_layer_id = layer_id
        self.w13_weight._vllm_retain_uva_cpu_data = True
        self.w2_weight._vllm_retain_uva_cpu_data = True
        runtime.layers.append(self)

    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        runtime = self._freetoken
        cache = runtime.cache
        if cache is None:
            cache = runtime.initialize(x.device)
        layer_id = self._freetoken_layer_id
        if x.shape[0] == 1:
            # The vLLM router emits -1 for padding, including the capture warmup.
            # FreeToken's slot cache requires every index to name a real expert.
            ids = topk_ids.contiguous()
            weights = topk_weights.contiguous()
            safe_ids = torch.empty_like(ids)
            safe_weights = torch.empty_like(weights)
            no_valid_route = torch.empty(1, dtype=torch.bool, device=x.device)
            _prepare_decode_routes[(1,)](
                ids, weights, safe_ids, safe_weights, no_valid_route, TOP_K, NUM_EXPERTS
            )
            # FreeToken rewrites the scratch IDs to cache slot IDs in place.
            routed = runtime.decode_layers[layer_id]._decode_routed(
                x, safe_weights, safe_ids
            )
            return routed.masked_fill_(no_valid_route, 0)

        kernel = self.quant_method.moe_kernel
        assert kernel is not None
        if layer_id == 0:
            logger.debug("Qwen expert prefill chunk: tokens=%d", x.shape[0])
            cache.begin_prefill()
        cache.prefetch_prefill_layer(layer_id)
        if layer_id + 1 < NUM_LAYERS:
            cache.prefetch_prefill_layer(layer_id + 1)
        views = cache.wait_prefill_layer(layer_id)
        try:
            w1, w2 = views
            return kernel.apply(
                hidden_states=x,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=self.activation,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )
        finally:
            cache.release_prefill_layer(layer_id)
