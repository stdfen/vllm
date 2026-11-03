# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.weight_staging import copy_loaded_weight
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

    from .config import Exl3Config

_FUSED_TEMP_CACHE: dict[tuple[str, int, int, int, int], tuple[torch.Tensor, ...]] = {}


class Exl3MoEMethod(FusedMoEMethodBase):
    """Routed EXL3 experts using v0.32's expert ownership and loader ABI."""

    def __init__(
        self, moe: FusedMoEConfig, quant_config: Exl3Config, prefix: str
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.prefix = prefix
        self.bits = quant_config.bits_for_moe_prefix(prefix)

    def get_fused_moe_quant_config(self, layer: RoutedExperts):
        del layer
        return None

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        if hidden_size % 16 or intermediate_size_per_partition % 16:
            raise ValueError(
                "EXL3 MoE requires 16x16-aligned hidden/intermediate sizes"
            )
        hidden_tiles = hidden_size // 16
        intermediate_tiles = intermediate_size_per_partition // 16
        words = self.bits * 16
        codebook = self.quant_config.codebook
        shapes = {
            "w13_trellis": (
                num_experts,
                2,
                hidden_tiles,
                intermediate_tiles,
                words,
            ),
            "w13_suh": (num_experts, 2, hidden_size),
            "w13_svh": (num_experts, 2, intermediate_size_per_partition),
            f"w13_{codebook}": (num_experts, 2),
            "w2_trellis": (
                num_experts,
                intermediate_tiles,
                hidden_tiles,
                words,
            ),
            "w2_suh": (num_experts, intermediate_size_per_partition),
            "w2_svh": (num_experts, hidden_size),
            f"w2_{codebook}": (num_experts,),
        }
        dtypes = {
            "trellis": torch.int16,
            "suh": torch.float16,
            "svh": torch.float16,
            codebook: torch.int32,
        }
        attrs = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}
        for name, shape in shapes.items():
            role = name.rsplit("_", 1)[-1]
            parameter = Parameter(
                torch.empty(shape, dtype=dtypes[role]), requires_grad=False
            )
            set_weight_attrs(
                parameter,
                {
                    "weight_loader": self._load_expert_parameter,
                    "_exl3_owner": layer,
                    **attrs,
                },
            )
            layer.register_parameter(name, parameter)
        layer._exl3_hidden_size = hidden_size
        layer._exl3_intermediate_size = intermediate_size_per_partition

    def _load_expert_parameter(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        owner = param._exl3_owner
        local_id = owner._map_global_expert_id_to_local_expert_id(expert_id)
        if local_id == -1:
            return False if return_success else None
        tp_rank = self.moe.tp_rank
        tp_size = self.moe.moe_parallel_config.tp_size
        role = weight_name.rsplit("_", 1)[-1]
        value = loaded_weight
        if shard_id in ("w1", "w3"):
            slot = 0 if shard_id == "w1" else 1
            if role == "trellis":
                value = _tp_slice(value, 1, tp_rank, tp_size)
            elif role == "svh":
                value = _tp_slice(value, 0, tp_rank, tp_size)
            target = param.data[local_id, slot]
        elif shard_id == "w2":
            if role == "trellis" or role == "suh":
                value = _tp_slice(value, 0, tp_rank, tp_size)
            target = param.data[local_id]
        else:
            raise ValueError(f"unsupported EXL3 expert shard {shard_id!r}")
        value = value.reshape(target.shape)
        if value.shape != target.shape:
            raise ValueError(
                f"EXL3 expert shape mismatch for {weight_name}: "
                f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
            )
        copy_loaded_weight(target, value)
        return True if return_success else None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        from .runtime import create_exl3_linear_backend

        codebook = self.quant_config.codebook
        marker13 = getattr(layer, f"w13_{codebook}")
        marker2 = getattr(layer, f"w2_{codebook}")
        backends = []
        for expert_id in range(layer.local_num_experts):
            gate_up = tuple(
                create_exl3_linear_backend(
                    trellis=layer.w13_trellis[expert_id, slot],
                    suh=layer.w13_suh[expert_id, slot],
                    svh=layer.w13_svh[expert_id, slot],
                    marker=marker13[expert_id, slot],
                    bits=self.bits,
                    codebook=codebook,
                )
                for slot in range(2)
            )
            down = create_exl3_linear_backend(
                trellis=layer.w2_trellis[expert_id],
                suh=layer.w2_suh[expert_id],
                svh=layer.w2_svh[expert_id],
                marker=marker2[expert_id],
                bits=self.bits,
                codebook=codebook,
            )
            backends.append((*gate_up, down))
        layer._exl3_expert_backends = tuple(backends)
        self._build_fused_decode_state(layer)

    def _build_fused_decode_state(self, layer: RoutedExperts) -> None:
        import exllamav3_ext

        if not hasattr(exllamav3_ext, "exl3_moe"):
            layer._exl3_fused_decode = None
            return
        device = layer.w13_trellis.device

        def pointers(parameter: torch.Tensor, slot: int | None = None) -> torch.Tensor:
            values = parameter[:, slot] if slot is not None else parameter
            return torch.tensor(
                [int(value.data_ptr()) for value in values],
                dtype=torch.int64,
                device=device,
            )

        pointer_tables = {
            "gate_trellis": pointers(layer.w13_trellis, 0),
            "gate_suh": pointers(layer.w13_suh, 0),
            "gate_svh": pointers(layer.w13_svh, 0),
            "up_trellis": pointers(layer.w13_trellis, 1),
            "up_suh": pointers(layer.w13_suh, 1),
            "up_svh": pointers(layer.w13_svh, 1),
            "down_trellis": pointers(layer.w2_trellis),
            "down_suh": pointers(layer.w2_suh),
            "down_svh": pointers(layer.w2_svh),
        }
        device_index = device.index if device.index is not None else 0
        concurrency = max(int(exllamav3_ext.exl3_moe_max_concurrency(device_index)), 1)
        rows = 128
        hidden = layer._exl3_hidden_size
        intermediate = layer._exl3_intermediate_size
        cache_key = (str(device), hidden, intermediate, concurrency, rows)
        temps = _FUSED_TEMP_CACHE.get(cache_key)
        if temps is None:
            temps = tuple(
                torch.empty(shape, dtype=torch.float16, device=device)
                for shape in (
                    (concurrency, rows, hidden),
                    (concurrency, rows, hidden),
                    (concurrency, rows, intermediate),
                    (concurrency, rows, intermediate),
                )
            )
            _FUSED_TEMP_CACHE[cache_key] = temps
        layer._exl3_fused_decode = (pointer_tables, temps)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if getattr(layer, "_exl3_fused_decode", None) is not None:
            if x.shape[0] <= 128:
                output = self._apply_fused_decode(layer, x, topk_weights, topk_ids)
                layer._exl3_last_apply = "fused_decode"
                return output
            output = torch.cat(
                [
                    self._apply_fused_decode(
                        layer,
                        x[start : start + 128],
                        topk_weights[start : start + 128],
                        topk_ids[start : start + 128],
                    )
                    for start in range(0, x.shape[0], 128)
                ],
                dim=0,
            )
            layer._exl3_last_apply = "fused_prefill_tiles"
            return output
        layer._exl3_last_apply = "reference_loop"
        backends = layer._exl3_expert_backends
        output = torch.zeros_like(x)
        expert_map = layer.expert_map
        for global_id in torch.unique(topk_ids).tolist():
            local_id = (
                int(expert_map[global_id].item())
                if expert_map is not None
                else int(global_id)
            )
            if local_id < 0:
                continue
            token_index, topk_slot = torch.where(topk_ids == global_id)
            expert_input = x[token_index]
            gate_backend, up_backend, down_backend = backends[local_id]
            limit = float(getattr(layer, "swiglu_limit", None) or 10.0)
            gate = gate_backend(expert_input).clamp(max=limit)
            up = up_backend(expert_input).clamp(min=-limit, max=limit)
            activated = F.silu(gate) * up
            expert_output = down_backend(activated)
            routed = expert_output * topk_weights[token_index, topk_slot, None]
            output.index_add_(0, token_index, routed.to(output.dtype))
        return output

    def _apply_fused_decode(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        import exllamav3_ext

        pointer_tables, temps = layer._exl3_fused_decode
        tokens = x.shape[0]
        ids = topk_ids.reshape(tokens, -1).long()
        local = ids
        expert_map = layer.expert_map
        if expert_map is not None:
            safe = ids.clamp(min=0, max=max(expert_map.numel() - 1, 0))
            local = expert_map[safe]
        invalid = (local < 0) | (local >= layer.local_num_experts)
        local = torch.where(invalid, local.new_full((), layer.local_num_experts), local)
        topk = ids.shape[1]
        flat_token = torch.arange(tokens, device=x.device).repeat_interleave(topk)
        flat_local = local.reshape(-1)
        order = flat_local.argsort()
        token_sorted = flat_token[order]
        weight_sorted = topk_weights.reshape(-1)[order].half()
        expert_count = torch.zeros(
            layer.local_num_experts + 1, dtype=torch.long, device=x.device
        )
        expert_count.scatter_add_(
            0, flat_local, torch.ones_like(flat_local, dtype=torch.long)
        )
        output = torch.zeros_like(x, dtype=torch.float32)
        codebook = self.quant_config.codebook
        exllamav3_ext.exl3_moe(
            x.contiguous().half(),
            output,
            expert_count,
            token_sorted,
            weight_sorted,
            *temps,
            0,
            self.bits,
            self.bits,
            self.bits,
            pointer_tables["gate_trellis"],
            pointer_tables["gate_suh"],
            pointer_tables["gate_svh"],
            pointer_tables["up_trellis"],
            pointer_tables["up_suh"],
            pointer_tables["up_svh"],
            pointer_tables["down_trellis"],
            pointer_tables["down_suh"],
            pointer_tables["down_svh"],
            codebook == "mcg",
            codebook == "mul1",
            codebook == "mcg",
            codebook == "mul1",
            codebook == "mcg",
            codebook == "mul1",
            float(getattr(layer, "swiglu_limit", None) or 10.0),
            -1,
        )
        return output.to(x.dtype)


def _tp_slice(
    value: torch.Tensor, dim: int, tp_rank: int, tp_size: int
) -> torch.Tensor:
    if value.shape[dim] % tp_size:
        raise ValueError(
            f"EXL3 expert axis {value.shape[dim]} is not divisible by TP={tp_size}"
        )
    return value.chunk(tp_size, dim=dim)[tp_rank]
