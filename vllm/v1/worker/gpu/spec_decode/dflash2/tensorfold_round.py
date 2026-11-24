# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TensorFold 0.6.0 draft rounds with vLLM-owned target state and scheduling."""

import json
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from os import getenv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from prometheus_client import Counter
from safetensors import safe_open

from vllm.forward_context import get_forward_context, override_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_plan import (
    C1_GRAPH_ROWS,
    TreeBatchPlan,
    TreeWindow,
    VerifyCosts,
    batch_attention_chunk_bucket,
    c8_attention_chunk_bucket,
    can_capture_batch_verification_graph,
    can_capture_verification_graph,
    is_long_verification_graph,
)

if TYPE_CHECKING:
    from tensorfold.families.qwen3_5.cuda.decode import CopyIndex

    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_commit import C1CommitGraph
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_draft_graph import (
        C1DraftGraph,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_target import (
        TensorFoldNativeTarget,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify import (
        TensorFoldVerify,
    )

logger = init_logger(__name__)


@cache
def _verification_counter() -> Counter:
    return Counter(
        "vllm:tensorfold_verification_total",
        "Scheduled TensorFold verification steps, request rounds and draft rows.",
        ("unit", "batch_size", "short_c8", "bucketed", "route"),
    )


@cache
def _batch_bucket_counter() -> Counter:
    return Counter(
        "vllm:tensorfold_batch_graph_bucket_total",
        "TensorFold verification steps using the additional multi-request buckets.",
        ("batch_size", "route"),
    )


@dataclass
class DraftRequest:
    copy_index: "CopyIndex"
    context: list[int] = field(default_factory=list)
    taps: tuple | None = None
    proposed: TreeWindow | None = None
    reported_additional_graph: bool = False
    reported_padded_graph: bool = False
    reported_device_accept: bool = False
    reported_reference_curve: bool = False
    reported_pinned_readback: bool = False
    reported_async_draft_upload: bool = False
    reported_direct_stage: bool = False
    reported_resident_direct_stage: bool = False
    reported_grouped_resident_stage: bool = False
    reported_c1_qmm_split: bool = False
    reported_fp8_tree_tail: bool = False
    reported_direct_recurrent_commit: bool = False
    reported_recurrent_commit_coverage: bool = False


@dataclass
class DraftInput:
    committed_ids: list[int]
    position: int
    num_sampled: int
    pending: int


@dataclass
class VerificationGraph:
    verify: "TensorFoldVerify"
    input_ids: torch.Tensor
    positions: torch.Tensor
    graph: torch.cuda.CUDAGraph
    output: tuple[Any, ...]
    uses_additional_slot: bool = False


def store_tensorfold_draft_tokens(
    drafts: list[list[int]],
    request_rows: torch.Tensor,
    draft_buffer: torch.Tensor,
    *,
    async_upload: bool = False,
) -> None:
    width = draft_buffer.shape[1]
    rows = [row + [-1] * (width - len(row)) for row in drafts]
    if async_upload:
        staged = async_tensor_h2d(
            rows, device=draft_buffer.device, dtype=draft_buffer.dtype
        )
    else:
        staged = torch.tensor(
            rows, dtype=draft_buffer.dtype, device=draft_buffer.device
        )
    draft_buffer[request_rows] = staged


def _load_head(model_dir: Path, device: torch.device):
    from tensorfold.cuda.nvfp4.linear import Fp4Linear

    keys = ("lm_head.weight", "lm_head.weight_scale", "lm_head.weight_scale_2")
    with (model_dir / "model.safetensors.index.json").open() as f:
        shards = json.load(f)["weight_map"]
    weights = {}
    for shard in {shards[key] for key in keys}:
        with safe_open(model_dir / shard, framework="pt") as reader:
            weights.update(
                (key, reader.get_tensor(key)) for key in keys if shards[key] == shard
            )
    weight, scale, global_scale = (weights[key] for key in keys)
    return Fp4Linear.from_checkpoint(
        weight.to(device), scale.to(device), float(global_scale)
    )


class TensorFoldTreeDFlash2:
    def __init__(self, config: "VllmConfig", device: torch.device):
        try:
            installed = version("tensorfold")
        except PackageNotFoundError as exc:
            raise ValueError(
                "Install TensorFold 0.6.0 for tree mode=tensorfold."
            ) from exc
        if installed != "0.6.0":
            raise ValueError(f"TensorFold tree mode requires 0.6.0, found {installed}.")
        if torch.cuda.get_device_capability(device) != (12, 1):
            raise ValueError("The TensorFold tree mode currently targets NVIDIA GB10.")
        spec, model = config.speculative_config, config.model_config
        assert spec is not None and spec.draft_model_config is not None
        if model is None or model.quantization not in (
            "modelopt_fp4",
            "modelopt_mixed",
        ):
            raise ValueError("TensorFold tree mode requires a ModelOpt NVFP4 target.")
        self.model_dir = Path(model.model)
        self.draft_dir = Path(spec.draft_model_config.model)
        if (
            not (self.model_dir / "model.safetensors.index.json").is_file()
            or not (self.draft_dir / "model.safetensors").is_file()
        ):
            raise ValueError(
                "TensorFold tree mode needs local target and draft weights."
            )
        self.device = device
        self.requests: dict[str, DraftRequest] = {}
        self._committed_draft_input: tuple[InputBatch, list[DraftInput]] | None = None
        self.cost = VerifyCosts()
        self.max_nodes = config.num_speculative_tokens
        self.use_graph = spec.tree_dflash2_tensorfold_cuda_graph
        self.batch_device_commit = spec.tree_dflash2_tensorfold_batch_device_commit
        self.c1_graph_bucket = spec.tree_dflash2_tensorfold_c1_graph_bucket
        self.c1_direct_stage = spec.tree_dflash2_tensorfold_c1_direct_stage
        self.direct_recurrent_commit = (
            spec.tree_dflash2_tensorfold_c1_direct_recurrent_commit
        )
        self.recurrent_commit_coverage = (
            spec.tree_dflash2_tensorfold_c1_recurrent_commit_coverage
        )
        self.expand_c1_graph_bank = spec.tree_dflash2_tensorfold_c1_expand_graph_bank
        self.c1_fixed_width = spec.tree_dflash2_tensorfold_c1_fixed_width
        self.c1_commit_graph = spec.tree_dflash2_tensorfold_c1_commit_graph
        self.overlap_commit = spec.tree_dflash2_tensorfold_c1_overlap_commit
        self.device_accept = spec.tree_dflash2_tensorfold_c1_device_accept
        self.device_metadata = spec.tree_dflash2_tensorfold_c1_device_metadata
        self.grouped_resident_stage = (
            spec.tree_dflash2_tensorfold_c1_grouped_resident_stage
        )
        self.resident_direct_stage = (
            spec.tree_dflash2_tensorfold_c1_resident_direct_stage
            or self.grouped_resident_stage
        )
        self.c1_reference_curve = spec.tree_dflash2_tensorfold_c1_reference_curve
        self.pending_commit: C1CommitGraph | None = None
        self.use_c1_draft_graph = spec.tree_dflash2_tensorfold_c1_draft_graph
        self.c1_pinned_readback = spec.tree_dflash2_tensorfold_c1_pinned_readback
        self.c1_async_draft_upload = spec.tree_dflash2_tensorfold_c1_async_draft_upload
        self.c1_draft_graph: C1DraftGraph | None = None
        self.fused_conv = spec.tree_dflash2_tensorfold_fused_conv
        self.use_native_target = spec.tree_dflash2_tensorfold_native_target
        self.c1_qmm_split = spec.tree_dflash2_tensorfold_c1_qmm_split
        self.c1_fp8_tree_tail = spec.tree_dflash2_tensorfold_c1_fp8_tree_tail
        self.native_target: TensorFoldNativeTarget | None = None
        self.graphs: dict[tuple, VerificationGraph] = {}
        self.long_context_graphs: dict[int, dict[tuple, VerificationGraph]] | None = (
            {} if spec.tree_dflash2_tensorfold_long_context_graph else None
        )
        self.c8_graph_buckets = spec.tree_dflash2_tensorfold_c8_graph_buckets
        self.batch_graph_buckets = spec.tree_dflash2_tensorfold_batch_graph_buckets
        self.batch_graphs: dict[tuple, VerificationGraph] = {}
        self.verification_metrics = (
            _verification_counter()
            if getenv("TF_VERIFY_GRAPH_METRICS") == "1"
            else None
        )
        self.batch_bucket_metrics = (
            _batch_bucket_counter()
            if self.batch_graph_buckets and self.verification_metrics is not None
            else None
        )
        self.c1_graphs: dict[tuple, VerificationGraph] = {}
        self.observed_shapes: OrderedDict[tuple, None] = OrderedDict()

    def _replay_direct_c1(
        self,
        window: TreeWindow,
        plan: TreeBatchPlan,
        length: int,
        metadata: dict,
        layers: dict,
        slots: dict,
        ids: torch.Tensor,
        positions: torch.Tensor,
        request: DraftRequest,
    ) -> tuple["TensorFoldVerify", tuple[Any, ...], str] | None:
        from tensorfold.cuda.kernels import gdn

        from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify import (
            gdn_slot_bucket,
        )

        graphs = (
            self.c1_graphs
            if self.expand_c1_graph_bank or self.resident_direct_stage
            else self.graphs
        )
        if not graphs:
            return None
        entries, _, num_slots, _ = gdn.plan_host([window.parents])
        bucket = gdn_slot_bucket(num_slots)
        for graph in graphs.values():
            verify = graph.verify
            if (
                verify.c1_buffers is None
                or verify.layers is not layers
                or verify.gdn_plan.slots != bucket
                or graph.input_ids.shape != ids.shape
                or graph.positions.shape != positions.shape
            ):
                continue
            depths = verify.stage_c1(window, plan, length, entries, metadata, slots)
            if depths is not None:
                graph.input_ids.copy_(ids, non_blocking=True)
                graph.positions.copy_(positions[..., :1] + depths, non_blocking=True)
                graph.graph.replay()
                logger.info_once("TensorFold replayed a fixed-capacity C1 tree graph.")
                logger.info_once(
                    "TensorFold Tree-DFlash2 CUDA graph replay executed "
                    "for %d requests.",
                    1,
                )
                if verify.resident_recurrent is not None:
                    if not request.reported_resident_direct_stage:
                        request.reported_resident_direct_stage = True
                        logger.info(
                            "TensorFold refreshed a C1 graph from resident slots."
                        )
                    if (
                        verify.c1_resident_layout is not None
                        and not request.reported_grouped_resident_stage
                    ):
                        request.reported_grouped_resident_stage = True
                        logger.info(
                            "TensorFold refreshed a C1 graph with grouped GDN slots."
                        )
                elif not request.reported_direct_stage:
                    request.reported_direct_stage = True
                    logger.info("TensorFold refreshed a C1 graph directly.")
                return verify, graph.output, "replay"
        return None

    def _target_forward(
        self,
        verify: "TensorFoldVerify",
        ids: torch.Tensor,
        positions: torch.Tensor,
        packed_forward: Callable,
        compute_logits: Callable,
        c1_request: DraftRequest | None = None,
        logical_rows: int | None = None,
    ) -> tuple["TensorFoldVerify", tuple[Any, ...], str]:
        context = get_forward_context()

        def run(state: "TensorFoldVerify", tokens: torch.Tensor, pos: torch.Tensor):
            state.gdn_records.clear()
            state.kv_records.clear()
            state.prepare_conv_history()
            if self.native_target is not None:
                logger.info_once("TensorFold native target verified a packed tree.")
                return self.native_target.forward(state, tokens, pos)
            with override_forward_context(
                replace(
                    context,
                    attn_metadata=state.metadata,
                    slot_mapping={},
                    additional_kwargs={
                        **context.additional_kwargs,
                        "tree_dflash2": state,
                        "tensorfold_tree_attention": state,
                    },
                )
            ):
                hidden, aux = packed_forward(tokens, pos)
                return hidden, aux, compute_logits(hidden).argmax(-1)

        if not self.use_graph:
            return verify, run(verify, ids, positions), "eager"
        requests = len(verify.plan.starts) - 1
        chunks = verify.attention_plan.chunks
        use_c1_bank = (
            self.expand_c1_graph_bank or self.resident_direct_stage
        ) and verify.is_c1_graph_bucket
        if chunks > 16:
            if self.long_context_graphs is None or not is_long_verification_graph(
                requests, ids.numel(), chunks
            ):
                return verify, run(verify, ids, positions), "eager"
            # Each chunk count gets the existing two-shape limit for one batch size.
            graphs = self.long_context_graphs.setdefault(chunks, {})
        elif verify.is_batch_graph_bucket:
            graphs = self.batch_graphs
        else:
            graphs = self.c1_graphs if use_c1_bank else self.graphs
        shape = (requests, ids.shape, positions.shape, verify.graph_key())
        captured = graphs.get(shape)
        if captured is not None:
            captured.verify.stage(verify)
            captured.input_ids.copy_(ids, non_blocking=True)
            captured.positions.copy_(positions, non_blocking=True)
            captured.graph.replay()
            if verify.is_c8_graph_bucket:
                logger.info_once(
                    "TensorFold replayed bucketed C8 verifier: "
                    "%d chunks, %d GDN slots.",
                    chunks,
                    verify.gdn_plan.slots,
                )
            if verify.is_batch_graph_bucket:
                logger.info_once(
                    "TensorFold replayed a multi-request verifier bucket: "
                    "%d requests, %d rows, %d chunks, %d GDN slots.",
                    requests,
                    ids.numel(),
                    chunks,
                    verify.gdn_plan.slots,
                )
            if chunks > 16:
                logger.info_once(
                    "TensorFold long-context C8 verifier graph replay: %d chunks.",
                    chunks,
                )
            if verify.is_c1_graph_bucket:
                logger.info_once("TensorFold replayed a fixed-capacity C1 tree graph.")
            if (
                captured.uses_additional_slot
                and c1_request is not None
                and not c1_request.reported_additional_graph
            ):
                c1_request.reported_additional_graph = True
                logger.info(
                    "TensorFold replayed an additional C1 verifier graph for %d rows.",
                    ids.numel(),
                )
            if (
                logical_rows is not None
                and c1_request is not None
                and not c1_request.reported_padded_graph
            ):
                c1_request.reported_padded_graph = True
                logger.info(
                    "TensorFold replayed the padded C1 verifier for "
                    "%d real rows in %d rows.",
                    logical_rows,
                    ids.numel(),
                )
            logger.info_once(
                "TensorFold Tree-DFlash2 CUDA graph replay executed for %d requests.",
                requests,
            )
            return captured.verify, captured.output, "replay"
        can_capture = (
            can_capture_batch_verification_graph(graphs, requests)
            if verify.is_batch_graph_bucket
            else can_capture_verification_graph(graphs, requests, use_c1_bank)
        )
        if not can_capture:
            return verify, run(verify, ids, positions), "eager"
        if shape not in self.observed_shapes:
            self.observed_shapes[shape] = None
            if len(self.observed_shapes) > 64:
                self.observed_shapes.popitem(last=False)
            return verify, run(verify, ids, positions), "eager"
        del self.observed_shapes[shape]
        direct = (
            self.resident_direct_stage
            if verify.resident_recurrent is not None
            else self.c1_direct_stage
        )
        if (
            direct
            and verify.is_c1_graph_bucket
            and not verify.conv_groups
            and not self.c1_fixed_width
        ):
            verify.bind_c1_graph_inputs(grouped_resident=self.grouped_resident_stage)
        tokens, pos = ids.clone(), positions.clone()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            run(verify, tokens, pos)
            with torch.cuda.graph(graph, stream=stream):
                output = run(verify, tokens, pos)
        current.wait_stream(stream)
        if self.batch_device_commit and requests > 1:
            verify.capture_batch_commit = True
        if self.c1_commit_graph and verify.is_c1_graph_bucket:
            verify.bind_commit_graph(
                overlap=self.overlap_commit,
                device_accept=self.device_accept and ids.numel() <= C1_GRAPH_ROWS,
                direct_recurrent_commit=(
                    self.direct_recurrent_commit or self.recurrent_commit_coverage
                ),
            )
        graphs[shape] = VerificationGraph(
            verify,
            tokens,
            pos,
            graph,
            output,
            use_c1_bank and self.expand_c1_graph_bank and len(graphs) >= 2,
        )
        graph.replay()
        if verify.is_batch_graph_bucket:
            logger.info(
                "TensorFold captured a multi-request verifier bucket: "
                "%d requests, %d rows, %d chunks, %d GDN slots, "
                "allocated=%d bytes, reserved=%d bytes.",
                requests,
                ids.numel(),
                chunks,
                verify.gdn_plan.slots,
                torch.cuda.memory_allocated(self.device),
                torch.cuda.memory_reserved(self.device),
            )
        if chunks > 16:
            logger.info(
                "TensorFold captured long-context C8 verifier graph: "
                "%d chunks, allocated=%d bytes, reserved=%d bytes.",
                chunks,
                torch.cuda.memory_allocated(self.device),
                torch.cuda.memory_reserved(self.device),
            )
        logger.info(
            "TensorFold Tree-DFlash2 captured and launched a CUDA graph "
            "for %d requests and %d rows.",
            requests,
            ids.numel(),
        )
        if verify.is_c1_graph_bucket:
            logger.info(
                "TensorFold C1 verifier graph memory after capture: "
                "allocated=%d bytes, reserved=%d bytes.",
                torch.cuda.memory_allocated(self.device),
                torch.cuda.memory_reserved(self.device),
            )
        return verify, output, "capture"

    def _record_verification(
        self,
        batch: InputBatch,
        route: str,
        short_c8: bool = False,
        bucketed: bool = False,
    ) -> None:
        if self.verification_metrics is None or not batch.num_draft_tokens:
            return
        counts = batch.num_draft_tokens_per_req
        assert counts is not None
        for unit, value in (
            ("step", 1),
            ("request_round", int((counts > 0).sum())),
            ("draft_row", batch.num_draft_tokens),
        ):
            self.verification_metrics.labels(
                unit, batch.num_reqs, int(short_c8), int(bucketed), route
            ).inc(value)

    def load_drafter(self, model: torch.nn.Module) -> None:
        from tensorfold.cuda.kernels import gdn
        from tensorfold.families.qwen3_5.cuda.dflash2 import DFlash2
        from tensorfold.families.qwen3_5.cuda.weights import Config, Plain, Weights

        if (
            self.direct_recurrent_commit or self.recurrent_commit_coverage
        ) and not getattr(gdn, "SUPPORTS_REPLAY_DESTINATIONS", False):
            raise ValueError(
                "Direct recurrent commit requires destination-slot replay."
            )
        decoder = (
            model.get_language_model()
            if hasattr(model, "get_language_model")
            else model
        )
        target = decoder.model
        if self.use_native_target:
            from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_target import (
                TensorFoldNativeTarget,
            )

            self.native_target = TensorFoldNativeTarget(
                target, self.model_dir, self.device, c1_qmm_split=self.c1_qmm_split
            )
            weights = self.native_target.weights
        else:
            cfg = Config.read(self.model_dir)
            embedding = target.embed_tokens.weight.detach()[: cfg.vocab]
            if embedding.dtype != torch.bfloat16 or not embedding.is_contiguous():
                raise ValueError(
                    "TensorFold tree mode requires contiguous BF16 embeddings."
                )
            weights = Weights(
                config=cfg,
                embed=Plain(embedding),
                layers=[],
                norm=target.norm.weight,
                head=_load_head(self.model_dir, self.device),
                quant="nvfp4",
            )
        self.drafter = DFlash2(self.draft_dir, weights)
        if self.use_c1_draft_graph:
            from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_draft_graph import (
                C1DraftGraph,
            )

            if not (
                self.drafter.fast
                and self.drafter.world == 1
                and hasattr(self.drafter, "launch_c1_device")
            ):
                raise ValueError(
                    "C1 draft Graph needs the native device draft entry point."
                )
            self.c1_draft_graph = C1DraftGraph(
                self.drafter, self.max_nodes, pinned_readback=self.c1_pinned_readback
            )
        gdn._ext()
        logger.info(
            "TensorFold 0.6.0 DFlash2 loaded with tiled 4-bit draft projections."
        )

    def add_request(self, req_id: str, params: SamplingParams | None) -> None:
        from tensorfold.families.qwen3_5.cuda.decode import CopyIndex

        if params is not None and (
            params.temperature == 0
            and not params.min_tokens
            and params.allowed_token_ids is None
            and not params.logit_bias
            and not params.bad_words
            and not params.presence_penalty
            and not params.frequency_penalty
            and params.repetition_penalty == 1.0
            and params.structured_outputs is None
            and params.thinking_token_budget is None
            and params.logprobs is None
            and params.prompt_logprobs is None
            and not params.trace_decode_token_ids
        ):
            self.requests[req_id] = DraftRequest(CopyIndex())

    def remove_request(self, req_id: str) -> None:
        self.requests.pop(req_id, None)

    def wait_for_commit(self) -> None:
        if self.pending_commit is not None:
            self.pending_commit.wait()
            self.pending_commit = None

    def propose(
        self,
        batch: InputBatch,
        aux: list[torch.Tensor],
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        limit: int,
    ) -> list[list[int]]:
        from tensorfold.families.qwen3_5.cuda.draft_tree import allocate

        proposals: list[list[int]] = [[] for _ in batch.req_ids]
        committed, self._committed_draft_input = self._committed_draft_input, None
        if not any(req_id in self.requests for req_id in batch.req_ids):
            return proposals
        starts = batch.query_start_loc_np[: batch.num_reqs + 1].tolist()
        if committed is not None and committed[0] is batch:
            draft_inputs = committed[1]
            logger.info_once(
                "TensorFold reused the host verification receipt for drafting."
            )
        else:
            ids = batch.input_ids[: starts[-1]].tolist()
            pending = last_sampled.index_select(0, batch.idx_mapping).flatten()
            first_rows = batch.query_start_loc[: batch.num_reqs].long()
            positions = batch.positions.index_select(0, first_rows)
            receipt = torch.stack((num_rejected, num_sampled, pending, positions), 1)
            draft_inputs = [
                DraftInput(
                    ids[starts[i] : max(starts[i], starts[i + 1] - rejected)],
                    pos,
                    sampled,
                    token,
                )
                for i, (rejected, sampled, token, pos) in enumerate(receipt.tolist())
            ]
        touched: list[DraftRequest] = []
        snaps, taps = [], []
        live: list[tuple[int, DraftRequest, int]] = []
        for i, (req_id, draft_input) in enumerate(
            zip(batch.req_ids, draft_inputs, strict=True)
        ):
            entry = self.requests.get(req_id)
            if entry is None:
                continue
            entry.proposed = None
            pos = draft_input.position
            start, stop = starts[i], starts[i] + len(draft_input.committed_ids)
            if stop <= start:
                continue
            if entry.taps is None or entry.taps[3] != pos:
                self.drafter.restore(
                    ([None] * self.drafter.layers, [None] * self.drafter.layers, 0, pos)
                )
                entry.taps = self.drafter.snapshot()
                entry.context.clear()
                entry.copy_index = type(entry.copy_index)()
            entry.context.extend(draft_input.committed_ids)
            # All five taps use the same accepted-row order as the KV commit.
            if stop - start > self.drafter.window:
                start = stop - self.drafter.window
                self.drafter.restore(entry.taps)
                self.drafter.skip(start - starts[i])
                entry.taps = self.drafter.snapshot()
            taps.append(torch.cat([x[start:stop] for x in aux], dim=-1))
            snaps.append(entry.taps)
            touched.append(entry)
            if draft_input.num_sampled and limit:
                live.append((i, entry, draft_input.pending))
        if touched:
            for entry, snap in zip(
                touched, self.drafter.add_taps_streams(snaps, taps), strict=True
            ):
                entry.taps = snap

        copy, neural = {}, []
        for i, entry, token in live:
            copied = entry.copy_index.propose([*entry.context, token], limit)
            if copied:
                copy[i] = copied
            else:
                neural.append((i, entry, token))
        trees = {}
        if neural:
            block = None
            if self.c1_draft_graph is not None and batch.num_reqs == 1:
                _, entry, pending = neural[0]
                assert entry.taps is not None
                block = self.c1_draft_graph.launch(entry.taps, pending, limit)
                if (
                    block is not None
                    and block[0][2] is not None
                    and not entry.reported_pinned_readback
                ):
                    entry.reported_pinned_readback = True
                    logger.info("C1 draft Graph returned pinned host predictions.")
            launched = (
                [block]
                if block is not None
                else self.drafter.launch_blocks(
                    [entry.taps for _, entry, _ in neural],
                    [token for _, _, token in neural],
                    limit,
                )
            )
            for (i, entry, _), block in zip(neural, launched, strict=True):
                candidates, parents, scores = self.drafter.finish_tree(
                    block, entry.taps[3] + 1, limit
                )
                if candidates:
                    trees[i] = (candidates, parents, scores)
        total_streams = sum(item.num_sampled > 0 for item in draft_inputs)
        use_reference = self.c1_reference_curve and batch.num_reqs == 1
        counts = []
        if trees:
            counts = allocate(
                [tree[2] for tree in trees.values()],
                total_streams + sum(map(len, copy.values())),
                float(total_streams),
                self.cost.curve(use_reference=use_reference),
                8.0 + 1.5 * total_streams,
            )
            if use_reference:
                request = self.requests[batch.req_ids[0]]
                if not request.reported_reference_curve:
                    request.reported_reference_curve = True
                    logger.info("C1 drafting used the TensorFold reference cost curve.")
        widths = dict(zip(trees, counts, strict=True))
        for i, entry, token in live:
            if i in copy:
                tokens = copy[i]
                parents = list(range(len(tokens)))
            elif i in trees:
                ids, upstream, _ = trees[i]
                tokens = ids[: widths[i]]
                parents = [p + 1 for p in upstream[: widths[i]]]
            else:
                continue
            entry.proposed = TreeWindow((token, *tokens), (-1, *parents))
            proposals[i] = tokens
        return proposals

    def store_draft_tokens(
        self,
        batch: InputBatch,
        drafts: list[list[int]],
        draft_buffer: torch.Tensor,
    ) -> None:
        async_upload = self.c1_async_draft_upload and batch.num_reqs == 1
        store_tensorfold_draft_tokens(
            drafts, batch.idx_mapping, draft_buffer, async_upload=async_upload
        )
        if async_upload and drafts[0]:
            state = self.requests[batch.req_ids[0]]
            if not state.reported_async_draft_upload:
                state.reported_async_draft_upload = True
                logger.info("TensorFold uploaded C1 draft tokens asynchronously.")

    def verify(
        self,
        batch: InputBatch,
        attn_metadata: dict[str, Any],
        layers: dict[str, Any],
        forward: Callable[[], Any],
        sample: Callable,
        positions: torch.Tensor,
        packed_forward: Callable[[torch.Tensor, torch.Tensor], Any],
        compute_logits: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[Any, tuple[SamplerOutput, torch.Tensor, torch.Tensor] | None]:
        self._committed_draft_input = None
        counts = batch.num_draft_tokens_per_req
        if (
            batch.has_prefill
            or batch.has_structured_output_reqs
            or counts is None
            or not batch.num_draft_tokens
            or batch.num_tokens != batch.num_tokens_after_padding
            or not (batch.num_scheduled_tokens == counts + 1).all()
            or any(req not in self.requests for req in batch.req_ids)
        ):
            output = forward()
            self._record_verification(batch, "linear")
            return output, None
        windows = []
        for req, count in zip(batch.req_ids, counts.tolist(), strict=True):
            if count == 0:
                windows.append(TreeWindow((-1,), (-1,)))
                continue
            entry = self.requests.get(req)
            if entry is None or entry.proposed is None:
                raise RuntimeError(
                    "Scheduler owns draft rows without a TensorFold tree."
                )
            windows.append(entry.proposed.prefix(count + 1))

        for name, meta in attn_metadata.items():
            layer = layers[name]
            if isinstance(meta, GDNAttentionMetadata):
                if (
                    not isinstance(layer, QwenGatedDeltaNetAttention)
                    or not layer.enable_fused_gdn_spec_decode
                    or layer.head_k_dim != 128
                    or layer.head_v_dim != 128
                    or layer.conv1d.bias is not None
                    or layer.norm.activation != "silu"
                    or layer.kv_cache[1].dtype != torch.float32
                    or meta.num_spec_decodes != batch.num_reqs
                    or meta.num_prefills
                    or meta.num_decodes
                ):
                    raise RuntimeError("Unsupported GDN state for TensorFold tree.")
            elif (
                not isinstance(meta, TritonAttentionMetadata)
                or meta.causal is not True
                or meta.rswa_window is not None
                or layer.impl.sliding_window != (-1, -1)
                or layer.impl.alibi_slopes is not None
                or layer.impl.logits_soft_cap
                or layer.impl.sinks is not None
            ):
                raise RuntimeError("TensorFold tree requires causal Triton Attention.")

        from tensorfold.cuda.kernels.gdn import to_device

        from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_attention import (
            C1_GRAPH_PREFIX_CAPACITY,
            CHUNK,
        )
        from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify import (
            TensorFoldVerify,
            device_verify_slots,
            read_verify_metadata,
        )

        observe_cost = not (self.c1_reference_curve and batch.num_reqs == 1)
        if observe_cost:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        slot_groups = {}
        if (
            (self.device_metadata or self.resident_direct_stage)
            and batch.num_reqs == 1
            and batch.num_tokens <= C1_GRAPH_ROWS
        ):
            prefix = int(batch.num_computed_tokens_np[0])
            if 0 <= prefix <= C1_GRAPH_PREFIX_CAPACITY:
                planned_rows = (
                    C1_GRAPH_ROWS if self.c1_fixed_width else batch.num_tokens
                )
                slot_groups = device_verify_slots(attn_metadata, planned_rows)
        if slot_groups:
            lengths = [prefix]
        else:
            sequence_lengths, slot_groups = read_verify_metadata(
                batch.seq_lens[: batch.num_reqs], attn_metadata
            )
            lengths = [
                total - len(tree.tokens)
                for total, tree in zip(sequence_lengths, windows, strict=True)
            ]
        fixed_c1 = (
            self.c1_fixed_width
            and len(windows) == 1
            and batch.num_tokens <= C1_GRAPH_ROWS
            and 0 <= lengths[0] <= C1_GRAPH_PREFIX_CAPACITY
        )
        extra_rows = C1_GRAPH_ROWS - batch.num_tokens if fixed_c1 else 0
        model_windows = [windows[0].pad(C1_GRAPH_ROWS)] if extra_rows else windows
        widths = [len(window.tokens) for window in model_windows]
        c8_chunks = (
            c8_attention_chunk_bucket(lengths, widths, CHUNK)
            if batch.num_reqs == 8
            and (self.c8_graph_buckets or self.verification_metrics is not None)
            else None
        )
        legacy_c8_chunks = c8_chunks if self.c8_graph_buckets else None
        batch_chunks = (
            batch_attention_chunk_bucket(lengths, widths, CHUNK)
            if self.batch_graph_buckets and legacy_c8_chunks is None
            else None
        )
        model_ids = batch.input_ids
        if extra_rows:
            model_ids = torch.nn.functional.pad(model_ids, (0, extra_rows))
        plan = TreeBatchPlan.build(model_windows)
        verification = None
        use_direct = False
        if self.c1_direct_stage or self.resident_direct_stage:
            device_slots = any(
                slots.device_ids is not None for slots in slot_groups.values()
            )
            use_direct = (
                self.resident_direct_stage if device_slots else self.c1_direct_stage
            )
        if (
            use_direct
            and len(windows) == 1
            and not extra_rows
            and 0 <= lengths[0] <= C1_GRAPH_PREFIX_CAPACITY
        ):
            verification = self._replay_direct_c1(
                windows[0],
                plan,
                lengths[0],
                attn_metadata,
                layers,
                slot_groups,
                model_ids,
                positions,
                self.requests[batch.req_ids[0]],
            )
        if verification is None:
            verify = TensorFoldVerify(
                model_windows,
                plan,
                lengths,
                attn_metadata,
                layers,
                self.device,
                slot_groups,
                c1_graph_bucket=self.c1_graph_bucket,
                c1_fixed_width=fixed_c1,
                fused_conv=self.fused_conv,
                round_fp8_tail=self.c1_fp8_tree_tail and len(windows) == 1,
                c8_graph_chunks=legacy_c8_chunks,
                batch_graph_chunks=batch_chunks,
            )
            root = to_device(plan.roots, torch.int64, self.device)
            depths = to_device(plan.depths, positions.dtype, self.device)
            verification = self._target_forward(
                verify,
                model_ids,
                positions.index_select(-1, root) + depths,
                packed_forward,
                compute_logits,
                self.requests[batch.req_ids[0]]
                if batch.num_reqs == 1 and (self.expand_c1_graph_bank or extra_rows)
                else None,
                batch.num_tokens if extra_rows else None,
            )
        verify, (hidden, aux, predicted), route = verification
        if verify.round_fp8_tail:
            request = self.requests[batch.req_ids[0]]
            if not request.reported_fp8_tree_tail:
                request.reported_fp8_tree_tail = True
                logger.info("TensorFold used cache-rounded C1 FP8 tree tail.")
        if (
            self.c1_qmm_split
            and batch.num_reqs == 1
            and model_ids.numel() == C1_GRAPH_ROWS
        ):
            request = self.requests[batch.req_ids[0]]
            if not request.reported_c1_qmm_split:
                request.reported_c1_qmm_split = True
                logger.info("TensorFold used tuned QMM splits for C1 verification.")
        if observe_cost:
            end.record()
        self._record_verification(
            batch, route, c8_chunks is not None, verify.is_c8_graph_bucket
        )
        if self.batch_bucket_metrics is not None and verify.is_batch_graph_bucket:
            self.batch_bucket_metrics.labels(batch.num_reqs, route).inc()
        batch_acceptance = (
            verify.accept_and_commit_batch(windows, predicted)
            if self.batch_device_commit and 2 <= batch.num_reqs <= 8
            else None
        )
        accepted = (
            verify.accept_and_commit(windows[0], predicted)
            if verify.commit_graph is not None and verify.commit_graph.device_accept
            else None
        )
        if batch_acceptance is not None:
            paths, emitted = batch_acceptance.paths, batch_acceptance.emitted
            source, dest = batch_acceptance.source, batch_acceptance.destination
            logger.info_once(
                "TensorFold committed GPU-selected paths for %d requests.",
                batch.num_reqs,
            )
            if batch_acceptance.replayed:
                logger.info_once(
                    "TensorFold replayed batched commit graph for %d requests.",
                    batch.num_reqs,
                )
        elif accepted is None:
            predictions = predicted.tolist()
            paths, emitted = zip(
                *(
                    tree.select(predictions[start : start + len(tree.tokens)])
                    for tree, start in zip(windows, plan.starts[:-1], strict=True)
                ),
                strict=True,
            )
            source, dest = verify.commit(
                list(paths), direct_recurrent_commit=self.recurrent_commit_coverage
            )
        else:
            path, tokens, source, dest, replayed = accepted
            paths, emitted = (path,), (tokens,)
            request = self.requests[batch.req_ids[0]]
            if (
                replayed
                and self.direct_recurrent_commit
                and not request.reported_direct_recurrent_commit
            ):
                request.reported_direct_recurrent_commit = True
                logger.info(
                    "TensorFold replayed C1 recurrent state directly to its slot."
                )
            if replayed and not request.reported_device_accept:
                request.reported_device_accept = True
                assert verify.commit_graph is not None
                logger.info(
                    "TensorFold replayed device C1 acceptance and state commit.%s%s",
                    " Device-selected C1 commit used a side stream."
                    if verify.commit_graph.pending
                    else "",
                    " C1 verification used device cache metadata."
                    if self.device_metadata and verify.resident_recurrent is not None
                    else "",
                )
        if self.recurrent_commit_coverage and len(paths) == 1:
            request = self.requests[batch.req_ids[0]]
            if not request.reported_recurrent_commit_coverage:
                request.reported_recurrent_commit_coverage = True
                logger.info("TensorFold used the C1 direct recurrent commit path.")
        if observe_cost and route != "capture":
            self.cost.observe(batch.num_tokens, start.elapsed_time(end))
        if any(tree.chose_sibling(path) for tree, path in zip(windows, paths)):
            logger.info_once("TensorFold Tree-DFlash2 accepted a sibling branch.")
        if verify.commit_graph is not None and verify.commit_graph.pending:
            self.pending_commit = verify.commit_graph
        if batch.num_reqs == 1 and batch.num_tokens > 16:
            logger.info_once(
                "TensorFold verified a wide C1 tree with %d rows.", batch.num_tokens
            )
        # Native vLLM postprocessing reads accepted rows as a contiguous prefix.
        for tensor in (batch.input_ids, hidden, *aux):
            tensor.index_copy_(0, dest, tensor.index_select(0, source))
        if counts.all():
            self._committed_draft_input = (
                batch,
                [
                    DraftInput(
                        [window.tokens[row] for row in path],
                        pos,
                        len(path),
                        tokens[-1],
                    )
                    for window, pos, path, tokens in zip(
                        windows, lengths, paths, emitted, strict=True
                    )
                ],
            )
        if batch_acceptance is not None:
            sampled_ids = batch_acceptance.sampled_ids
            sampled = batch_acceptance.counts
        else:
            width = max(map(len, emitted))
            sampled_ids = to_device(
                [token for row in emitted for token in row + [-1] * (width - len(row))],
                torch.int64,
                self.device,
            ).view(batch.num_reqs, width)
            sampled = to_device([len(p) for p in paths], torch.int32, self.device)
        scheduled = to_device(
            batch.num_scheduled_tokens.tolist(), torch.int32, self.device
        )
        rejected = scheduled - sampled
        if extra_rows:
            hidden = hidden[: batch.num_tokens]
            aux = [value[: batch.num_tokens] for value in aux]
        return (hidden, aux), (
            SamplerOutput(sampled_ids, None, None, sampled, rejected),
            sampled,
            rejected,
        )
