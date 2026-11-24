# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-forward packed tree verification for single-GPU dense Qwen."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch

from vllm.forward_context import get_forward_context, override_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    fused_mtp_decode_impl,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.dflash2.tree import DraftTree
from vllm.v1.worker.gpu.spec_decode.dflash2.tree_layout import (
    DEPTHS,
    LEVELS,
    PARENTS,
    WIDTH,
    ancestor_bits,
    commit_rows,
    conv_history_rows,
    pack_drafts,
    select_path,
)

logger = init_logger(__name__)
MAX_CONTEXT = 4096
MAX_REQUESTS = 8


@dataclass
class GDNInputs:
    state_slots: torch.Tensor
    accepted: torch.Tensor

    def source_slots(self) -> torch.Tensor:
        return self.state_slots.gather(1, self.accepted.long()[:, None] - 1)


class PackedBatch:
    """Fixed-address inputs shared by eager execution and graph capture."""

    def __init__(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        live_attn: dict[str, Any],
    ):
        self.num_reqs = tokens.numel() // WIDTH
        device = tokens.device
        self.tokens = torch.empty_like(tokens)
        self.positions = torch.empty_like(positions)
        self.conv_rows = torch.arange(tokens.numel(), dtype=torch.int32, device=device)
        offsets = torch.arange(self.num_reqs, device=device)[:, None] * WIDTH
        self.rope_rows = (offsets + torch.tensor(DEPTHS, device=device)).flatten()
        mask = torch.tensor(
            [ancestor_bits(i) for i in range(WIDTH)], dtype=torch.int32, device=device
        ).repeat(self.num_reqs)
        live_context = get_forward_context()
        self.layers = live_context.no_compile_layers
        self.gdn_inputs: dict[str, GDNInputs] = {}
        self.attentions: dict[str, TritonAttentionMetadata] = {}
        self.conv_indices: dict[int, torch.Tensor] = {}
        self.conv_records: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output: Any = None

        new_metadata = live_attn.copy()
        new_slot_mapping = live_context.slot_mapping.copy()
        for name, meta in live_attn.items():
            if isinstance(meta, TritonAttentionMetadata):
                static = replace(
                    meta,
                    query_start_loc=meta.query_start_loc.clone(),
                    seq_lens=meta.seq_lens.clone(),
                    block_table=meta.block_table.clone(),
                    slot_mapping=meta.slot_mapping.clone(),
                    max_seq_len=MAX_CONTEXT,
                    seq_threshold_3D=0,
                    tree_ancestor_bits=mask,
                )
                self.attentions[name] = static
                new_metadata[name] = static
                new_slot_mapping[name] = static.slot_mapping
            elif isinstance(meta, GDNAttentionMetadata):
                self.gdn_inputs[name] = GDNInputs(
                    meta.spec_state_indices_tensor.clone(),
                    meta.num_accepted_tokens.clone(),
                )
                hist = self.layers[name].conv1d.weight.shape[-1] - 1
                if hist not in self.conv_indices:
                    self.conv_indices[hist] = torch.tensor(
                        conv_history_rows(hist), dtype=torch.long, device=device
                    )

        self.context = replace(
            live_context,
            attn_metadata=new_metadata,
            slot_mapping=new_slot_mapping,
            additional_kwargs={**live_context.additional_kwargs, "tree_dflash2": self},
        )
        self.levels = []
        for nodes in LEVELS:
            node_ids = torch.tensor(nodes, dtype=torch.long, device=device)
            rows = (offsets + node_ids).flatten()
            parent_ids = torch.tensor(
                [max(0, PARENTS[i]) for i in nodes], dtype=torch.long, device=device
            )
            starts = torch.arange(rows.numel() + 1, dtype=torch.int32, device=device)
            accepted = torch.full_like(starts[:-1], 2)
            self.levels.append((node_ids, rows, parent_ids, starts, accepted))

    def stage(
        self,
        packed: list[list[int]],
        tokens: torch.Tensor,
        positions: torch.Tensor,
        metadata: dict[str, Any],
    ) -> None:
        self.tokens.view(self.num_reqs, WIDTH)[:, 0].copy_(tokens[::WIDTH])
        self.tokens.view(self.num_reqs, WIDTH)[:, 1:].copy_(
            torch.tensor(packed, dtype=tokens.dtype, device=tokens.device)
        )
        self.positions.copy_(positions.index_select(-1, self.rope_rows))
        for name, static in self.attentions.items():
            live = metadata[name]
            static.query_start_loc.copy_(live.query_start_loc)
            static.seq_lens.copy_(live.seq_lens)
            static.block_table.copy_(live.block_table)
            static.slot_mapping.copy_(live.slot_mapping)
        for name, inputs in self.gdn_inputs.items():
            live = metadata[name]
            inputs.state_slots.copy_(live.spec_state_indices_tensor)
            inputs.accepted.copy_(live.num_accepted_tokens)

    def gdn(
        self,
        layer: QwenGatedDeltaNetAttention,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        z: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        node_slots = self.gdn_inputs[layer.prefix]
        conv_state, recurrent = layer.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        weight = layer.conv1d.weight.flatten(1)
        history = weight.shape[1] - 1
        offsets = (
            node_slots.accepted.long()[:, None]
            - 1
            + torch.arange(history, device=mixed_qkv.device)
        )
        conv = conv_state.index_select(0, node_slots.state_slots[:, 0].long())
        prefix = conv.gather(2, offsets[:, None].expand(-1, conv.shape[1], -1))
        raw = mixed_qkv.contiguous().view(self.num_reqs, WIDTH, -1)
        self.conv_records[layer.prefix] = prefix, raw
        history_source = torch.cat([prefix.transpose(1, 2), raw], dim=1)
        history_input = (
            history_source[:, self.conv_indices[history]]
            .reshape(self.num_reqs * WIDTH, history, -1)
            .transpose(1, 2)
        )
        convolved = causal_conv1d_update(
            mixed_qkv.clone().contiguous(),
            history_input,
            weight,
            layer.conv1d.bias,
            layer.activation,
            conv_state_indices=self.conv_rows,
            null_block_id=-1,
        )
        for level, (nodes, rows, parents, starts, accepted) in enumerate(self.levels):
            destination = node_slots.state_slots.index_select(1, nodes)
            source = (
                node_slots.source_slots()
                if level == 0
                else node_slots.state_slots.index_select(1, parents)
            )
            pairs = torch.stack([destination.flatten(), source.flatten()], dim=1)
            result = output.new_empty((rows.numel(), *output.shape[1:]))
            fused_mtp_decode_impl()(
                mixed_qkv=convolved.index_select(0, rows).contiguous(),
                a=a.index_select(0, rows).contiguous(),
                b=b.index_select(0, rows).contiguous(),
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                state_indices=pairs,
                cu_seqlens=starts,
                num_accepted_tokens=accepted,
                state=recurrent,
                output_gate=z.index_select(0, rows).contiguous(),
                norm_weight=layer.norm.weight,
                out=result,
                scale=layer.head_k_dim**-0.5,
                norm_eps=layer.layer_norm_epsilon,
                output_gate_activation=layer.norm.activation,
            )
            output.index_copy_(0, rows, result)

    def forward(self, model: Callable[..., Any], use_graph: bool) -> Any:
        def run() -> Any:
            with override_forward_context(self.context):
                return model(self.tokens, self.positions)

        if not use_graph:
            output = run()
            logger.info_once(
                "Tree-DFlash2 packed eager forward executed for %d requests.",
                self.num_reqs,
            )
            return output
        if self.graph is None:
            graph = torch.cuda.CUDAGraph()
            current = torch.cuda.current_stream(self.tokens.device)
            side = torch.cuda.Stream(device=self.tokens.device)
            side.wait_stream(current)
            with torch.cuda.stream(side):
                saved = []
                for name, inputs in self.gdn_inputs.items():
                    recurrent = self.layers[name].kv_cache[1]
                    sources = inputs.source_slots().flatten().long()
                    saved.append(
                        (recurrent, sources, recurrent.index_select(0, sources))
                    )
                run()
                for recurrent, sources, values in saved:
                    recurrent.index_copy_(0, sources, values)
                with torch.cuda.graph(graph, stream=side):
                    self.output = run()
                for recurrent, sources, values in saved:
                    recurrent.index_copy_(0, sources, values)
            current.wait_stream(side)
            self.graph = graph
            logger.info(
                "Tree-DFlash2 captured a packed CUDA graph for %d requests.",
                self.num_reqs,
            )
        self.graph.replay()
        logger.info_once(
            "Tree-DFlash2 packed CUDA graph replay executed for %d requests.",
            self.num_reqs,
        )
        return self.output

    def commit(self, paths: list[list[int]]) -> torch.Tensor:
        device = self.tokens.device
        # Gather every source before writing overlapping destination slots.
        sources, targets = commit_rows(paths)
        src = torch.tensor(sources, dtype=torch.long, device=device)
        dst = torch.tensor(targets, dtype=torch.long, device=device)
        for name, meta in self.attentions.items():
            kv = self.layers[name].kv_cache.view(torch.uint8).transpose(1, 2)
            block_size = kv.shape[1]
            source_slots = meta.slot_mapping.index_select(0, src).long()
            target_slots = meta.slot_mapping.index_select(0, dst).long()
            values = kv[source_slots // block_size, source_slots % block_size]
            kv[target_slots // block_size, target_slots % block_size] = values
        last_nodes = torch.tensor([path[-1] for path in paths], device=device)
        chain_slots = torch.tensor([len(path) - 1 for path in paths], device=device)
        for name, inputs in self.gdn_inputs.items():
            conv, recurrent = self.layers[name].kv_cache
            if not is_conv_state_dim_first():
                conv = conv.transpose(-1, -2)
            prefix, raw = self.conv_records[name]
            source_slots = (
                inputs.state_slots.gather(1, last_nodes[:, None]).flatten().long()
            )
            target_slots = (
                inputs.state_slots.gather(1, chain_slots[:, None]).flatten().long()
            )
            recurrent.index_copy_(
                0, target_slots, recurrent.index_select(0, source_slots)
            )
            for req, path in enumerate(paths):
                history = torch.cat(
                    [prefix[req, :, 1:], raw[req, path].transpose(0, 1)], dim=-1
                )
                conv[
                    inputs.state_slots[req : req + 1, 0].long(), :, : history.shape[-1]
                ] = history.to(conv.dtype).unsqueeze(0)
        permutation = torch.arange(self.tokens.numel(), device=device)
        permutation.index_copy_(0, dst, src)
        return permutation


class PackedTreeVerifier:
    def __init__(self, use_graph: bool):
        self.use_graph = use_graph
        self.batches: dict[int, PackedBatch] = {}

    def verify(
        self,
        trees: list[DraftTree | None],
        batch: InputBatch,
        positions: torch.Tensor,
        metadata: dict[str, Any],
        model: Callable[..., Any],
        compute_logits: Callable[[torch.Tensor], torch.Tensor],
        draft_layers: set[str],
    ) -> tuple[Any, tuple[SamplerOutput, torch.Tensor, torch.Tensor]] | None:
        count = batch.num_reqs
        if (
            count > MAX_REQUESTS
            or batch.num_tokens_after_padding != count * WIDTH
            or not (batch.num_scheduled_tokens == WIDTH).all()
            or batch.seq_lens_cpu_upper_bound[:count].max() > MAX_CONTEXT
            or any(tree is None for tree in trees)
        ):
            return None
        metadata = {
            name: meta for name, meta in metadata.items() if name not in draft_layers
        }
        layers = get_forward_context().no_compile_layers
        checked_slots: set[int] = set()
        for name, meta in metadata.items():
            if isinstance(meta, GDNAttentionMetadata):
                layer = layers[name]
                if (
                    not isinstance(layer, QwenGatedDeltaNetAttention)
                    or not layer._can_use_fused_gdn_mtp_decode(meta)
                    or meta.num_spec_decodes != count
                    or meta.num_prefills
                    or meta.spec_state_indices_tensor is None
                    or meta.spec_state_indices_tensor.shape != (count, WIDTH)
                    or meta.num_accepted_tokens is None
                ):
                    return None
                if layer.kv_cache[1].dtype != torch.float32:
                    raise RuntimeError("Packed Tree-DFlash2 requires FP32 GDN state")
                if id(meta) not in checked_slots:
                    count_invalid = (meta.num_accepted_tokens < 1) | (
                        meta.num_accepted_tokens > WIDTH
                    )
                    if bool(
                        count_invalid.any()
                        | (meta.spec_state_indices_tensor <= 0).any()
                    ):
                        return None
                    slots = meta.spec_state_indices_tensor
                    if slots.unique().numel() != slots.numel():
                        return None
                    checked_slots.add(id(meta))
            elif (
                not isinstance(meta, TritonAttentionMetadata)
                or meta.causal is not True
                or meta.rswa_window is not None
                or layers[name].impl.sliding_window != (-1, -1)
            ):
                return None
        tokens = batch.input_ids.view(count, WIDTH).tolist()
        packed = [
            pack_drafts(row[1:], tree) if tree is not None else None
            for row, tree in zip(tokens, trees, strict=True)
        ]
        if any(row is None for row in packed):
            return None
        packed = [row for row in packed if row is not None]
        workspace = self.batches.get(count)
        if workspace is None:
            workspace = PackedBatch(batch.input_ids, positions, metadata)
            if not workspace.attentions or not workspace.gdn_inputs:
                raise RuntimeError(
                    "Packed Tree-DFlash2 needs Triton Attention and CUDA GDN"
                )
            self.batches[count] = workspace
        workspace.stage(packed, batch.input_ids, positions, metadata)
        output = workspace.forward(model, self.use_graph)
        hidden = output[0] if isinstance(output, tuple) else output
        logits = compute_logits(hidden)
        predicted = logits.argmax(dim=-1).view(count, WIDTH).tolist()
        selected = [
            select_path([row[0], *draft], target)
            for row, draft, target in zip(tokens, packed, predicted, strict=True)
        ]
        paths = [path for path, _ in selected]
        if any(len(path) > 1 and path[1] == 2 for path in paths):
            logger.info_once("Tree-DFlash2 packed verification accepted a sibling.")
        permutation = workspace.commit(paths)
        hidden = hidden.index_select(0, permutation)
        if isinstance(output, tuple):
            output = hidden, [aux.index_select(0, permutation) for aux in output[1]]
        else:
            output = hidden
        emitted = [row + [-1] * (WIDTH - len(row)) for _, row in selected]
        device = hidden.device
        ids = torch.tensor(emitted, dtype=torch.int64, device=device)
        sampled = torch.tensor(
            [len(path) for path in paths], dtype=torch.int32, device=device
        )
        rejected = WIDTH - sampled
        return output, (
            SamplerOutput(ids, None, None, sampled, rejected),
            sampled,
            rejected,
        )
