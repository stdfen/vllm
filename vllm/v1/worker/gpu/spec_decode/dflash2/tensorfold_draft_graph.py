# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture one TensorFold draft block with resident dynamic inputs."""

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from tensorfold.families.qwen3_5.cuda.dflash2 import DFlash2

logger = init_logger(__name__)


class C1DraftGraph:
    """Reuse one native draft shape across requests and rolling contexts."""

    def __init__(
        self, drafter: "DFlash2", max_nodes: int, *, pinned_readback: bool = False
    ):
        self.drafter = drafter
        self.length = min(drafter.block, max_nodes + 1)
        device = drafter.device
        self.inputs = torch.empty(
            2 + 3 * drafter.layers, dtype=torch.int64, device=device
        )
        end = 2 + 2 * drafter.layers
        self.tables = self.inputs[2:end].view(drafter.layers, 2)
        self.lengths = self.inputs[end:]
        self.relative = torch.arange(self.length, dtype=torch.float32, device=device)
        self.masks = torch.full(
            (self.length - 1,), drafter.mask_id, dtype=torch.int32, device=device
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self.outputs: tuple[torch.Tensor, torch.Tensor] | None = None
        self.host_outputs: tuple[torch.Tensor, ...] = ()
        self.readback_event = (
            torch.cuda.Event(blocking=True) if pinned_readback else None
        )

    def _run(self) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.cat((self.inputs[:1].int(), self.masks))
        positions = self.relative + self.inputs[1:2]
        return self.drafter.launch_c1_device(
            tokens, positions, self.tables, self.lengths.int()[:, None]
        )

    def launch(self, snap: tuple, pending: int, max_nodes: int):
        keys, values, length, position = snap
        if (
            max_nodes < 1
            or min(self.drafter.block, max_nodes + 1) != self.length
            or length < 1
            or len(keys) != self.drafter.layers
            or len(values) != self.drafter.layers
        ):
            return None

        pointers = []
        shape = (self.drafter.kv_local, length, self.drafter.head_dim)
        stream = torch.cuda.current_stream(self.drafter.device)
        for key, value in zip(keys, values, strict=True):
            for tensor in (key, value):
                if (
                    tensor is None
                    or tensor.shape != shape
                    or tensor.device != self.inputs.device
                    or tensor.dtype != torch.bfloat16
                    or not tensor.is_contiguous()
                ):
                    return None
                address = tensor.data_ptr()
                if address % 16:
                    return None
                tensor.record_stream(stream)
                pointers.append(address)
        packed = [pending, position, *pointers, *([length] * self.drafter.layers)]
        self.inputs.copy_(
            torch.tensor(packed, dtype=torch.int64).pin_memory(), non_blocking=True
        )
        if self.graph is None:
            capture_stream = torch.cuda.Stream(device=self.drafter.device)
            capture_stream.wait_stream(stream)
            with torch.cuda.stream(capture_stream):
                warmed = self._run()
                if self.readback_event is not None and all(
                    output.is_contiguous() for output in warmed
                ):
                    self.host_outputs = tuple(
                        torch.empty(
                            output.shape,
                            dtype=output.dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                        for output in warmed
                    )
                del warmed
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    self.outputs = self._run()
                    if self.host_outputs:
                        for host, output in zip(
                            self.host_outputs, self.outputs, strict=True
                        ):
                            host.copy_(output, non_blocking=True)
            stream.wait_stream(capture_stream)
            self.graph = graph
        else:
            logger.info_once("TensorFold replayed the C1 draft CUDA graph.")
        self.graph.replay()
        assert self.outputs is not None
        ids, floats = self.outputs
        host = None
        if self.host_outputs:
            assert self.readback_event is not None
            self.readback_event.record(stream)
            self.readback_event.synchronize()
            # The pinned buffers are reused by later replays. NumPy casts
            # produce the same independent arrays as TensorFold finish_tree.
            host_ids, host_scores = self.host_outputs
            host = (
                host_ids.numpy().astype(np.int64),
                host_scores.numpy().astype(np.float64),
            )
        return [ids, floats, host], 0, self.length - 1, pending
