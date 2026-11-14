# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility reader for the original checkpoint's Engram tables.

The native host callback executes on every CUDA graph replay. This module is
part of the common A/B compatibility base, not the P0 read-path optimization.
"""

import importlib
import json
from pathlib import Path

import torch

from vllm.config import get_current_vllm_config
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

_EXTENSION = None


def resolve_source(layer_id: int) -> tuple[str, int, str, int]:
    model = get_current_vllm_config().model_config
    root = Path(model.model)
    if not root.is_dir():
        from vllm.transformers_utils.repo_utils import hf_api

        root = Path(
            hf_api().snapshot_download(
                repo_id=model.model, revision=model.revision, local_files_only=True
            )
        )
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    source: list[tuple[str, int]] = []
    for suffix in ("weight", "scale"):
        name = f"layers.{layer_id}.engram.embed.{suffix}"
        path = root / index[name]
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"Truncated safetensors header: {path}")
            size = int.from_bytes(raw, "little")
            if size > 128 << 20:
                raise ValueError(f"Invalid safetensors header length: {path}")
            header = json.loads(stream.read(size))
        start, end = header[name]["data_offsets"]
        if not 0 <= start <= end <= path.stat().st_size - 8 - size:
            raise ValueError(f"Invalid Engram tensor range: {path}:{name}")
        source.append((str(path), 8 + size + start))
    return (*source[0], *source[1])


def load_extension():
    global _EXTENSION
    if _EXTENSION is None:
        try:
            _EXTENSION = importlib.import_module("vllm_engram_disk_v033")
            return _EXTENSION
        except ModuleNotFoundError as exc:
            if exc.name != "vllm_engram_disk_v033":
                raise
        from torch.utils.cpp_extension import load

        _EXTENSION = load(
            name="vllm_engram_disk_v033",
            sources=[str(Path(__file__).with_name("engram_disk.cpp"))],
            extra_cflags=["-O2"],
            with_cuda=True,
            verbose=False,
        )
    return _EXTENSION


def skip_table_load(param, loaded_weight):
    """The checkpoint remains the storage; do not materialize its table."""


class CheckpointDiskReader:
    def __init__(self, source, row_start, rows, dim, block_size, threads):
        self.dim, self.block_size = dim, block_size
        self.native = load_extension().EngramDiskTable(
            *source,
            row_start,
            rows,
            dim,
            dim // block_size,
            0,
            threads,
        )
        self.views = None
        self.recorded = False

    def allocate_staging(self, max_rows):
        self.ids = torch.empty(
            max_rows, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.weight = torch.empty(
            max_rows, self.dim, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        self.scale = torch.empty(
            max_rows,
            self.dim // self.block_size,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        self.row_ids = torch.arange(max_rows, dtype=torch.int64)
        self.destinations = torch.arange(max_rows, dtype=torch.int32)
        self.done = torch.cuda.Event()

    def lookup(self, table, indices, out, background):
        from vllm.models.deepseek_v41.common.engram import _engram_lookup_kernel

        rows = indices.shape[0] * table.part_n_hash_cols
        if rows == 0:
            return
        if rows > self.ids.shape[0]:
            raise ValueError("Engram disk staging capacity exceeded")
        stream = torch.cuda.current_stream()
        capturing = torch.cuda.is_current_stream_capturing()
        if self.recorded and not capturing:
            stream.wait_event(self.done)
        local = indices[:, table.head_start : table.head_start + table.part_n_hash_cols]
        local = local.to(torch.int64) - table.vocab_start_idx
        if local.shape[1] < table.part_n_hash_cols:
            local = torch.nn.functional.pad(
                local,
                (0, table.part_n_hash_cols - local.shape[1]),
                value=-1,
            )
        self.ids[:rows].copy_(local.reshape(-1), non_blocking=True)
        self.enqueue(stream, rows, capturing)
        if self.views is None:
            self.views = (
                get_accelerator_view_from_cpu_tensor(
                    self.weight.view(torch.float8_e4m3fn)
                ),
                get_accelerator_view_from_cpu_tensor(self.scale),
            )
        grid = min(
            (rows + 15) // 16,
            max(1, table._num_sms // 2 if background else table._num_sms),
        )
        _engram_lookup_kernel[(grid,)](
            *self.views,
            self.row_ids,
            self.destinations,
            out,
            0,
            rows,
            rows,
            1,
            1,
            HEAD_START=table.head_start,
            LOCAL_HEADS=table.part_n_hash_cols,
            TOTAL_HEADS=table.n_hash_cols,
            DIM=table.dim,
            QUANT_BLOCK=table.block_size,
            BLOCK_R=16,
            GRID=grid,
            SORTED=True,
            PACKED=out.dtype == torch.uint8,
            OUT_STRIDE=out.stride(0),
            PEER_ROWS=0,
        )
        if not capturing:
            self.done.record(stream)
            self.recorded = True

    def enqueue(self, stream, rows, capturing):
        self.native.enqueue_read(
            stream.cuda_stream,
            self.ids.data_ptr(),
            self.weight.data_ptr(),
            self.scale.data_ptr(),
            rows,
        )
