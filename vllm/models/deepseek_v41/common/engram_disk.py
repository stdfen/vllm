import json
import os

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.triton_utils import triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

def _safetensors_tensor_offset(path: str, name: str) -> int:
    """Absolute byte offset of tensor `name` inside a safetensors file."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    return 8 + header_len + header[name]["data_offsets"][0]


def resolve_disk_source(layer_id: int) -> tuple[str, int, str, int]:
    """(weight file, offset, scale file, offset) of a layer's Engram table in
    the checkpoint, for EngramConfig.disk_offload."""
    config = get_current_vllm_config()
    model_config = config.model_config
    model_dir = config.engram_config.table_dir or model_config.model
    if not os.path.isdir(model_dir):
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(
            model_dir, revision=model_config.revision, local_files_only=True
        )
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    source: list = []
    for suffix in ("weight", "scale"):
        name = f"layers.{layer_id}.engram.embed.{suffix}"
        path = os.path.join(model_dir, weight_map[name])
        source += [path, _safetensors_tensor_offset(path, name)]
    return tuple(source)  # type: ignore[return-value]


_ENGRAM_DISK_EXT = None


def _load_engram_disk_ext():
    """The host-side row reader (engram_disk.cpp), JIT-built once per env."""
    global _ENGRAM_DISK_EXT
    if _ENGRAM_DISK_EXT is None:
        from torch.utils.cpp_extension import load

        _ENGRAM_DISK_EXT = load(
            name="vllm_engram_disk",
            sources=[os.path.join(os.path.dirname(__file__), "engram_disk.cpp")],
            extra_cflags=["-O2"],
            with_cuda=True,
            verbose=False,
        )
    return _ENGRAM_DISK_EXT


def disk_weight_loader(
    param: torch.nn.Parameter, loaded_weight: torch.Tensor
) -> None:
    """Disk-offloaded tables are read from the shards at lookup time."""


class DiskEngramMixin:
    def _init_disk_reader(self, source, threads: int, cache_rows: int) -> None:
        self._disk = None
        self._disk_views = None
        self._disk_recorded = False
        if source is None:
            return
        weight_file, weight_offset, scale_file, scale_offset = source
        self._disk = _load_engram_disk_ext().EngramDiskTable(
            weight_file, weight_offset, scale_file, scale_offset,
            self.vocab_start_idx, self.part_num_embeddings,
            self.dim, self.dim // self.block_size, cache_rows, threads,
        )

    def init_disk_staging(self, max_rows: int) -> None:
        """Allocate address-stable pinned buffers for disk-backed lookups."""
        if not self.disk_offload:
            return
        scale_dim = self.dim // self.block_size
        self._disk_ids = torch.empty(
            max_rows, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._disk_w = torch.empty(
            max_rows, self.dim, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        self._disk_s = torch.empty(
            max_rows, scale_dim, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        self._disk_rows = torch.arange(max_rows, dtype=torch.int64, device="cuda")
        self._disk_dst = torch.arange(max_rows, dtype=torch.int32, device="cuda")
        self._disk_done = torch.cuda.Event()

    def _disk_lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool, kernel
    ) -> None:
        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        assert self._disk is not None
        assert rows <= self._disk_ids.shape[0], "engram disk staging too small"
        stream = torch.cuda.current_stream()
        capturing = torch.cuda.is_current_stream_capturing()
        if self._disk_recorded and not capturing:
            stream.wait_event(self._disk_done)
        head_end = self.head_start + self.part_n_hash_cols
        local = indices[:, self.head_start : head_end].to(torch.int64)
        local = local - self.vocab_start_idx
        if (pad := self.part_n_hash_cols - local.shape[1]) > 0:
            local = nn.functional.pad(local, (0, pad), value=-1)
        self._disk_ids[:rows].copy_(local.reshape(-1), non_blocking=True)
        self._disk.enqueue_read(
            stream.cuda_stream,
            self._disk_ids.data_ptr(),
            self._disk_w.data_ptr(),
            self._disk_s.data_ptr(),
            rows,
        )
        if self._disk_views is None:
            self._disk_views = (
                get_accelerator_view_from_cpu_tensor(
                    self._disk_w.view(torch.float8_e4m3fn)
                ),
                get_accelerator_view_from_cpu_tensor(self._disk_s),
            )
        weight, scales = self._disk_views
        tiles = triton.cdiv(rows, 16)
        grid = min(tiles, self._num_sms // 2 if background else self._num_sms)
        kernel[(grid,)](
            weight,
            scales,
            self._disk_rows,
            self._disk_dst,
            out,
            0,
            rows,
            rows,
            1,
            1,
            HEAD_START=self.head_start,
            LOCAL_HEADS=self.part_n_hash_cols,
            TOTAL_HEADS=self.n_hash_cols,
            DIM=self.dim,
            QUANT_BLOCK=self.block_size,
            BLOCK_R=16,
            GRID=grid,
            SORTED=True,
            PACKED=out.dtype == torch.uint8,
            OUT_STRIDE=out.stride(0),
        )
        if not capturing:
            self._disk_done.record(stream)
            self._disk_recorded = True
