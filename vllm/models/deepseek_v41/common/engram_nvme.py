# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded direct I/O over validated rank-local packs; keeps vLLM's lookup kernel."""

import importlib
import json
import threading
import time
from pathlib import Path

import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.models.deepseek_v41.common.engram_disk import CheckpointDiskReader
from vllm.models.deepseek_v41.common.engram_pack import select_pack
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)
_EXTENSION = None


def load_extension():
    global _EXTENSION
    if _EXTENSION is None:
        try:
            _EXTENSION = importlib.import_module("vllm_engram_pack_v033")
        except ModuleNotFoundError as exc:
            if exc.name != "vllm_engram_pack_v033":
                raise
            from torch.utils.cpp_extension import load

            _EXTENSION = load(
                name="vllm_engram_pack_v033",
                sources=[str(Path(__file__).with_suffix(".cpp"))],
                extra_cflags=["-O2"],
                with_cuda=True,
                verbose=False,
            )
    return _EXTENSION


def record_stats(native, path):
    """Only the native reader counts graph replays; Python capture is not a read."""
    with path.open("a", buffering=1) as stream:
        while True:
            stream.write(
                json.dumps(dict(unix_time=time.time(), **native.stats())) + "\n"
            )
            time.sleep(10)


class PackedDiskReader(CheckpointDiskReader):
    @classmethod
    def from_config(cls, layer, row_start, rows, dim, block_size, threads):
        config = get_current_vllm_config()
        options = config.engram_config
        if options is None or options.disk_pack_manifest is None:
            raise ValueError("Packed Engram reader requires manifest configuration")
        rank = get_tensor_model_parallel_rank()
        path, record = select_pack(
            options.disk_pack_manifest,
            config.model_config.model,
            layer,
            config.parallel_config.tensor_parallel_size,
            rank,
            row_start,
            rows,
            dim,
            dim // block_size,
            options.disk_verify,
        )
        stats_path = None
        if options.disk_stats_dir:
            stats_path = Path(options.disk_stats_dir) / f"engram-r{rank}-l{layer}.jsonl"
        logger.info(
            "P0 Engram rank=%d layer=%d rows=%d stride=%d verification=%s "
            "cache_limit=%d: %s",
            rank,
            layer,
            rows,
            record["stride"],
            options.disk_verify,
            options.disk_cache_bytes,
            path,
        )
        return cls(
            path,
            record,
            block_size,
            options.disk_pack_threads or threads,
            options.disk_cache_bytes,
            stats_path,
            direct_io=options.disk_pack_io == "direct",
        )

    def __init__(
        self,
        path,
        record,
        block_size,
        threads,
        cache_bytes=0,
        stats_path=None,
        *,
        direct_io=True,
    ):
        self.dim, self.block_size = record["dim"], block_size
        self.cache_bytes = cache_bytes
        self.native = load_extension().PackReader(
            str(path),
            record["rows"],
            self.dim,
            self.dim // block_size,
            record["stride"],
            threads,
            direct_io,
        )
        self.views, self.recorded = None, False
        self.stats_path = stats_path
        if stats_path:
            stats_path.parent.mkdir(parents=True, exist_ok=True)

    def allocate_staging(self, max_rows):
        super().allocate_staging(max_rows)
        self.native.allocate(max_rows, self.cache_bytes)
        self.status = torch.zeros(1, dtype=torch.int32, device="cpu", pin_memory=True)
        self.status_view = get_accelerator_view_from_cpu_tensor(self.status)
        logger.info("P0 Engram bounded allocations: %s", self.native.stats())
        if self.stats_path:
            threading.Thread(
                target=record_stats, args=(self.native, self.stats_path), daemon=True
            ).start()

    def enqueue(self, stream, rows, capturing):
        self.native.enqueue_read(
            stream.cuda_stream,
            self.ids.data_ptr(),
            self.weight.data_ptr(),
            self.scale.data_ptr(),
            rows,
            self.status.data_ptr(),
            capturing,
        )
        # This device assertion is captured and runs after EVERY host-node replay.
        # I/O errors fail the worker; returning silently zero-filled rows is unsafe.
        torch._assert_async(
            self.status_view[0].eq(0),
            "Engram NVMe short read/EIO; restart the failed worker",
        )
