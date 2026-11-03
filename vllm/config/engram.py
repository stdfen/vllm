# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Self

from pydantic import model_validator

from vllm.config.utils import config, get_hash_factors, hash_factors

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig

# Architecture -> the hf_text_config field naming its n-gram layers. A model is
# only configurable here if it actually has such layers to store.
_NGRAM_LAYER_FIELDS = {
    "DeepseekV41ForCausalLM": "engram_layer_ids",
    "Qwen4ExpForCausalLM": "ple_layer_ids",
    "Qwen4ExpForConditionalGeneration": "ple_layer_ids",
}


def model_has_engram_layers(model_config: "ModelConfig | None") -> bool:
    """Whether the model carries n-gram embedding layers."""
    if model_config is None:
        return False
    field = _NGRAM_LAYER_FIELDS.get(model_config.architecture)
    if field is None:
        return False
    return bool(getattr(model_config.hf_text_config, field, None))


@config
class EngramConfig:
    """Configuration for Engram embedding storage and sharding."""

    cpu_offload: bool = True
    """Store embedding weights in pinned CPU memory for UVA lookup."""

    embedding_across_dp: bool = False
    """Shard embeddings across TP and all DP ranks when enabled.
    Otherwise, each DP rank has a separate TP-sharded embedding replica."""

    dp_shared_memory: bool | None = None
    """Share CPU-offloaded embedding weights between co-located
    DP replicas. Each node stores one copy of every TP shard, reducing host
    memory without per-step Engram DP collectives. Requires sufficient
    /dev/shm capacity and a shared IPC namespace. Defaults to enabled whenever
    the other settings allow it, falling back to per-replica tables when DP
    replicas are not co-located on one node or /dev/shm cannot hold them."""

    use_thp: bool = False
    """Back private CPU-offloaded tables with transparent huge pages (best
    effort, falls back to ordinary pinned pages). Prefaulting the tables at
    startup takes longer. Requires cpu_offload without dp_shared_memory."""

    file_backed: bool = False
    """Read only the requested Engram rows from a separate safetensors model.

    This avoids allocating or pinning the full table and is intended for
    checkpoints whose quantized model omits the native Engram embedding files.
    """

    table_dir: str | None = None
    """Local directory containing the native Engram safetensors and index."""

    disk_offload: bool = False
    """Read table rows from this checkpoint's safetensors shards on demand."""

    disk_threads: int = 32
    """Reader threads per Engram layer for disk_offload."""

    disk_cache_rows: int = 0
    """Rows per Engram layer retained in the disk reader's hot-row cache."""

    @model_validator(mode="after")
    def _validate_shared_memory(self) -> Self:
        if self.file_backed and self.disk_offload:
            raise ValueError("file_backed and disk_offload are mutually exclusive")
        if self.file_backed:
            if not self.cpu_offload:
                raise ValueError("file_backed Engram requires cpu_offload=True")
            if self.dp_shared_memory:
                raise ValueError("file_backed Engram does not use dp_shared_memory")
            if self.use_thp:
                raise ValueError("file_backed Engram does not use THP")
            if not self.table_dir:
                raise ValueError("file_backed Engram requires table_dir")
            self.dp_shared_memory = False
        if self.dp_shared_memory and not self.cpu_offload:
            raise ValueError("dp_shared_memory requires cpu_offload=True")
        if self.use_thp and (not self.cpu_offload or self.dp_shared_memory):
            raise ValueError(
                "use_thp requires cpu_offload=True and dp_shared_memory=False"
            )
        if self.disk_offload and (
            not self.cpu_offload
            or self.dp_shared_memory
            or self.use_thp
            or self.embedding_across_dp
        ):
            raise ValueError(
                "disk_offload requires cpu_offload=True, dp_shared_memory=False, "
                "use_thp=False and embedding_across_dp=False"
            )
        if self.disk_threads < 1 or self.disk_cache_rows < 0:
            raise ValueError("disk_threads must be >= 1 and disk_cache_rows >= 0")
        return self

    def verify_model_config(self, model_config: "ModelConfig | None") -> None:
        """Reject Engram configuration for models without n-gram embeddings."""
        field = (
            _NGRAM_LAYER_FIELDS.get(model_config.architecture)
            if model_config is not None
            else None
        )
        if (
            model_config is None
            or field is None
            or not getattr(model_config.hf_text_config, field, None)
        ):
            raise ValueError(
                "EngramConfig requires a model with supported Engram "
                "embeddings and non-empty n-gram layer ids."
            )

    def resolve_dp_shared_memory(self, parallel_config: "ParallelConfig") -> None:
        """Share host tables by default wherever the configuration permits."""
        if self.file_backed:
            self.dp_shared_memory = False
        elif self.dp_shared_memory is None:
            self.dp_shared_memory = (
                self.cpu_offload
                and not self.use_thp
                and not self.disk_offload
                and parallel_config.data_parallel_size > 1
                and not parallel_config.enable_elastic_ep
            )

    def verify_parallel_config(self, parallel_config: "ParallelConfig") -> None:
        """Reject unsupported embedding parallel topologies."""
        if self.dp_shared_memory:
            if parallel_config.data_parallel_size <= 1:
                raise ValueError("dp_shared_memory requires data_parallel_size > 1.")
            if parallel_config.enable_elastic_ep:
                raise ValueError("dp_shared_memory is not supported with elastic EP.")
        if (
            self.embedding_across_dp
            and parallel_config.data_parallel_size > 1
            and parallel_config.enable_elastic_ep
        ):
            raise ValueError(
                "Engram embedding_across_dp is not supported with elastic EP yet."
            )

    def get_parallel_size(self, parallel_config: "ParallelConfig") -> int:
        """Derive the embedding group size from the parallel configuration."""
        size = parallel_config.tensor_parallel_size
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1:
            size *= parallel_config.data_parallel_size
        return size

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, set()))
