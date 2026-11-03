# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.weight_staging import copy_loaded_weight
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from .config import Exl3Config


class Exl3LinearMethod(LinearMethodBase):
    """EXL3 linear parameter ABI.

    This class deliberately separates the checkpoint/TP ABI from the CUDA
    backend. The latter is attached in ``process_weights_after_loading`` once
    the upstream ExLlamaV3 extension has been built and validated.
    """

    def __init__(self, quant_config: Exl3Config, prefix: str) -> None:
        self.quant_config = quant_config
        self.prefix = prefix
        self.grouped_slices = quant_config.grouped_slices_for_prefix(prefix)
        self.bits = quant_config.bits_for_prefix(prefix)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        local_output = sum(output_partition_sizes)
        if self.grouped_slices:
            self._create_grouped_weights(
                layer,
                input_size_per_partition,
                local_output,
                input_size,
                output_size,
                extra_weight_attrs,
            )
            return
        _validate_tile_shape(input_size_per_partition, local_output, self.prefix)
        axis = _parallel_axis(
            input_size_per_partition,
            local_output,
            input_size,
            output_size,
        )
        layer._exl3_parallel_axis = axis
        layer._exl3_bits = self.bits

        loader = extra_weight_attrs.pop("weight_loader")
        trellis = Parameter(
            torch.empty(
                input_size_per_partition // 16,
                local_output // 16,
                self.bits * 16,
                dtype=torch.int16,
            ),
            requires_grad=False,
        )
        suh = Parameter(
            torch.empty(input_size_per_partition, dtype=torch.float16),
            requires_grad=False,
        )
        svh = Parameter(
            torch.empty(local_output, dtype=torch.float16), requires_grad=False
        )
        marker = Parameter(torch.zeros(1, dtype=torch.int32), requires_grad=False)

        _set_loader_attrs(trellis, loader, input_dim=0, output_dim=1, packed_dim=1)
        _set_loader_attrs(suh, loader, input_dim=0)
        _set_loader_attrs(svh, loader, output_dim=0)
        _set_loader_attrs(marker, _load_exl3_marker)
        for parameter in (trellis, suh, svh, marker):
            set_weight_attrs(parameter, extra_weight_attrs)

        layer.register_parameter("trellis", trellis)
        layer.register_parameter("suh", suh)
        layer.register_parameter("svh", svh)
        layer.register_parameter(self.quant_config.codebook, marker)

    def _create_grouped_weights(
        self,
        layer: torch.nn.Module,
        local_input: int,
        local_output: int,
        global_input: int,
        global_output: int,
        extra_weight_attrs: dict,
    ) -> None:
        group_count = len(self.grouped_slices)
        if local_input != global_input:
            raise ValueError("grouped EXL3 supports output-axis TP only")
        if global_output % group_count:
            raise ValueError(
                f"grouped EXL3 output {global_output} is not divisible by "
                f"{group_count} checkpoint slices"
            )
        output_per_group = global_output // group_count
        if local_output % output_per_group:
            raise ValueError(
                f"local grouped EXL3 output {local_output} is not divisible by "
                f"the per-group output {output_per_group}"
            )
        _validate_tile_shape(local_input, output_per_group, self.prefix)
        local_groups = local_output // output_per_group
        tp_rank = int(getattr(layer, "tp_rank", 0))
        group_start = tp_rank * local_groups
        if group_start + local_groups > group_count:
            raise ValueError("grouped EXL3 TP slice exceeds checkpoint groups")

        original_loader = extra_weight_attrs.pop("weight_loader")
        del original_loader
        loader = partial(
            _load_grouped_parameter,
            group_start=group_start,
            local_groups=local_groups,
        )
        shapes = {
            "trellis": (
                local_groups,
                local_input // 16,
                output_per_group // 16,
                self.bits * 16,
            ),
            "suh": (local_groups, local_input),
            "svh": (local_groups, output_per_group),
            self.quant_config.codebook: (local_groups,),
        }
        dtypes = {
            "trellis": torch.int16,
            "suh": torch.float16,
            "svh": torch.float16,
            self.quant_config.codebook: torch.int32,
        }
        for name, shape in shapes.items():
            parameter = Parameter(
                torch.empty(shape, dtype=dtypes[name]), requires_grad=False
            )
            set_weight_attrs(parameter, {"weight_loader": loader, **extra_weight_attrs})
            layer.register_parameter(name, parameter)
        layer._exl3_group_count = local_groups
        layer._exl3_group_start = group_start
        layer._exl3_group_output = output_per_group
        layer._exl3_bits = self.bits

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from .runtime import create_exl3_linear_backend

        group_count = getattr(layer, "_exl3_group_count", 0)
        if group_count:
            marker = getattr(layer, self.quant_config.codebook)
            layer._exl3_backends = tuple(
                create_exl3_linear_backend(
                    trellis=layer.trellis[index],
                    suh=layer.suh[index],
                    svh=layer.svh[index],
                    marker=marker[index],
                    bits=self.bits,
                    codebook=self.quant_config.codebook,
                )
                for index in range(group_count)
            )
            return
        layer._exl3_backend = create_exl3_linear_backend(
            trellis=layer.trellis,
            suh=layer.suh,
            svh=layer.svh,
            marker=getattr(layer, self.quant_config.codebook),
            bits=self.bits,
            codebook=self.quant_config.codebook,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        backends = getattr(layer, "_exl3_backends", None)
        if backends is not None:
            if x.ndim < 3 or x.shape[-2] != len(backends):
                raise ValueError(
                    "grouped EXL3 input must end in [local_groups, input_size]"
                )
            original_shape = x.shape
            grouped_input = x.reshape(-1, original_shape[-2], original_shape[-1])
            output = torch.stack(
                [
                    backend(grouped_input[:, index, :])
                    for index, backend in enumerate(backends)
                ],
                dim=1,
            )
            output = output.reshape(*original_shape[:-2], *output.shape[-2:])
            if bias is not None:
                output = output + bias.reshape(output.shape[-2:])
            return output
        backend = getattr(layer, "_exl3_backend", None)
        if backend is None:
            raise RuntimeError("EXL3 linear backend was not initialized")
        original_shape = x.shape
        output = backend(x.reshape(-1, original_shape[-1]))
        output = output.reshape(*original_shape[:-1], output.shape[-1])
        if bias is not None:
            output = output + bias
        return output


def _validate_tile_shape(input_size: int, output_size: int, prefix: str) -> None:
    if input_size % 16 or output_size % 16:
        raise ValueError(
            f"EXL3 requires 16x16-aligned local dimensions for {prefix!r}, "
            f"got input={input_size}, output={output_size}"
        )


def _parallel_axis(
    local_input: int,
    local_output: int,
    global_input: int,
    global_output: int,
) -> Literal["row", "column", "replicated"]:
    row_sharded = local_input != global_input
    column_sharded = local_output != global_output
    if row_sharded and column_sharded:
        raise ValueError("EXL3 linear cannot be sharded on both axes")
    if row_sharded:
        return "row"
    if column_sharded:
        return "column"
    return "replicated"


def _set_loader_attrs(
    parameter: Parameter,
    loader,
    *,
    input_dim: int | None = None,
    output_dim: int | None = None,
    packed_dim: int | None = None,
) -> None:
    attrs = {"weight_loader": loader}
    if input_dim is not None:
        attrs["input_dim"] = input_dim
    if output_dim is not None:
        attrs["output_dim"] = output_dim
    if packed_dim is not None:
        attrs.update({"packed_dim": packed_dim, "packed_factor": 16})
    set_weight_attrs(parameter, attrs)


def _load_grouped_parameter(
    param: Parameter,
    loaded_weight: torch.Tensor,
    slice_id: int | None = None,
    *,
    group_start: int,
    local_groups: int,
) -> bool:
    if slice_id is None:
        raise ValueError("grouped EXL3 checkpoint tensor requires a slice id")
    local_index = int(slice_id) - group_start
    if not 0 <= local_index < local_groups:
        return False
    target = param.data[local_index]
    if loaded_weight.ndim == 0 and target.ndim == 0:
        copy_loaded_weight(target, loaded_weight)
    else:
        if target.shape != loaded_weight.shape:
            raise ValueError(
                f"grouped EXL3 slice {slice_id} shape mismatch: "
                f"expected {tuple(target.shape)}, got {tuple(loaded_weight.shape)}"
            )
        copy_loaded_weight(target, loaded_weight)
    return True


def _load_exl3_marker(
    param: Parameter,
    loaded_weight: torch.Tensor,
    shard_id: int | str | None = None,
) -> None:
    """Load scalar codebook markers, validating fused shards agree."""
    del shard_id
    if loaded_weight.numel() != 1 or param.numel() != 1:
        raise ValueError(
            "EXL3 codebook marker must contain exactly one value, "
            f"got param={tuple(param.shape)}, loaded={tuple(loaded_weight.shape)}"
        )
    loaded = loaded_weight.reshape(1).to(device=param.device, dtype=param.dtype)
    if getattr(param, "_exl3_marker_loaded", False) and not torch.equal(param.data, loaded):
        raise ValueError(
            "fused EXL3 checkpoint shards use different codebook markers: "
            f"existing={param.item()}, loaded={loaded.item()}"
        )
    copy_loaded_weight(param.data, loaded)
    param._exl3_marker_loaded = True
