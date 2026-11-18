# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped MXFP8 projection on SM12x (GB10, RTX 50xx): DeepSeek-V4.1's wo_a.

DeepGEMM's MXFP8 bmm only runs on datacenter Blackwell, so on SM12x wo_a fell back to EmulationMxfp8LinearKernel, which
dequantizes the weight to BF16 once at load and runs a BF16 bmm: twice the bytes and memory, and cuBLAS BF16 at one row
is slow on GB10 (137 us per 1024 x 4096 group at m=1, measured). This kernel keeps the FP8 weight and runs one Marlin
MXFP8 (W8A16) GEMM per group, 26 us per group at m=1 on GB10 (the CUTLASS MXFP8 GEMM is 151 us there).
deep_gemm_fp8_o_proj routes to it when wo_a carries `sm12x_bmm_groups`.
"""

import os

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import MXFP8_BLOCK_SIZE
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


def _group_shape(c: Mxfp8LinearLayerConfig) -> tuple[int, int, int]:
    rows, k = c.weight_shape
    g = c.bmm_batch_size or 0
    return g, (rows // g if g and rows % g == 0 else 0), k


class Sm12xMarlinMxfp8BmmLinearKernel(Mxfp8LinearKernel):
    """MXFP8 bmm as one Marlin MXFP8 (W8A16) GEMM per group (SM12x)."""

    supports_pre_processed_weights = False

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            is_fp8_marlin_supported,
        )

        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
        ):
            return False, "SM12x only (datacenter Blackwell uses DeepGEMM's bmm)"
        if not is_fp8_marlin_supported():
            return False, "Marlin FP8 not available"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        g, n, k = _group_shape(c)
        if g <= 0 or n <= 0:
            return False, "bmm with rows divisible by the batch size only"
        if k % MXFP8_BLOCK_SIZE != 0:
            return False, f"Marlin MXFP8 needs K % {MXFP8_BLOCK_SIZE} == 0, got K={k}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_mxfp8_layer_for_marlin,
        )

        weight = layer.weight.data  # [G * N, K] fp8
        g = self.config.bmm_batch_size
        rows, k = weight.shape
        n = rows // g
        scale = layer.weight_scale.data[:rows, : k // MXFP8_BLOCK_SIZE]
        packed, scales = [], []
        for i in range(g):
            holder = torch.nn.Module()
            holder.weight = Parameter(weight[i * n : (i + 1) * n].contiguous(), requires_grad=False)
            holder.weight_scale = Parameter(scale[i * n : (i + 1) * n].contiguous(), requires_grad=False)
            holder.output_size_per_partition, holder.input_size_per_partition = n, k
            prepare_mxfp8_layer_for_marlin(holder)
            packed.append(holder.weight.data)
            scales.append(holder.weight_scale.data)
        layer.weight = Parameter(torch.stack(packed), requires_grad=False)
        layer.weight_scale = Parameter(torch.stack(scales), requires_grad=False)
        layer.sm12x_bmm_groups = g
        layer.sm12x_bmm_nk = (n, k)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[tokens, groups, K] -> [tokens, groups, N]."""
        n, k = layer.sm12x_bmm_nk
        out = sm12x_marlin_bmm(x, layer.weight, layer.weight_scale, layer.sm12x_bmm_groups, n, k)
        if bias is not None:
            out = out + bias.view(out.shape[1:])
        return out


_STREAMS_MAX_TOKENS = 16
_side_streams: dict[int, list[torch.cuda.Stream]] = {}


def _sm12x_marlin_bmm_streams(x, weight, scales, groups, n, k, apply):
    device = x.device.index if x.device.index is not None else torch.cuda.current_device()
    side = _side_streams.setdefault(device, [])
    while len(side) < groups - 1:
        side.append(torch.cuda.Stream(device=x.device))
    side = side[: groups - 1]
    cur = torch.cuda.current_stream(x.device)
    for stream in side:
        stream.wait_stream(cur)
    outs = []
    for i in range(groups):
        stream = cur if i == 0 else side[i - 1]
        with torch.cuda.stream(stream):
            outs.append(apply(input=x[:, i, :].contiguous(), weight=weight[i],
                              weight_scale=scales[i], workspace=None,
                              size_n=n, size_k=k))
    for stream in side:
        cur.wait_stream(stream)
    return torch.stack(outs, dim=1)


def sm12x_marlin_bmm(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, groups: int, n: int, k: int
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_mxfp8_marlin_linear,
    )

    tokens, g, _ = x.shape
    assert g == groups, (g, groups)
    if (os.environ.get("VLLM_SM12X_BMM_STREAMS", "0") == "1"
            and 1 < groups and tokens <= _STREAMS_MAX_TOKENS and x.is_cuda):
        return _sm12x_marlin_bmm_streams(
            x, weight, scales, groups, n, k, apply_mxfp8_marlin_linear
        )
    out = torch.empty((tokens, groups, n), dtype=x.dtype, device=x.device)
    for i in range(groups):
        out[:, i, :] = apply_mxfp8_marlin_linear(
            input=x[:, i, :].contiguous(),
            weight=weight[i],
            weight_scale=scales[i],
            workspace=None,
            size_n=n,
            size_k=k,
        )
    return out
