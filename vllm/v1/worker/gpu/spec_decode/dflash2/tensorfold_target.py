# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TensorFold dense target verification with vLLM-owned cache pages."""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from tensorfold.cuda.nvfp4 import linear
from tensorfold.families.qwen3_5.cuda import glue
from tensorfold.families.qwen3_5.cuda.forward import _mlp, _mm
from tensorfold.families.qwen3_5.cuda.nvfp4_load import load_nvfp4
from tensorfold.families.qwen3_5.cuda.weights import Plain, Weights

from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify import TensorFoldVerify

logger = init_logger(__name__)
C1_QMM_ROWS = 16
# Each entry is (K splits, number of projections in the qualified checkpoint).
C1_QMM_SPLITS = {
    (linear.FP4, 5120, 17408): (2, 64),
    (linear.FP4, 17408, 5120): (4, 128),
    (linear.FP4, 248320, 5120): (4, 1),
    (linear.FP8, 1024, 5120): (2, 32),
    (linear.FP8, 5120, 6144): (1, 64),
}


class _C1SplitProjection:
    """Apply the stock TensorFold kernel with the selected K split count."""

    def __init__(
        self, projection: linear.Fp4Linear | linear.Fp8Linear, splits: int
    ) -> None:
        if isinstance(projection, linear.Fp4Linear):
            self.mode, self.packed, self.scales = (
                linear.FP4,
                projection.words,
                projection.bs,
            )
        else:
            self.mode, self.packed, self.scales = linear.FP8, projection.w8, None
        self.scale = projection.scale
        self.n, self.npad = projection.n, projection.npad
        self.splits = splits

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or x.stride(-1) != 1:
            x = x.to(torch.bfloat16).contiguous()
        output = torch.empty(
            (x.shape[0], self.n), dtype=torch.bfloat16, device=x.device
        )
        linear._ext().qmmf(
            x,
            self.packed,
            self.scales,
            self.scale,
            output,
            None,
            self.mode,
            self.n,
            self.splits,
            self.npad,
            C1_QMM_ROWS,
            False,
        )
        return output


def _with_c1_qmm_splits(weights: Weights) -> Weights:
    applied = Counter()

    def tune(projection):
        if isinstance(projection, linear.Fp4Linear):
            mode = linear.FP4
        elif isinstance(projection, linear.Fp8Linear):
            mode = linear.FP8
        else:
            return projection
        key = mode, projection.n, projection.k
        if key not in C1_QMM_SPLITS:
            return projection
        applied[key] += 1
        return _C1SplitProjection(projection, C1_QMM_SPLITS[key][0])

    layers = []
    for layer in weights.layers:
        gdn, attn = layer.gdn, layer.attn
        if gdn is not None:
            inputs = ("zba",) if gdn.zba is not None else ("z", "b", "a")
            gdn = replace(
                gdn, **{n: tune(getattr(gdn, n)) for n in ("qkv", "out", *inputs)}
            )
        if attn is not None:
            inputs = ("kv",) if attn.kv is not None else ("k", "v")
            attn = replace(
                attn, **{n: tune(getattr(attn, n)) for n in ("q", "o", *inputs)}
            )
        layers.append(
            replace(
                layer,
                gdn=gdn,
                attn=attn,
                gate=tune(layer.gate),
                up=tune(layer.up),
                down=tune(layer.down),
            )
        )
    head = tune(weights.head)
    expected = {key: count for key, (_, count) in C1_QMM_SPLITS.items()}
    if applied != expected:
        raise ValueError("C1 QMM tuning requires the tested Qwen3.8 NVFP4 checkpoint.")
    logger.info(
        "TensorFold prepared %d tuned C1 quantized projections.", sum(applied.values())
    )
    return replace(weights, layers=layers, head=head)


class TensorFoldNativeTarget:
    """Use the checkpoint's native projections for a pure tree-verification batch."""

    def __init__(
        self,
        target: torch.nn.Module,
        model_dir: Path,
        device: torch.device,
        *,
        c1_qmm_split: bool = False,
    ):
        self.weights = load_nvfp4(model_dir, str(device))
        self.taps = tuple(target.aux_hidden_state_layers)
        cfg = self.weights.config
        embedding = target.embed_tokens.weight.detach()[: cfg.vocab]
        if (
            self.taps != (6, 20, 34, 48, 62)
            or cfg.layers != 64
            or embedding.shape != (cfg.vocab, cfg.hidden)
        ):
            raise ValueError("The native verifier requires Qwen3.8-27B DFlash2 taps.")
        if embedding.dtype != torch.bfloat16 or not embedding.is_contiguous():
            raise ValueError("The native verifier needs a contiguous BF16 embedding.")
        self.weights.embed = Plain(embedding)
        self.cache_layers = []
        for weights, layer in zip(self.weights.layers, target.layers, strict=True):
            if weights.moe is not None or weights.linear != (
                layer.layer_type == "linear_attention"
            ):
                raise ValueError("Native verifier and vLLM target layers differ.")
            self.cache_layers.append(
                layer.linear_attn if weights.linear else layer.self_attn.attn
            )
        self.c1_weights = _with_c1_qmm_splits(self.weights) if c1_qmm_split else None

    def forward(
        self, verify: TensorFoldVerify, ids: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        w = (
            self.c1_weights
            if self.c1_weights is not None
            and len(verify.plan.starts) == 2
            and ids.numel() == C1_QMM_ROWS
            else self.weights
        )
        cfg = w.config
        x = glue.embedding(ids, w.embed)
        pending = None
        taps = []
        for i, (layer, cache) in enumerate(
            zip(w.layers, self.cache_layers, strict=True), start=1
        ):
            x, row, sums = glue.add_rmsnorm(x, pending, layer.input_norm, cfg.eps)
            if layer.linear:
                gdn = layer.gdn
                qkv = _mm(row, gdn.qkv, sums)
                if gdn.zba is None:
                    z = _mm(row, gdn.z, sums)
                    b = _mm(row, gdn.b, sums)
                    a = _mm(row, gdn.a, sums)
                else:
                    z, b, a = _mm(row, gdn.zba, sums).split(
                        (cfg.v_heads * cfg.dv, cfg.v_heads, cfg.v_heads), dim=-1
                    )
                z = z.contiguous().view(ids.numel(), cfg.v_heads, cfg.dv)
                gated, gate_sums = verify.gdn(cache, qkv, b, a, z)
                residual = _mm(gated, gdn.out, gate_sums)
            else:
                attn = layer.attn
                qg = _mm(row, attn.q, sums)
                if attn.kv is None:
                    key = _mm(row, attn.k, sums)
                    value = _mm(row, attn.v, sums)
                else:
                    key, value = _mm(row, attn.kv, sums).chunk(2, dim=-1)
                    key, value = key.contiguous(), value.contiguous()
                query, key = glue.attn_prep(
                    qg,
                    key,
                    attn.q_norm,
                    attn.k_norm,
                    positions,
                    w.inv_freq,
                    cfg.eps,
                    heads=cfg.heads,
                    kv_heads=cfg.kv_heads,
                    head_dim=cfg.head_dim,
                    mrope_section=cfg.mrope_section,
                )
                value = value.view(ids.numel(), cfg.kv_heads, cfg.head_dim)
                out = torch.empty_like(query)
                verify.attention(cache.layer_name, cache, query, key, value, out)
                gated, gate_sums = glue.gate_mul(
                    out, qg, heads=cfg.heads, head_dim=cfg.head_dim
                )
                residual = _mm(gated, attn.o, gate_sums)
            x, row, sums = glue.add_rmsnorm(x, residual, layer.post_norm, cfg.eps)
            pending = _mlp(layer, row, sums, False)
            if i in self.taps:
                taps.append(x + pending)

        _, hidden, sums = glue.add_rmsnorm(x, pending, w.norm, cfg.eps)
        return hidden, taps, _mm(hidden, w.head, sums).argmax(-1)
