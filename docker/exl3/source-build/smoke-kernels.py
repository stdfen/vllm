"""Small numeric GPU checks; run offline with an initially empty cache."""

import json
import math

import flashinfer
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(a, b, output, count: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < count
    tl.store(
        output + offsets,
        tl.load(a + offsets, valid) + tl.load(b + offsets, valid),
        valid,
    )


def main():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability() == (12, 1)
    torch.manual_seed(7)
    checked = []

    from vllm import _custom_ops

    value = torch.randn(32, 128, device="cuda", dtype=torch.float16)
    weight = torch.randn(128, device="cuda", dtype=torch.float16)
    output = torch.empty_like(value)
    _custom_ops.rms_norm(output, value, weight, 1e-6)
    expected = (
        value.float()
        * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
        * weight.float()
    )
    torch.testing.assert_close(output.float(), expected, rtol=0.02, atol=0.004)
    checked.append("vllm_native_rms_norm")

    key = torch.randn(128, 8, 128, device="cuda", dtype=torch.float16)
    val = torch.randn_like(key)
    query = torch.randn(32, 8, 128, device="cuda", dtype=torch.float16)
    prefill = flashinfer.single_prefill_with_kv_cache(query, key, val, causal=False)
    scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) / math.sqrt(128)
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), val.float())
    torch.testing.assert_close(prefill.float(), expected, rtol=0.03, atol=0.004)
    checked.append("flashinfer_prefill")

    decode = flashinfer.single_decode_with_kv_cache(query[0], key, val)
    torch.testing.assert_close(decode.float(), expected[0], rtol=0.03, atol=0.004)
    checked.append("flashinfer_decode")

    a = torch.randn(1024, device="cuda")
    b = torch.randn_like(a)
    result = torch.empty_like(a)
    add_kernel[(4,)](a, b, result, a.numel(), 256)
    torch.testing.assert_close(result, a + b)
    checked.append("triton_jit")

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_output = a + b
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, a + b)
    checked.append("cuda_graph_replay")
    print(
        json.dumps(
            {"gpu": torch.cuda.get_device_name(), "checks": checked, "status": "passed"}
        )
    )


if __name__ == "__main__":
    main()
