# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm the exact observed sparse-index mapping keys, including slice alignment."""


def warm():
    import torch

    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        compute_global_topk_indices_and_lens,
    )

    # Pinned model/runtime: int32 indices/mapping, topk512, table stride8192,
    # compressed cache blocks32/64. Prefill slices shift mapping/bool pointers.
    topk = torch.zeros((8, 512), dtype=torch.int32, device="cuda")
    table = torch.full((8, 8192), 3, dtype=torch.int32, device="cuda")
    mapping = torch.zeros(24, dtype=torch.int32, device="cuda")
    valid = torch.ones(24, dtype=torch.bool, device="cuda")
    for block in (32, 64):
        for offset in range(16):
            valid.fill_(True)
            valid[offset + 7] = False
            selected = valid[offset : offset + 8]
            slots, lens = compute_global_topk_indices_and_lens(
                topk, mapping[offset : offset + 8], table, block, selected
            )
            assert torch.equal(slots, torch.full_like(slots, 3 * block))
            assert torch.equal(lens, selected.int() * 512)
    torch.accelerator.synchronize()
    print(
        "BENCH_KERNEL_WARMUP_COMPLETE global_topk512 table8192 "
        "blocks32,64 offsets0..15",
        flush=True,
    )
    warm_dflash_inputs()


def warm_dflash_inputs():
    """Exercise all bounded context-tail tiles on scratch, never model state.

    A continuously refilled prefill can end with 30 context rows, requiring
    a 64-wide tile even when long warmup prompts happened to use 8/16/128/256.
    Triton specializes this constexpr independently of the launch grid.
    """
    import torch

    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        _prepare_dflash_inputs_kernel,
    )

    def zeros(dtype, count=2048):
        return torch.zeros(count, dtype=dtype, device="cuda")

    i32, i64 = torch.int32, torch.int64
    outputs = [
        zeros(t)
        for t in (
            i32,
            i64,
            torch.bool,
            i32,
            i32,
            i64,
            i64,
            i64,
            i64,
            i64,
            i64,
            torch.float32,
            i64,
        )
    ]
    positions = torch.arange(2048, dtype=i64, device="cuda")
    starts = zeros(i32, 9)
    mapping = torch.arange(8, dtype=i64, device="cuda")
    last = torch.full((8,), 7, dtype=i64, device="cuda")
    next_tokens, sampled, rejected = zeros(i32, 8), zeros(i32, 8), zeros(i32, 8)
    sampled.fill_(1)
    temperatures, seeds = zeros(torch.float32, 8), zeros(i64, 8)
    checks = 0
    for block_size in (32, 64):
        stride = 524288 // block_size
        table = torch.ones((8, stride), dtype=i32, device="cuda")
        for width in range(1, 6):
            for tile in (8, 16, 32, 64, 128, 256):
                context = tile - width
                starts[1] = context
                _prepare_dflash_inputs_kernel[(1, 1)](
                    *outputs,
                    positions,
                    starts,
                    mapping,
                    last,
                    next_tokens,
                    sampled,
                    rejected,
                    temperatures,
                    seeds,
                    table,
                    stride,
                    128799,
                    block_size,
                    width,
                    width,
                    8,
                    2048,
                    524288,
                    0,
                    SAMPLE_FROM_ANCHOR=True,
                    PAD_SLOT_ID=-1,
                    CP_SIZE=1,
                    CP_INTERLEAVE=1,
                    BLOCK_SIZE=tile,
                )
                expected = torch.full((width,), 128799, dtype=i32, device="cuda")
                expected[0] = 7
                assert torch.equal(outputs[0][:width], expected)
                assert torch.equal(
                    outputs[1][:width], positions[context : context + width]
                )
                assert outputs[3][1].item() == width
                assert outputs[4][0].item() == context + width
                assert torch.equal(outputs[8][:width], positions[:width])
                assert torch.all(outputs[10][width : 8 * width] == -1)
                checks += 1
    torch.accelerator.synchronize()
    print(
        "BENCH_KERNEL_WARMUP_COMPLETE dflash_inputs tiles8..256 widths1..5 "
        f"blocks32,64 exact_checks={checks}",
        flush=True,
    )
