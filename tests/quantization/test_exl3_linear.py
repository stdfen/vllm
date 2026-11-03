# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch.nn import Parameter

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.exl3.config import Exl3Config
from vllm.model_executor.layers.quantization.exl3.linear import Exl3LinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)


def _loader(*args) -> None:
    del args


@pytest.mark.parametrize(
    "local_input,local_output,global_input,global_output,axis",
    [
        (64, 128, 64, 256, "column"),
        (64, 128, 128, 128, "row"),
        (64, 128, 64, 128, "replicated"),
    ],
)
def test_create_weights_shapes_and_axis(
    local_input: int,
    local_output: int,
    global_input: int,
    global_output: int,
    axis: str,
) -> None:
    config = Exl3Config(tensor_bits={"proj": 3})
    method = Exl3LinearMethod(config, "proj")
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        local_input,
        [local_output],
        global_input,
        global_output,
        torch.bfloat16,
        weight_loader=_loader,
    )
    assert layer._exl3_parallel_axis == axis
    assert layer.trellis.shape == (local_input // 16, local_output // 16, 48)
    assert layer.suh.shape == (local_input,)
    assert layer.svh.shape == (local_output,)
    assert layer.mul1.shape == (1,)
    assert layer.trellis.input_dim == 0
    assert layer.trellis.output_dim == 1
    assert layer.trellis.packed_factor == 16


def test_create_weights_rejects_bad_tile() -> None:
    method = Exl3LinearMethod(Exl3Config(tensor_bits={"proj": 3}), "proj")
    with pytest.raises(ValueError, match="16x16-aligned"):
        method.create_weights(
            torch.nn.Module(),
            63,
            [128],
            63,
            128,
            torch.bfloat16,
            weight_loader=_loader,
        )


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_column_loader_slices_packed_trellis(tp_rank: int) -> None:
    owner = type("Owner", (), {"tp_rank": tp_rank})()
    param = Parameter(torch.empty(4, 4, 48, dtype=torch.int16), requires_grad=False)
    param.output_dim = 1
    loaded = torch.arange(4 * 8 * 48, dtype=torch.int32).reshape(4, 8, 48)

    ColumnParallelLinear.weight_loader(owner, param, loaded)

    expected = loaded[:, tp_rank * 4 : (tp_rank + 1) * 4].to(torch.int16)
    torch.testing.assert_close(param, expected)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_row_loader_slices_packed_trellis(tp_rank: int) -> None:
    owner = type("Owner", (), {"tp_rank": tp_rank})()
    param = Parameter(torch.empty(4, 8, 48, dtype=torch.int16), requires_grad=False)
    param.input_dim = 0
    loaded = torch.arange(8 * 8 * 48, dtype=torch.int32).reshape(8, 8, 48)

    RowParallelLinear.weight_loader(owner, param, loaded)

    expected = loaded[tp_rank * 4 : (tp_rank + 1) * 4].to(torch.int16)
    torch.testing.assert_close(param, expected)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_merged_column_loader_slices_each_packed_shard(tp_rank: int) -> None:
    owner = type(
        "Owner",
        (),
        {
            "tp_rank": tp_rank,
            "tp_size": 2,
            "output_sizes": [64, 64],
            "validate_shard_id": lambda self, shard_id: True,
            "weight_loader": MergedColumnParallelLinear.weight_loader,
        },
    )()
    param = Parameter(torch.empty(4, 4, 48, dtype=torch.int16), requires_grad=False)
    param.output_dim = 1
    param.packed_dim = 1
    param.packed_factor = 16
    first = torch.full((4, 4, 48), 10, dtype=torch.int16)
    second = torch.full((4, 4, 48), 20, dtype=torch.int16)

    MergedColumnParallelLinear.weight_loader(owner, param, first, 0)
    MergedColumnParallelLinear.weight_loader(owner, param, second, 1)

    assert torch.equal(param[:, :2], first[:, tp_rank * 2 : (tp_rank + 1) * 2])
    assert torch.equal(param[:, 2:], second[:, tp_rank * 2 : (tp_rank + 1) * 2])


def test_fused_exl3_scalar_markers_are_normalized_and_must_agree() -> None:
    config = Exl3Config(
        tensor_bits={"layers.0.attn.wq_a": 6, "layers.0.attn.wkv": 6}
    )
    method = Exl3LinearMethod(
        config, "language_model.model.layers.0.attn.fused_wqa_wkv"
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        64,
        [64, 64],
        64,
        256,
        torch.bfloat16,
        weight_loader=_loader,
    )

    layer.mul1.weight_loader(layer.mul1, torch.tensor(1), 0)
    layer.mul1.weight_loader(layer.mul1, torch.tensor(1), 1)
    assert layer.mul1.tolist() == [1]

    with pytest.raises(ValueError, match="different codebook markers"):
        layer.mul1.weight_loader(layer.mul1, torch.tensor(2), 1)


def test_vocab_loader_pads_nonzero_packed_output_axis() -> None:
    shard_indices = type(
        "ShardIndices",
        (),
        {"org_vocab_start_index": 64, "org_vocab_end_index": 96},
    )()
    owner = type("Owner", (), {"shard_indices": shard_indices, "org_vocab_size": 96})()
    param = Parameter(
        torch.full((4, 4, 48), -1, dtype=torch.int16),
        requires_grad=False,
    )
    param.output_dim = 1
    param.packed_dim = 1
    param.packed_factor = 16
    loaded = torch.arange(4 * 6 * 48, dtype=torch.int32).reshape(4, 6, 48)

    VocabParallelEmbedding.weight_loader(owner, param, loaded)

    torch.testing.assert_close(param[:, :2], loaded[:, 4:6].to(torch.int16))
    assert torch.count_nonzero(param[:, 2:]) == 0


def test_vocab_loader_keeps_standard_axis_zero_padding() -> None:
    shard_indices = type(
        "ShardIndices",
        (),
        {"org_vocab_start_index": 64, "org_vocab_end_index": 96},
    )()
    owner = type("Owner", (), {"shard_indices": shard_indices, "org_vocab_size": 96})()
    param = Parameter(torch.full((64, 8), -1, dtype=torch.float32))
    param.output_dim = 0
    loaded = torch.arange(96 * 8, dtype=torch.float32).reshape(96, 8)

    VocabParallelEmbedding.weight_loader(owner, param, loaded)

    torch.testing.assert_close(param[:32], loaded[64:96])
    assert torch.count_nonzero(param[32:]) == 0


def test_apply_restores_multidimensional_shape_and_adds_bias() -> None:
    method = Exl3LinearMethod(Exl3Config(tensor_bits={"proj": 3}), "proj")
    layer = torch.nn.Module()
    layer._exl3_backend = lambda x: x[:, :4] * 2
    x = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0])

    output = method.apply(layer, x, bias)

    assert output.shape == (2, 3, 4)
    torch.testing.assert_close(output, x[..., :4] * 2 + bias)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_grouped_checkpoint_slices_load_into_local_groups(tp_rank: int) -> None:
    tensor_bits = {f"layers.0.attn.wo_a.slice.{index}": 5 for index in range(8)}
    method = Exl3LinearMethod(
        Exl3Config(tensor_bits=tensor_bits), "model.layers.0.attn.wo_a"
    )
    layer = torch.nn.Module()
    layer.tp_rank = tp_rank
    method.create_weights(
        layer,
        4096,
        [4096],
        4096,
        8192,
        torch.bfloat16,
        weight_loader=_loader,
    )

    assert layer.trellis.shape == (4, 256, 64, 80)
    assert layer.suh.shape == (4, 4096)
    assert layer.svh.shape == (4, 1024)
    assert layer._exl3_group_start == tp_rank * 4

    owned_slice = tp_rank * 4 + 1
    trellis = torch.full((256, 64, 80), 7, dtype=torch.int16)
    assert layer.trellis.weight_loader(layer.trellis, trellis, owned_slice)
    assert torch.equal(layer.trellis[1], trellis)
    assert not layer.trellis.weight_loader(layer.trellis, trellis, (1 - tp_rank) * 4)


def test_grouped_apply_keeps_group_axis() -> None:
    tensor_bits = {f"layers.0.attn.wo_a.slice.{index}": 5 for index in range(2)}
    method = Exl3LinearMethod(
        Exl3Config(tensor_bits=tensor_bits), "model.layers.0.attn.wo_a"
    )
    layer = torch.nn.Module()
    layer._exl3_backends = (
        lambda x: x[:, :2] + 10,
        lambda x: x[:, :2] + 20,
    )
    x = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)

    output = method.apply(layer, x)

    assert output.shape == (3, 2, 2)
    torch.testing.assert_close(output[:, 0], x[:, 0, :2] + 10)
    torch.testing.assert_close(output[:, 1], x[:, 1, :2] + 20)
