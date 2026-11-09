# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed import parallel_state
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.model_executor.models.mimo import MiMoModel
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    reserve_aux_intermediate_tensor_slots,
    verify_supports_aux_hidden_states_over_pp,
)


def test_aux_layers_are_sorted_and_deduplicated():
    model = EagleModelMixin()
    model._set_aux_hidden_state_layers((48, 3, 90, 24, 48))
    assert model.aux_hidden_state_layers == (3, 24, 48, 90)


def test_mimo_does_not_inherit_aux_hidden_state_pp_support():
    inner = MiMoModel.__new__(MiMoModel)
    target = SimpleNamespace(model=inner)

    assert not inner.supports_aux_hidden_states_over_pp
    with pytest.raises(ValueError, match="does not support eagle3"):
        verify_supports_aux_hidden_states_over_pp(target, "eagle3")


@pytest.mark.parametrize("first_rank", [True, False])
def test_aux_slots_are_reserved_only_on_receiving_stage(monkeypatch, first_rank):
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=2, is_first_rank=first_rank),
    )
    inner = SimpleNamespace(
        supports_aux_hidden_states_over_pp=True,
        _aux_slot_base_cached=2,
        AUX_HIDDEN_STATE_KEY="aux_hidden_states_",
        config=SimpleNamespace(hidden_size=8),
    )
    target = SimpleNamespace(
        model=inner,
        make_empty_intermediate_tensors=lambda batch_size, dtype, device: (
            IntermediateTensors({"hidden_states": torch.empty(batch_size, 8)})
        ),
    )
    reserve_aux_intermediate_tensor_slots(target)
    tensors = target.make_empty_intermediate_tensors(3, torch.float32, "cpu")
    assert set(tensors.tensors) == (
        {"hidden_states"}
        if first_rank
        else {"hidden_states", "aux_hidden_states_0", "aux_hidden_states_1"}
    )
    if not first_rank:
        for index in range(2):
            torch.testing.assert_close(
                tensors[f"aux_hidden_states_{index}"], torch.zeros(3, 8)
            )
