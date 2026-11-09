# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLDummyInputsBuilder,
    Qwen2VLForConditionalGeneration,
    Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2VLMultiModalProcessor,
    info=Qwen2VLProcessingInfo,
    dummy_inputs=Qwen2VLDummyInputsBuilder,
)
class MyQwen2VL(Qwen2VLForConditionalGeneration):
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # this dummy model always predicts the first token
        logits = super().compute_logits(hidden_states)
        if logits is not None:
            logits.zero_()
            logits[:, 0] += 1.0
        return logits
