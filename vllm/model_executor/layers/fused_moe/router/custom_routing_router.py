# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.platforms import current_platform


@torch.compile(backend=current_platform.simple_compile_backend)
def token_choice_with_bias(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    """Sigmoid -> top-k (-> renormalize) custom routing for Cohere2Moe."""
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.float().sigmoid()
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def topk_sigmoid_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.models.utils import fast_topk

    router_scores, router_indices = fast_topk(gating_output, topk, dim=-1)
    router_scores = torch.sigmoid(router_scores.float())
    return router_scores, router_indices.to(torch.int32)


class CustomRoutingRouter(BaseRouter):
    """Router using a custom user-provided routing function."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        custom_routing_function: Callable,
        eplb_state: EplbLayerState | None = None,
        renormalize: bool = True,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )
        self.custom_routing_function = custom_routing_function
        self.renormalize = renormalize

    @property
    def routing_method_type(self) -> RoutingMethodType:
        # NOTE: FLASHINFER_TRTLLM support the Llama4 router.
        if self.custom_routing_function == topk_sigmoid_routing:
            return RoutingMethodType.Llama4
        # Cohere MoE uses sigmoid -> top-k, optionally followed by renormalize.
        if self.custom_routing_function == token_choice_with_bias:
            if self.renormalize:
                return RoutingMethodType.SigmoidRenorm
            return RoutingMethodType.Sigmoid
        return RoutingMethodType.Custom

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using the custom routing function."""
        topk_weights, topk_ids = self.custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )

        return topk_weights.to(torch.float32), topk_ids.to(
            torch.int32 if indices_type is None else indices_type
        )
