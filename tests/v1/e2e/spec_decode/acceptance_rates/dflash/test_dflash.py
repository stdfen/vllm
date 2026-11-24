# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass

import pytest

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform

from ...utils import compute_acceptance_len, get_spec_decode_metric_value
from ..utils import run_acceptance_length_eval


@dataclass(frozen=True)
class DFlashCorrectnessConfig:
    model: str
    draft_model: str
    expected_accuracy: float
    expected_acceptance_len: float
    num_speculative_tokens: int = 16
    max_model_len: int = 4096
    max_num_seqs: int = 128
    num_questions: int = 1319
    use_chat_completions: bool = False
    enforce_eager: bool = False
    disable_flashinfer_sampler: bool = False
    language_model_only: bool = False
    chat_template_kwargs: dict[str, object] | None = None


QWEN3_DFLASH = DFlashCorrectnessConfig(
    model="Qwen/Qwen3-8B",
    draft_model="z-lab/Qwen3-8B-DFlash-b16",
    expected_accuracy=0.8,
    expected_acceptance_len=3.5,
)

QWEN3_8_DFLASH2_NVFP4 = DFlashCorrectnessConfig(
    model="nvidia/Qwen3.8-27B-NVFP4",
    draft_model="z-lab/Qwen3.8-27B-DFlash2",
    expected_accuracy=0.90,
    expected_acceptance_len=6.14 * 0.95,
    num_speculative_tokens=7,
    max_num_seqs=32,
    use_chat_completions=True,
    language_model_only=True,
    chat_template_kwargs={"enable_thinking": False},
)

LAGUNA_DFLASH_NVFP4 = DFlashCorrectnessConfig(
    model="poolside/Laguna-XS-2.1-NVFP4",
    draft_model="poolside/Laguna-XS-2.1-DFlash-NVFP4",
    expected_accuracy=0.7,  # Standard GSM8K sanity floor.
    expected_acceptance_len=3.05 * 0.9,
    num_speculative_tokens=15,
    max_model_len=8192,
    max_num_seqs=32,
    num_questions=200,
    use_chat_completions=True,
    enforce_eager=True,
    disable_flashinfer_sampler=True,
)


@pytest.mark.parametrize("use_mrv2", [False, True])
def test_dflash_reference_acceptance_lengths(
    monkeypatch: pytest.MonkeyPatch,
    use_mrv2: bool,
    vllm_runner,
):
    run_acceptance_length_eval(
        monkeypatch,
        vllm_runner,
        spec_config={
            "model": QWEN3_DFLASH.model,
            "trust_remote_code": True,
            "speculative_config": {
                "method": "dflash",
                "model": QWEN3_DFLASH.draft_model,
                "num_speculative_tokens": QWEN3_DFLASH.num_speculative_tokens,
                "max_model_len": 32768,
            },
            "max_model_len": 32768,
            "max_num_seqs": 128,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": False,
            "disable_log_stats": False,
        },
        # Table 1 in https://arxiv.org/pdf/2602.06036.
        expected_acceptance_lengths={
            "mt-bench": 4.24,
            "humaneval": 6.50,
            "gsm8k": 6.54 * 0.975,
        },
        chat_template_kwargs={"enable_thinking": False},
        use_mrv2=use_mrv2,
    )


@single_gpu_only
@pytest.mark.parametrize(
    ("config", "use_mrv2"),
    [
        pytest.param(
            QWEN3_DFLASH,
            False,
            id="qwen3-mrv1",
        ),
        pytest.param(
            QWEN3_DFLASH,
            True,
            id="qwen3-mrv2",
        ),
        pytest.param(
            LAGUNA_DFLASH_NVFP4,
            True,
            id="laguna-nvfp4-mrv2",
        ),
        pytest.param(
            QWEN3_8_DFLASH2_NVFP4,
            True,
            id="qwen3.8-dflash2-nvfp4-mrv2",
        ),
    ],
)
def test_dflash_correctness(
    monkeypatch: pytest.MonkeyPatch,
    config: DFlashCorrectnessConfig,
    use_mrv2: bool,
    vllm_runner,
):
    """Guard GSM8K accuracy and batched acceptance length for DFlash models."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if use_mrv2 else "0")
    if config.disable_flashinfer_sampler:
        monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    with vllm_runner(
        config.model,
        block_size=None,
        trust_remote_code=True,
        speculative_config={
            "method": "dflash",
            "model": config.draft_model,
            "num_speculative_tokens": config.num_speculative_tokens,
            "max_model_len": config.max_model_len,
        },
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        gpu_memory_utilization=0.85,
        enforce_eager=config.enforce_eager,
        disable_log_stats=False,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
        language_model_only=config.language_model_only,
    ) as spec_runner:
        spec_llm = spec_runner.llm
        results = evaluate_gsm8k_offline(
            spec_llm,
            num_questions=config.num_questions,
            use_chat_completions=config.use_chat_completions,
            chat_template_kwargs=config.chat_template_kwargs,
        )
        accuracy = results["accuracy"]
        acceptance_len = compute_acceptance_len(spec_llm.get_metrics())
        context = (
            f"DFlash target={config.model}, draft={config.draft_model}, MRV2={use_mrv2}"
        )
        print(
            f"{context}: GSM8K accuracy={accuracy:.3f}, "
            f"acceptance_len={acceptance_len:.2f}"
        )

        assert accuracy >= config.expected_accuracy, (
            f"{context}: GSM8K accuracy {accuracy:.3f} is below "
            f"{config.expected_accuracy:.3f}; acceptance_len={acceptance_len:.3f}"
        )
        assert acceptance_len >= config.expected_acceptance_len, (
            f"{context}: acceptance_len {acceptance_len:.3f} is below "
            f"{config.expected_acceptance_len:.3f}; accuracy={accuracy:.3f}"
        )


@single_gpu_only
@pytest.mark.parametrize("tree_mode", ["packed", "graph"])
def test_dflash2_tree_draft_backend_is_independent(
    monkeypatch: pytest.MonkeyPatch, vllm_runner, tree_mode: str
):
    """The draft can use FlashInfer while the tree target requires Triton."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    config = QWEN3_8_DFLASH2_NVFP4
    kwargs = dict(
        dtype="bfloat16",
        enforce_eager=True,
        language_model_only=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        async_scheduling=False,
        max_model_len=config.max_model_len,
        max_num_seqs=8,
        kv_cache_dtype="fp8",
        mamba_ssm_cache_dtype="float32",
        disable_log_stats=False,
        speculative_config={
            "method": "dflash",
            "model": config.draft_model,
            "num_speculative_tokens": config.num_speculative_tokens,
            "attention_backend": "FLASHINFER",
            "enable_tree_dflash2": True,
            "tree_dflash2_mode": tree_mode,
        },
    )
    with pytest.raises(ValueError, match="Packed Tree-DFlash2 requires TRITON_ATTN"):
        EngineArgs(
            model=config.model, attention_backend="FLASHINFER", **kwargs
        ).create_engine_config()

    with vllm_runner(
        config.model, attention_backend="TRITON_ATTN", block_size=None, **kwargs
    ) as runner:
        max_tokens = 16
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        (result,) = runner.llm.generate(["The capital of France is"], params)
        assert len(result.outputs[0].token_ids) == max_tokens
        assert (
            get_spec_decode_metric_value(
                runner.llm.get_metrics(), "vllm:spec_decode_num_drafts"
            )
            > 0
        )


@pytest.fixture
def tensorfold_gb10_checkpoint_args(monkeypatch: pytest.MonkeyPatch):
    target = os.getenv("VLLM_TEST_TENSORFOLD_TARGET")
    draft = os.getenv("VLLM_TEST_TENSORFOLD_DRAFT")
    if not target or not draft:
        pytest.skip("Set the two local TensorFold GB10 checkpoint paths.")
    if not current_platform.is_cuda() or not current_platform.has_device_capability(
        121
    ):
        pytest.skip("TensorFold tree mode requires GB10.")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    config = dict(
        attention_backend="TRITON_ATTN",
        dtype="bfloat16",
        enforce_eager=True,
        language_model_only=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        max_model_len=4096,
        max_num_seqs=8,
        kv_cache_dtype="fp8",
        mamba_ssm_cache_dtype="float32",
        disable_log_stats=False,
        block_size=None,
    )
    speculative_config = {
        "method": "dflash",
        "model": draft,
        "num_speculative_tokens": 15,
        "enable_tree_dflash2": True,
        "tree_dflash2_mode": "tensorfold",
    }
    return target, config, speculative_config


@single_gpu_only
@pytest.mark.parametrize(
    ("use_graph", "batch_device_commit"),
    [(False, False), (True, False), (True, True)],
    ids=["eager", "graph", "batch-device-commit"],
)
def test_tensorfold_tree_matches_target_with_local_nvfp4_weights(
    tensorfold_gb10_checkpoint_args,
    vllm_runner,
    use_graph: bool,
    batch_device_commit: bool,
):
    target, config, speculative_config = tensorfold_gb10_checkpoint_args
    speculative_config["tree_dflash2_tensorfold_cuda_graph"] = use_graph
    speculative_config["tree_dflash2_tensorfold_batch_device_commit"] = (
        batch_device_commit
    )
    prompts = [
        "Count from one to twenty and separate the numbers with commas.",
        "Translate the colors red, white and blue to Chinese.",
    ]
    params = [
        SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True) for n in (64, 97)
    ]
    with vllm_runner(target, **config) as reference:
        expected = [
            tuple(r.outputs[0].token_ids)
            for r in reference.llm.generate(prompts, params)
        ]
    with vllm_runner(
        target, speculative_config=speculative_config, **config
    ) as candidate:
        actual = [
            tuple(r.outputs[0].token_ids)
            for r in candidate.llm.generate(prompts, params)
        ]
        assert (
            get_spec_decode_metric_value(
                candidate.llm.get_metrics(), "vllm:spec_decode_num_drafts"
            )
            > 0
        )
    assert actual == expected


@single_gpu_only
def test_tensorfold_tree_completes_mixed_prefill_and_draft_decode(
    monkeypatch: pytest.MonkeyPatch, tensorfold_gb10_checkpoint_args, vllm_runner
):
    """A new prompt can prefill while an earlier request verifies drafts."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    target, config, speculative_config = tensorfold_gb10_checkpoint_args
    config.update(enable_chunked_prefill=True, max_num_batched_tokens=8192)
    params = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
    with vllm_runner(target, speculative_config=speculative_config, **config) as runner:
        llm = runner.get_llm()
        engine = llm.llm_engine
        decode_id, prefill_id = "0", "1"
        engine.add_request(
            decode_id,
            "Count from one to twenty and separate the numbers with commas.",
            params,
        )
        (first,) = engine.step()
        assert first.request_id == decode_id
        assert len(first.outputs[0].token_ids) == 1

        engine.add_request(
            prefill_id,
            "Translate the colors red, white and blue to Chinese.",
            params,
        )
        mixed = {output.request_id: output for output in engine.step()}
        assert mixed.keys() == {decode_id, prefill_id}
        assert len(mixed[prefill_id].outputs[0].token_ids) == 1
        assert len(mixed[decode_id].outputs[0].token_ids) > 1
        assert (
            get_spec_decode_metric_value(
                llm.get_metrics(), "vllm:spec_decode_num_drafts"
            )
            > 0
        )
        completed = {
            output.request_id: output
            for output in llm.wait_for_completion(use_tqdm=False)
        }
        assert completed.keys() == {decode_id, prefill_id}
        for output in completed.values():
            assert output.finished
            assert len(output.outputs[0].token_ids) == params.max_tokens
