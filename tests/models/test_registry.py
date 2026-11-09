# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings

import pytest
import torch.cuda

from vllm.model_executor.models import (
    is_pooling_model,
    is_text_generation_model,
    supports_multimodal,
)
from vllm.model_executor.models.adapters import (
    as_embedding_model,
    as_seq_cls_model,
)
from vllm.model_executor.models.registry import (
    _MULTIMODAL_MODELS,
    _SPECULATIVE_DECODING_MODELS,
    _TEXT_GENERATION_MODELS,
    ModelRegistry,
    _LazyRegisteredModel,
)
from vllm.platforms import current_platform

from ..utils import create_new_process_for_each_test
from .registry import HF_EXAMPLE_MODELS


@pytest.mark.parametrize("model_arch", ModelRegistry.get_supported_archs())
def test_registry_imports(model_arch):
    # Skip if transformers version is incompatible
    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_transformers_version(
        on_fail="skip",
        check_max_version=False,
        check_version_reason="vllm",
    )

    if model_arch == "Terratorch":
        import importlib.util

        if importlib.util.find_spec("terratorch") is None:
            pytest.skip(
                "terratorch is not installed; "
                "temporarily skipped while PyPI has `lightning` quarantined "
                "(see #41376)"
            )

    # DSpark draft model is supported on CUDA and ROCm; stubbed to None on XPU.
    if model_arch == "DSparkDraftModel" and not (
        current_platform.is_cuda() or current_platform.is_rocm()
    ):
        pytest.skip("DSparkDraftModel is only supported on CUDA and ROCm")

    if model_arch in ("HYV4ForCausalLM", "HYV4MTPModel") and not (
        current_platform.is_cuda() or current_platform.is_rocm()
    ):
        pytest.skip("HY V4 is only supported on CUDA and ROCm")

    if (
        model_arch == "DeepseekV4ForConditionalGeneration"
        and not current_platform.is_cuda_alike()
    ):
        pytest.skip("Deepseek V4 vision is only supported on CUDA and ROCm")

    # Ensure all model classes can be imported successfully
    model_cls = ModelRegistry._try_load_model_cls(model_arch)
    assert model_cls is not None

    if model_arch in _SPECULATIVE_DECODING_MODELS:
        return  # Ignore these models which do not have a unified format

    if model_arch in _TEXT_GENERATION_MODELS or model_arch in _MULTIMODAL_MODELS:
        assert is_text_generation_model(model_cls)

    # All vLLM models should be convertible to a pooling model
    assert is_pooling_model(as_seq_cls_model(model_cls))
    assert is_pooling_model(as_embedding_model(model_cls))

    if model_arch in _MULTIMODAL_MODELS:
        assert supports_multimodal(model_cls)


@pytest.mark.parametrize(
    "model_arch,is_mm,score_type",
    [
        ("Qwen3ForCausalLM", False, "bi-encoder"),
        ("Qwen2VLForConditionalGeneration", True, "bi-encoder"),
        ("DeepseekV41ForCausalLM", True, "bi-encoder"),
        ("Qwen2Model", False, "bi-encoder"),
        ("TransformersForSequenceClassification", False, "cross-encoder"),
        ("ColQwen3", True, "late-interaction"),
    ],
)
def test_registry_model_property(model_arch, is_mm, score_type):
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None

    assert model_info.supports_multimodal is is_mm
    assert model_info.score_type == score_type


@pytest.mark.parametrize(
    "model_arch,is_pp",
    [
        # TODO(woosuk): Re-enable this once the MLP Speculator is supported
        # in V1.
        # ("MLPSpeculatorPreTrainedModel", False),
        ("DeepseekV2ForCausalLM", True),
        ("Qwen2VLForConditionalGeneration", True),
    ],
)
def test_registry_is_pp(model_arch, is_pp):
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None

    assert model_info.supports_pp is is_pp


@pytest.mark.parametrize(
    "model_arch,supported",
    [
        ("Qwen4ExpForCausalLM", not current_platform.is_rocm()),
        ("Qwen4ExpForConditionalGeneration", not current_platform.is_rocm()),
        ("Qwen4ExpMTP", not current_platform.is_rocm()),
        ("Qwen3ForCausalLM", False),
    ],
)
def test_registry_supports_hc_sp(model_arch, supported):
    """HC SP is opt-in for each model implementation, including MTP."""
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None
    assert model_info.supports_hc_sp is supported


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "model_arch",
    ["Qwen3VLForConditionalGeneration", "Qwen2VLForConditionalGeneration"],
)
def test_registry_inspect_does_not_init_cuda(model_arch):
    """Inspecting a model must not import it: these architectures initialize
    CUDA at import time, so the inspection path has to stay out-of-process."""
    if not current_platform.is_cuda_alike():
        pytest.skip("Requires a CUDA-like platform")

    assert ModelRegistry._try_inspect_model_cls(model_arch) is not None
    assert not torch.cuda.is_initialized()

    ModelRegistry._try_load_model_cls(model_arch)
    if not torch.cuda.is_initialized():
        warnings.warn(
            "This model no longer initializes CUDA on import. "
            "Please test using a different one.",
            stacklevel=2,
        )


@pytest.mark.parametrize(
    "model_arch,supported",
    [
        # ReplaySSM is opt-in per model.
        ("KimiLinearForCausalLM", not current_platform.is_rocm()),
        ("KimiK3ForConditionalGeneration", not current_platform.is_rocm()),
        ("Qwen3ForCausalLM", False),
    ],
)
def test_registry_supports_replayssm(model_arch, supported):
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None
    assert model_info.supports_replayssm is supported


def test_lazy_modelinfo_package_hash_includes_submodules(tmp_path):
    package_dir = tmp_path / "model_package"
    package_dir.mkdir()
    init_file = package_dir / "__init__.py"
    init_file.write_text("from .model import Model\n", encoding="utf-8")
    model_file = package_dir / "model.py"
    model_file.write_text("class Model: pass\n", encoding="utf-8")

    first_hash = _LazyRegisteredModel._get_modelinfo_module_hash(init_file)

    model_file.write_text("class Model:\n    supports_pp = True\n", encoding="utf-8")
    second_hash = _LazyRegisteredModel._get_modelinfo_module_hash(init_file)

    assert first_hash != second_hash


def test_lazy_modelinfo_package_attempts_cache_load(monkeypatch):
    cached_model_info = object()
    loaded_hashes = []

    def fake_load_cache(self, module_hash):
        loaded_hashes.append(module_hash)
        return cached_model_info

    monkeypatch.setattr(
        _LazyRegisteredModel,
        "_load_modelinfo_from_cache",
        fake_load_cache,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.registry._run_in_subprocess",
        lambda _: pytest.fail("Package-backed model should use the cache path"),
    )

    registered_model = _LazyRegisteredModel(
        module_name="vllm.model_executor.models.transformers",
        class_name="TransformersForCausalLM",
    )

    result = registered_model.inspect_model_cls()

    assert result is cached_model_info
    assert len(loaded_hashes) == 1
    assert loaded_hashes[0]


def test_hf_registry_coverage():
    untested_archs = (
        ModelRegistry.get_supported_archs() - HF_EXAMPLE_MODELS.get_supported_archs()
    )

    assert not untested_archs, (
        "Please add the following architectures to "
        f"`tests/models/registry.py`: {untested_archs}"
    )


@pytest.mark.parametrize("method", ["inspect_model_cls", "resolve_model_cls"])
@pytest.mark.parametrize(
    "implementation", ["auto", "vllm", "transformers", "terratorch"]
)
@pytest.mark.parametrize(
    "architectures",
    [
        ["LlamaForCausalLM"],
        ["Qwen3ForCausalLM", "LlamaForCausalLM"],
        ["TransformersForCausalLM"],
        ["UnapprovedPluginAlias"],
    ],
)
def test_four_family_rejects_before_dynamic_resolution(
    monkeypatch, method, implementation, architectures
):
    """Excluded checkpoints cannot enter fallback, normalization, or plugin loading."""
    from types import SimpleNamespace

    from vllm.model_executor.models.registry import _ModelRegistry

    registry = _ModelRegistry()
    registry.register_model("UnapprovedPluginAlias", "untrusted.module:Model")

    def forbidden(*args, **kwargs):
        pytest.fail("Admission must precede dynamic model resolution")

    for name in (
        "_try_resolve_transformers",
        "_try_inspect_model_cls",
        "_try_load_model_cls",
        "_normalize_arch",
    ):
        monkeypatch.setattr(registry, name, forbidden)
    config = SimpleNamespace(model_impl=implementation)
    with pytest.raises(
        ValueError, match="Supported families: GLM, DeepSeek, Qwen, Kimi"
    ):
        getattr(registry, method)(architectures, config)


def test_target_repacked_eagle_allows_only_intermediate_draft_identity():
    """A raw EAGLE3 config may enter only with target-bound EAGLE context."""
    from types import SimpleNamespace

    from vllm.model_executor.models.registry import _validate_four_family_architectures

    approved = SimpleNamespace(
        runner="draft",
        speculative_method="eagle3",
        spec_target_architectures=("Qwen3VLForConditionalGeneration",),
        model_impl="auto",
    )
    _validate_four_family_architectures(["LlamaForCausalLMEagle3"], approved)
    with pytest.raises(ValueError, match="not approved"):
        _validate_four_family_architectures(["LlamaForCausalLM"], approved)

    for key, value in (
        ("runner", "generate"),
        ("speculative_method", "mtp"),
        ("speculative_method", None),
        ("spec_target_architectures", ()),
    ):
        rejected = SimpleNamespace(**vars(approved))
        setattr(rejected, key, value)
        with pytest.raises(ValueError, match="not approved"):
            _validate_four_family_architectures(["LlamaForCausalLMEagle3"], rejected)

    from vllm.model_executor.models.registry import ModelRegistry

    registry = ModelRegistry
    assert (
        registry._normalize_arch("LlamaForCausalLMEagle3", approved)
        == "PeagleLlamaForCausalLM"
    )
    for key, value in (
        ("runner", "generate"),
        ("speculative_method", "mtp"),
        ("spec_target_architectures", ()),
    ):
        rejected = SimpleNamespace(**vars(approved))
        setattr(rejected, key, value)
        assert registry._normalize_arch("LlamaForCausalLMEagle3", rejected) == (
            "LlamaForCausalLMEagle3"
        )


@pytest.mark.parametrize("method", ["inspect_model_cls", "resolve_model_cls"])
@pytest.mark.parametrize(
    "architecture",
    [
        "GlmForCausalLM",
        "DeepseekV41ForCausalLM",
        "Qwen3ForCausalLM",
        "KimiLinearForCausalLM",
    ],
)
def test_four_family_preserves_explicit_transformers_dispatch(
    monkeypatch, method, architecture
):
    """The family gate does not disable the approved Transformers backend path."""
    from types import SimpleNamespace

    from vllm.model_executor.models.registry import _ModelRegistry

    registry = _ModelRegistry()
    sentinel = object()
    monkeypatch.setattr(
        registry, "_try_resolve_transformers", lambda *args: "TransformersForCausalLM"
    )
    monkeypatch.setattr(registry, "_try_inspect_model_cls", lambda *args: sentinel)
    monkeypatch.setattr(registry, "_try_load_model_cls", lambda *args: sentinel)
    assert getattr(registry, method)(
        [architecture], SimpleNamespace(model_impl="transformers")
    ) == (sentinel, "TransformersForCausalLM")


@pytest.mark.parametrize(
    ("target", "draft"),
    [
        ("Qwen3ForCausalLM", "DFlashDraftModel"),
        ("DeepseekV41ForCausalLM", "DSparkV41DraftModel"),
        ("DeepseekV4ForCausalLM", "DeepSeekV4MTPModel"),
        ("Qwen3ForCausalLM", "Eagle3Qwen3ForCausalLM"),
        ("KimiK3ForConditionalGeneration", "KimiK3MTPModel"),
        ("KimiK3ForConditionalGeneration", "K3DSparkModel"),
        ("DeepseekV3ForCausalLM", "LlamaForCausalLM"),
    ],
)
def test_four_family_speculation_pairing(target, draft):
    from vllm.model_executor.models.registry import validate_four_family_speculation

    if "Llama" in draft:
        with pytest.raises(ValueError, match="not approved"):
            validate_four_family_speculation([target], [draft])
        return
    validate_four_family_speculation([target], [draft])


@pytest.mark.parametrize(
    ("target", "draft"),
    [
        ("KimiK25ForConditionalGeneration", "Eagle3DeepseekV3ForCausalLM"),
        ("KimiK25ForConditionalGeneration", "Eagle3DeepseekV2ForCausalLM"),
    ],
)
def test_cross_family_eagle_draft_is_bound_to_evidenced_target(target, draft):
    """Only the evidenced Kimi-K2.5 reuse of the DeepSeek EAGLE3 draft is allowed."""
    from vllm.model_executor.models.registry import validate_four_family_speculation

    validate_four_family_speculation([target], [draft], method="eagle3")
    for other_target, other_method in (
        ("Qwen3ForCausalLM", "eagle3"),
        ("KimiK3ForConditionalGeneration", "eagle3"),
        (target, "mtp"),
    ):
        with pytest.raises(ValueError, match="not approved|Cross-family"):
            validate_four_family_speculation(
                [other_target], [draft], method=other_method
            )


def test_target_repacked_eagle_pair_requires_eagle_method():
    """The intermediate identity is valid only before target-bound repacking."""
    from vllm.model_executor.models.registry import validate_four_family_speculation

    validate_four_family_speculation(
        ["Qwen3VLForConditionalGeneration"],
        ["LlamaForCausalLMEagle3"],
        method="eagle3",
    )
    with pytest.raises(ValueError, match="not approved"):
        validate_four_family_speculation(
            ["Qwen3VLForConditionalGeneration"], ["LlamaForCausalLMEagle3"]
        )


@pytest.mark.parametrize(
    ("target", "draft"),
    [
        ("LlamaForCausalLM", "DFlashDraftModel"),
        ("Qwen3ForCausalLM", "Gemma4DSparkModel"),
        ("Qwen3ForCausalLM", "DSparkV41DraftModel"),
        ("Glm5NextForCausalLM", "KimiK3MTPModel"),
    ],
)
def test_four_family_speculation_rejects_cross_family(target, draft):
    from vllm.model_executor.models.registry import validate_four_family_speculation

    with pytest.raises(ValueError, match="not approved|do not belong"):
        validate_four_family_speculation([target], [draft])


def test_four_family_rejects_v41_classic_mtp():
    from vllm.model_executor.models.registry import validate_four_family_speculation

    with pytest.raises(ValueError, match="no classic-MTP"):
        validate_four_family_speculation(
            ["DeepseekV41ForCausalLM"], ["DeepSeekV4MTPModel"]
        )


@pytest.mark.parametrize("targets,drafts", [([], []), (["Qwen3ForCausalLM"], [])])
def test_four_family_rejects_empty_speculation_identity(targets, drafts):
    from vllm.model_executor.models.registry import validate_four_family_speculation

    with pytest.raises(ValueError, match="not approved"):
        validate_four_family_speculation(targets, drafts)


def test_four_family_classification_covers_retained_identities():
    """Every admitted identity is explicitly classified or a shared helper."""
    from vllm.model_executor.models.registry import (
        _FOUR_FAMILY_ARCHITECTURES,
        _FOUR_FAMILY_ARCH_TO_FAMILY,
    )

    assert _FOUR_FAMILY_ARCHITECTURES - _FOUR_FAMILY_ARCH_TO_FAMILY.keys() == {
        "ExtractHiddenStatesModel",
        "PeagleLlamaForCausalLM",
        "PEagleDraftModel",
        "DFlashDraftModel",
        "DFlash2DraftModel",
    }
    assert not (_FOUR_FAMILY_ARCH_TO_FAMILY.keys() - _FOUR_FAMILY_ARCHITECTURES)


@pytest.mark.parametrize(
    "target",
    [
        "Glm4ForCausalLM",
        "DeepseekV41ForCausalLM",
        "Qwen3ForCausalLM",
        "KimiLinearForCausalLM",
    ],
)
def test_hidden_state_helper_inherits_admitted_target(target):
    from vllm.model_executor.models.registry import validate_four_family_speculation

    validate_four_family_speculation(
        [target], ["ExtractHiddenStatesModel"], method="extract_hidden_states"
    )
    with pytest.raises(ValueError, match="requires the extract_hidden_states"):
        validate_four_family_speculation(
            [target], ["ExtractHiddenStatesModel"], method="draft_model"
        )


def test_hidden_state_helper_cannot_admit_excluded_target():
    from vllm.model_executor.models.registry import validate_four_family_speculation

    with pytest.raises(ValueError, match="not approved"):
        validate_four_family_speculation(
            ["LlamaForCausalLM"],
            ["ExtractHiddenStatesModel"],
            method="extract_hidden_states",
        )


@pytest.mark.parametrize(
    "architecture",
    [
        "LlamaForCausalLM",
        "LlavaForConditionalGeneration",
        "BertForSequenceClassification",
        "RobertaForSequenceClassification",
        "XLMRobertaForSequenceClassification",
        "GteNewModel",
        "GteNewForSequenceClassification",
        "HF_ColBERT",
        "NemotronHForCausalLM",
        "Mamba2ForCausalLM",
        "Zamba2ForCausalLM",
        "Dots3NoteForCausalLM",
        "Dots3NoteMTPModel",
        "Cohere2MoeForCausalLM",
        "Llama4ForCausalLM",
        "Llama4ForConditionalGeneration",
        "EagleLlama4ForCausalLM",
        "DFlashLagunaForCausalLM",
        "InklingForCausalLM",
        "InklingForConditionalGeneration",
        "InklingMTPModel",
        "MiMoV2MTPModel",
        "MiMoV2OmniMTPModel",
        "MiniMaxM3SparseForConditionalGeneration",
        "HYV4ForCausalLM",
    ],
)
def test_removed_builtin_has_no_lazy_model(architecture):
    assert architecture not in ModelRegistry.get_supported_archs()
    assert ModelRegistry._try_inspect_model_cls(architecture) is None
    assert ModelRegistry._try_load_model_cls(architecture) is None


@pytest.mark.parametrize("method", ["inspect_model_cls", "resolve_model_cls"])
@pytest.mark.parametrize("implementation", ["auto", "vllm"])
@pytest.mark.parametrize("approved", [True, False])
def test_plugin_registration_does_not_change_checkpoint_admission(
    method, implementation, approved
):
    """Registration replaces implementations, not the fixed checkpoint allowlist."""
    from types import SimpleNamespace

    from vllm.model_executor.models.registry import _ModelRegistry

    registry = _ModelRegistry()
    architecture = "Qwen3ForCausalLM" if approved else "QwenUnapprovedPlugin"
    registry.register_model(architecture, torch.nn.Linear)
    config = SimpleNamespace(model_impl=implementation)
    if not approved:
        with pytest.raises(ValueError, match="not approved"):
            getattr(registry, method)(architecture, config)
        return
    result, resolved = getattr(registry, method)(architecture, config)
    assert resolved == architecture
    if method == "resolve_model_cls":
        assert result is torch.nn.Linear
    else:
        assert result == registry.models[architecture].inspect_model_cls()


@pytest.mark.parametrize("method", ["inspect_model_cls", "resolve_model_cls"])
@pytest.mark.parametrize("implementation", ["auto", "vllm", "transformers"])
def test_excluded_identity_with_remote_config_never_loads(
    monkeypatch, method, implementation
):
    """Remote-code metadata cannot authorize an excluded effective architecture."""
    from types import SimpleNamespace

    from vllm.model_executor.models.registry import _ModelRegistry

    registry = _ModelRegistry()

    def forbidden(*args, **kwargs):
        pytest.fail("Excluded identity reached dynamic resolution")

    monkeypatch.setattr(registry, "_try_resolve_transformers", forbidden)
    monkeypatch.setattr(registry, "_normalize_arch", forbidden)
    config = SimpleNamespace(
        model_impl=implementation,
        trust_remote_code=True,
        hf_config=SimpleNamespace(
            architectures=["LlamaForCausalLM"],
            auto_map={"AutoModel": "custom.Model"},
        ),
        hf_overrides={"architectures": ["LlamaForCausalLM"]},
    )
    with pytest.raises(ValueError, match="LlamaForCausalLM.*not approved"):
        getattr(registry, method)(config.hf_config.architectures, config)


@pytest.mark.parametrize("method", ["inspect_model_cls", "resolve_model_cls"])
@pytest.mark.parametrize("implementation", ["auto", "transformers"])
@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_admitted_fallback_preserves_remote_code_trust(
    monkeypatch, method, implementation, trust_remote_code
):
    """Family admission must not grant permission to execute remote model code."""
    from types import SimpleNamespace

    import vllm.model_executor.models.registry as registry_module

    registry = registry_module._ModelRegistry()
    calls = []

    def dynamic_lookup(module, model, **kwargs):
        calls.append((module, kwargs["trust_remote_code"]))
        return None

    monkeypatch.setattr(
        registry_module, "try_get_class_from_dynamic_module", dynamic_lookup
    )
    sentinel = object()
    monkeypatch.setattr(registry, "_try_inspect_model_cls", lambda arch: sentinel)
    monkeypatch.setattr(registry, "_try_load_model_cls", lambda arch: sentinel)
    config = SimpleNamespace(
        model_impl=implementation,
        model="local-test-checkpoint",
        revision=None,
        code_revision=None,
        trust_remote_code=trust_remote_code,
        hf_config=SimpleNamespace(
            auto_map={"AutoConfig": "custom.Config", "AutoModel": "custom.Model"}
        ),
        _get_transformers_backend_cls=lambda: "TransformersForCausalLM",
    )
    assert getattr(registry, method)("Qwen3ForCausalLM", config) == (
        sentinel,
        "TransformersForCausalLM",
    )
    assert calls == [
        ("custom.Config", trust_remote_code),
        ("custom.Model", trust_remote_code),
    ]


@pytest.mark.parametrize("implementation", ["auto", "vllm", "transformers"])
@pytest.mark.parametrize("override_kind", ["dict", "callable"])
def test_model_config_rejects_excluded_effective_architecture(
    tmp_path, implementation, override_kind
):
    """Actual config loading applies overrides before the common admission gate."""
    import json

    from vllm.config import ModelConfig

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "max_position_embeddings": 128,
            }
        ),
        encoding="utf-8",
    )

    def override(config):
        config.architectures = ["LlamaForCausalLM"]
        return config

    overrides = (
        override
        if override_kind == "callable"
        else {"architectures": ["LlamaForCausalLM"]}
    )
    with pytest.raises(ValueError, match="LlamaForCausalLM.*not approved"):
        ModelConfig(
            model=str(tmp_path),
            tokenizer=str(tmp_path),
            skip_tokenizer_init=True,
            model_impl=implementation,
            hf_overrides=overrides,
        )


def test_fresh_registry_process_preserves_admission():
    """A fresh interpreter must enforce the same gate after plugin registration."""
    from types import SimpleNamespace

    from vllm.model_executor.models.registry import _run_in_subprocess

    def resolve_in_fresh_process():
        import torch

        from vllm.model_executor.models.registry import _ModelRegistry

        registry = _ModelRegistry()
        registry.register_model("Qwen3ForCausalLM", torch.nn.Linear)
        registry.register_model("UnapprovedPluginAlias", torch.nn.Linear)
        outcomes = []
        for implementation in ("auto", "vllm", "transformers"):
            for method in ("inspect_model_cls", "resolve_model_cls"):
                try:
                    getattr(registry, method)(
                        ["Qwen3ForCausalLM", "UnapprovedPluginAlias"],
                        SimpleNamespace(model_impl=implementation),
                    )
                except ValueError as error:
                    outcomes.append("not approved" in str(error))
                else:
                    outcomes.append(False)
        result, arch = registry.resolve_model_cls(
            "Qwen3ForCausalLM", SimpleNamespace(model_impl="vllm")
        )
        return outcomes, result is torch.nn.Linear, arch

    assert _run_in_subprocess(resolve_in_fresh_process) == (
        [True] * 6,
        True,
        "Qwen3ForCausalLM",
    )


@pytest.mark.parametrize("implementation", ["auto", "vllm", "transformers"])
def test_model_config_keeps_qwen_native_and_transformers_paths(
    tmp_path, implementation
):
    """Retained checkpoints complete real config initialization on both backends."""
    from transformers import Qwen3Config

    from vllm.config import ModelConfig

    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=128,
    ).save_pretrained(tmp_path)
    config = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        skip_tokenizer_init=True,
        model_impl=implementation,
        enforce_eager=True,
    )
    expected = (
        "TransformersForCausalLM"
        if implementation == "transformers"
        else "Qwen3ForCausalLM"
    )
    assert config.architectures == ["Qwen3ForCausalLM"]
    assert config.architecture == expected
    model_cls, architecture = config.registry.resolve_model_cls(
        config.architectures, config
    )
    assert architecture == expected
    assert model_cls is not None


@pytest.mark.parametrize("method", ["ngram", "extract_hidden_states"])
def test_real_qwen_speculative_config_lifecycle(tmp_path, method):
    """Shared proposers retain target identity through real config construction."""
    from transformers import Qwen3Config

    from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig

    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=128,
    ).save_pretrained(tmp_path)
    target = ModelConfig(
        model=str(tmp_path), skip_tokenizer_init=True, enforce_eager=True
    )
    speculative = SpeculativeConfig(
        method=method,
        num_speculative_tokens=1,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
    )
    assert target.architectures == ["Qwen3ForCausalLM"]
    if method == "ngram":
        assert speculative.draft_model_config is target
    else:
        assert speculative.draft_model_config is not target
        assert speculative.draft_model_config.architectures == [
            "ExtractHiddenStatesModel"
        ]


@pytest.mark.parametrize("method", ["inspect_model_cls", "resolve_model_cls"])
@pytest.mark.parametrize("target", ["Qwen3ForCausalLM", "LlamaForCausalLM", None])
@pytest.mark.parametrize(
    "architecture", ["PEagleDraftModel", "PeagleLlamaForCausalLM", "DFlashDraftModel"]
)
@pytest.mark.parametrize("runner", ["draft", "generate"])
def test_shared_peagle_requires_actual_target_context(
    method, target, architecture, runner
):
    from types import SimpleNamespace

    config = SimpleNamespace(
        model_impl="vllm",
        runner=runner,
        spec_target_architectures=(() if target is None else (target,)),
    )
    if target != "Qwen3ForCausalLM" or runner != "draft":
        with pytest.raises(ValueError, match="not approved"):
            getattr(ModelRegistry, method)(architecture, config)
    else:
        result, arch = getattr(ModelRegistry, method)(architecture, config)
        assert result is not None
        assert arch == architecture


@pytest.mark.parametrize(
    "target,accepted",
    [
        ("Qwen3_5ForConditionalGeneration", True),
        ("Qwen3ForCausalLM", False),
        ("LlamaForCausalLM", False),
    ],
)
def test_dflash2_uses_explicit_target_pair(target, accepted):
    from vllm.model_executor.models.registry import validate_four_family_speculation

    if accepted:
        validate_four_family_speculation([target], ["DFlash2DraftModel"])
    else:
        with pytest.raises(ValueError, match="not approved"):
            validate_four_family_speculation([target], ["DFlash2DraftModel"])


def test_four_family_dummy_plugin_behavior(monkeypatch):
    """Exercise plugin contracts without loading weights or running attention."""
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm.model_executor.models.registry import _ModelRegistry
    from vllm.sequence import IntermediateTensors

    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "plugins" / "vllm_add_dummy_model")
    )
    import vllm_add_dummy_model as plugin
    from vllm_add_dummy_model.my_qwen3 import MyQwen3ForCausalLM
    from vllm_add_dummy_model.my_qwen_embedding import MyQwen2Embedding
    from vllm_add_dummy_model.my_qwen_vl import MyQwen2VL
    from vllm.model_executor.models.qwen2_vl import Qwen2VLForConditionalGeneration

    registry = _ModelRegistry()
    monkeypatch.setattr(plugin, "ModelRegistry", registry)
    plugin.register()
    config = SimpleNamespace(model_impl="vllm")
    for arch, cls in (
        ("Qwen3ForCausalLM", MyQwen3ForCausalLM),
        ("Qwen2ForCausalLM", MyQwen2Embedding),
        ("Qwen2VLForConditionalGeneration", MyQwen2VL),
    ):
        assert registry.resolve_model_cls(arch, config)[0] is cls
    for arch in ("MyOPTForCausalLM", "MyGemma2Embedding", "MyLlava"):
        with pytest.raises(ValueError, match="not approved"):
            registry.resolve_model_cls(arch, config)

    embedding = object.__new__(MyQwen2Embedding)
    torch.nn.Module.__init__(embedding)
    embedding.model = Mock(return_value=torch.ones(2, 4))
    assert torch.equal(embedding(None, torch.arange(2)), torch.zeros(2, 4))
    intermediate = IntermediateTensors({"hidden_states": torch.ones(2, 4)})
    embedding.model.return_value = intermediate
    assert embedding(None, torch.arange(2)) is intermediate
    embedding.model.load_weights.side_effect = lambda weights: dict(weights)
    weight = torch.ones(4)
    loaded = embedding.load_weights(
        [("model.norm.weight", weight), ("lm_head.weight", weight)]
    )
    assert list(loaded) == ["norm.weight"]
    assert loaded["norm.weight"] is weight

    multimodal = object.__new__(MyQwen2VL)
    monkeypatch.setattr(
        Qwen2VLForConditionalGeneration,
        "compute_logits",
        lambda self, hidden: torch.ones(2, 8),
    )
    logits = multimodal.compute_logits(torch.zeros(2, 4))
    assert torch.equal(logits[:, 0], torch.ones(2))
    assert torch.equal(logits[:, 1:], torch.zeros(2, 7))
    monkeypatch.setattr(
        Qwen2VLForConditionalGeneration, "compute_logits", lambda self, hidden: None
    )
    assert multimodal.compute_logits(torch.zeros(2, 4)) is None
