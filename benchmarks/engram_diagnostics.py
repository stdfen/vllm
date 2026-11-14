# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in eager tensor snapshots; never use this worker extension for timing."""

import dataclasses
import hashlib
import importlib
import json
import sys
from pathlib import Path

import regex as re
import torch

_ACTIVE = None
_PREVIOUS_EXTENSIONS = globals().get("_PREVIOUS_EXTENSIONS", [])
if "DiagnosticExtension" in globals():
    _PREVIOUS_EXTENSIONS.append(globals()["DiagnosticExtension"])


def _tensors(value, name="value"):
    if isinstance(value, torch.Tensor):
        yield name, value
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _tensors(child, f"{name}.{index}")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _tensors(child, f"{name}.{key}")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            child = getattr(value, field.name)
            if isinstance(child, torch.Tensor):
                yield f"{name}.{field.name}", child


class _Recorder:
    def __init__(self, label, rank, layers):
        self.directory = Path("/diagnostic") / label / f"rank{rank}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sequence = 0
        self.layers = layers
        self.full_topk_phases = set()
        self.full_hash = Path("/diagnostic/fullhash.enabled").exists()
        self.all_topk = Path("/diagnostic/alltopk.enabled").exists()

    def record(self, module, phase, value):
        match = re.match(r"layers\.(\d+)\.", module)
        if match and int(match[1]) not in self.layers:
            return
        snapshots, metadata = {}, {}
        for key, tensor in _tensors(value):
            if tensor.numel() == 0 or tensor.is_meta:
                continue
            full_topk = module == "topk-prefill" and (
                self.all_topk or phase not in self.full_topk_phases
            )
            selected = tensor if full_topk else (tensor[-1:] if tensor.ndim else tensor)
            cpu = selected.detach().contiguous().cpu().clone()
            snapshots[key] = cpu
            raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
            metadata[key] = dict(
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
            if self.full_hash or module in ("layers.0", "layers.1", "layers.2"):
                full = tensor.detach().contiguous().cpu()
                full_raw = full.reshape(-1).view(torch.uint8).numpy().tobytes()
                metadata[key]["full_sha256"] = hashlib.sha256(full_raw).hexdigest()
        if not snapshots:
            return
        if module == "topk-prefill":
            self.full_topk_phases.add(phase)
        self.sequence += 1
        filename = f"{self.sequence:04d}.pt"
        torch.save(snapshots, self.directory / filename)
        with (self.directory / "records.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        module=module,
                        phase=phase,
                        tensors=metadata,
                        file=filename,
                    )
                )
                + "\n"
            )


def _pre(name):
    def hook(module, args):
        if _ACTIVE is not None:
            _ACTIVE.record(name, "input", args)

    return hook


def _post(name):
    def hook(module, args, result):
        if _ACTIVE is not None:
            _ACTIVE.record(name, "output", result)

    return hook


class DiagnosticExtension:
    def diagnostic_reload(self):
        importlib.reload(sys.modules[__name__])
        return {"diagnostic_module_reloaded": True}

    def diagnostic_topk_probe(self, label="a05capture-c000"):
        from vllm import _custom_ops as ops
        from vllm.distributed import get_tensor_model_parallel_rank

        if not re.fullmatch(r"[a-zA-Z0-9_-]+", label):
            raise ValueError("Invalid snapshot label")
        rank = get_tensor_model_parallel_rank()
        root = Path("/diagnostic")
        directory = root / label / f"rank{rank}"
        records = [
            json.loads(x)
            for x in (directory / "records.jsonl").read_text().splitlines()
        ]
        record = next(
            x
            for x in records
            if x["module"] == "topk-prefill" and x["phase"] == "input"
        )
        values = torch.load(
            directory / record["file"], map_location="cpu", weights_only=True
        )
        logits = values["value.0"].cuda().contiguous()
        starts, ends = values["value.1"].cuda(), values["value.2"].cuda()
        assert logits.shape[0] == 1 and int(starts[0]) == 0
        outputs = []
        for _ in range(30):
            result = torch.full((1, 512), -1, dtype=torch.int32, device="cuda")
            ops.top_k_per_row_prefill(
                logits, starts, ends, result, 1, logits.stride(0), 1, 512
            )
            outputs.append(result.cpu()[0].tolist())
        reference = torch.argsort(
            logits[0, : int(ends[0])], descending=True, stable=True
        )[:512]
        cutoff = logits[0, reference[-1]]
        summary = dict(
            rank=rank,
            source=label,
            repeats=30,
            distinct_orders=len({tuple(x) for x in outputs}),
            distinct_sets=len({tuple(sorted(x)) for x in outputs}),
            threshold_ties=int((logits[0, : int(ends[0])] == cutoff).sum()),
            selected_indices=outputs,
        )
        (root / f"{label}-topk-probe-r{rank}.json").write_text(
            json.dumps(summary, indent=2)
        )
        return {k: v for k, v in summary.items() if k != "selected_indices"}

    def init_worker_extension(self):
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.models.deepseek_v41.nvidia.model import DeepseekV4Model

        original = DeepseekV4Model.forward

        def forward(model, *args, **kwargs):
            global _ACTIVE
            arm = Path("/diagnostic/arm.json")
            if not arm.exists() or torch.cuda.is_current_stream_capturing():
                return original(model, *args, **kwargs)
            specification = json.loads(arm.read_text())
            label = specification["label"]
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", label):
                raise ValueError("Invalid diagnostic snapshot label")
            if not hasattr(model, "_engram_diagnostic_handles"):
                handles = []
                for name, module in model.named_modules():
                    if (
                        name == "embed_tokens"
                        or re.fullmatch(r"layers\.\d+", name)
                        or name.startswith("layers.")
                    ):
                        handles.append(module.register_forward_pre_hook(_pre(name)))
                        handles.append(module.register_forward_hook(_post(name)))
                model._engram_diagnostic_handles = handles
            if not hasattr(model, "_engram_diagnostic_calls"):
                model._engram_diagnostic_calls = {}
            count = model._engram_diagnostic_calls.get(label, 0)
            model._engram_diagnostic_calls[label] = count + 1
            _ACTIVE = _Recorder(
                f"{label}-c{count:03d}",
                get_tensor_model_parallel_rank(),
                specification.get("layers", [0, 1, 2, 14]),
            )
            try:
                _ACTIVE.record("model", "input", (args, kwargs))
                result = original(model, *args, **kwargs)
                _ACTIVE.record("model", "output", result)
                return result
            finally:
                _ACTIVE = None

        DeepseekV4Model.forward = forward


def _install_topk_probe():
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

    current = ops.top_k_per_row_prefill
    original = getattr(current, "_diagnostic_original", current)
    stable_prefill = Path("/diagnostic/stable-topk.enabled").exists()

    def topk(logits, starts, ends, output, *args):
        if _ACTIVE is not None:
            _ACTIVE.record("topk-prefill", "input", (logits, starts, ends))
        result = original(logits, starts, ends, output, *args)
        if stable_prefill:
            # Diagnostic C1 reference only; preserve exact FP32 scores and use
            # increasing logical index to resolve equal scores.
            if bool(torch.any(starts != 0)):
                raise ValueError("Stable diagnostic top-k requires zero row starts")
            columns = torch.arange(logits.shape[1], device=logits.device)
            masked = logits.masked_fill(
                columns[None, :] >= ends[:, None], -float("inf")
            )
            indices = torch.argsort(masked, dim=-1, descending=True, stable=True)
            indices = indices[:, : output.shape[1]]
            output.copy_(indices.masked_fill(indices >= ends[:, None], -1))
        if _ACTIVE is not None:
            _ACTIVE.record("topk-prefill", "output", output)
        return result

    topk._diagnostic_original = original
    ops.top_k_per_row_prefill = topk

    current_forward = SparseAttnIndexer.forward
    original_forward = getattr(current_forward, "_diagnostic_original", current_forward)
    canonical = Path("/diagnostic/canonical-topk.enabled").exists()

    def forward(module, hidden_states, *args, **kwargs):
        output = original_forward(module, hidden_states, *args, **kwargs)
        if canonical:
            live = output[: hidden_states.shape[0]]
            sentinel = torch.iinfo(live.dtype).max
            ordered = live.masked_fill(live < 0, sentinel).sort(dim=-1).values
            live.copy_(ordered.masked_fill(ordered == sentinel, -1))
        return output

    forward._diagnostic_original = original_forward
    SparseAttnIndexer.forward = forward


_install_topk_probe()

for _previous in _PREVIOUS_EXTENSIONS:
    for _name in ("diagnostic_reload", "diagnostic_topk_probe"):
        setattr(_previous, _name, getattr(DiagnosticExtension, _name))
