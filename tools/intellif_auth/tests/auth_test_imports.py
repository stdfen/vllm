# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load the standalone runtime locally or the installed candidate package."""

import importlib
import importlib.util
import pathlib
import sys


def load_runtime():
    if "intellif_auth" in sys.modules:
        return
    for parent in pathlib.Path(__file__).resolve().parents:
        package = parent / "vllm/intellif_auth"
        if (package / "__init__.py").is_file():
            spec = importlib.util.spec_from_file_location(
                "intellif_auth",
                package / "__init__.py",
                submodule_search_locations=[str(package)],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["intellif_auth"] = module
            spec.loader.exec_module(module)
            return
    sys.modules["intellif_auth"] = importlib.import_module("vllm.intellif_auth")


load_runtime()
