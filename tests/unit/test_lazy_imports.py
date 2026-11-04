"""``import economize`` imports no optional dependency: serving engines, kernel compilers and quantization libraries
are imported where they are first used (a technique's backend, a serving adapter, a kernel's first launch)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

#: Top-level packages ``import economize`` must not import (installed or not).
OPTIONAL = (
    "vllm",
    "sglang",
    "triton",
    "torchao",
    "flash_attn",
    "flashinfer",
    "bitsandbytes",
    "kernels",
    "compressed_tensors",
    "xformers",
    "awq",
    "auto_gptq",
    "gptqmodel",
    # the export runtimes (economize.backends.gguf, .onnx, .mlx, .trtllm)
    "gguf",
    "llama_cpp",
    "onnx",
    "onnxruntime",
    "onnxscript",
    "mlx",
    "tensorrt_llm",
)

_PROBE = """
import json, sys
{code}
print(json.dumps(sorted(sys.modules)))
"""

#: Imports the given modules (those of ``transformers`` that the probed code loaded) and nothing else.
_BASELINE = """
import importlib, json, sys
for name in json.loads(sys.argv[1]):
    importlib.import_module(name)
print(json.dumps(sorted(sys.modules)))
"""

#: Asking which techniques combine and where they run reads declarations only, whatever the backend.
_METADATA = """
import economize
economize.compatibility_matrix()
for name in economize.TECHNIQUES.names():
    economize.backend_support(name)
"""


def _modules(*args: str) -> list[str]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    result = subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env, check=True)
    return list(json.loads(result.stdout.strip().splitlines()[-1]))


def _top_level(modules: list[str]) -> set[str]:
    return {name.split(".")[0] for name in modules}


def _pulled(code: str) -> set[str]:
    loaded = _modules("-c", _PROBE.format(code=code))
    used = [name for name in loaded if name == "transformers" or name.startswith("transformers.")]
    baseline = _top_level(_modules("-c", _BASELINE, json.dumps(used))) if used else set()
    return (_top_level(loaded) - baseline) & set(OPTIONAL)


@pytest.mark.parametrize(
    "code",
    ["import economize", "import economize.techniques", "import economize.attention", _METADATA],
    ids=["economize", "economize.techniques", "economize.attention", "metadata"],
)
def test_import_pulls_in_no_optional_dependency(code: str) -> None:
    """Optional packages that the same ``transformers`` modules import on their own in this environment are
    discounted: transformers imports some of them when they are installed (``cache_utils`` imports ``hqq``, which
    imports ``triton``), which economize cannot prevent. The baseline is measured in a fresh interpreter that imports
    exactly the ``transformers`` modules the probed code loaded."""
    pulled = _pulled(code)
    assert not pulled, f"{code.strip()!r} imported {sorted(pulled)}"


def test_the_baseline_keeps_imports_of_economize_itself() -> None:
    """An optional package loaded next to economize (and not by transformers) is still reported. A stand-in module
    is placed in ``sys.modules``, so the check does not depend on what is installed."""
    code = "import economize, sys, types\nsys.modules['flash_attn'] = types.ModuleType('flash_attn')"
    assert _pulled(code) == {"flash_attn"}
