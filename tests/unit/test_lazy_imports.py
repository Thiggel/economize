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
    "hqq",
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

#: Asking which techniques combine and where they run reads declarations only, whatever the backend.
_METADATA = """
import economize
economize.compatibility_matrix()
for name in economize.TECHNIQUES.names():
    economize.backend_support(name)
"""


def _pulled(code: str) -> set[str]:
    """The optional top-level packages in ``sys.modules`` after running ``code`` in a fresh interpreter."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    probe = _PROBE.format(code=code)
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True)
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    return {name.split(".")[0] for name in loaded} & set(OPTIONAL)


@pytest.mark.parametrize(
    "code",
    ["import economize", "import economize.techniques", "import economize.attention", _METADATA],
    ids=["economize", "economize.techniques", "economize.attention", "metadata"],
)
def test_import_pulls_in_no_optional_dependency(code: str) -> None:
    """Nothing is discounted: economize also defers the ``transformers`` modules that import optional kernels when
    they are installed (``transformers.cache_utils`` imports ``hqq``, which imports ``triton``)."""
    pulled = _pulled(code)
    assert not pulled, f"{code.strip()!r} imported {sorted(pulled)}"


def test_the_probe_reports_an_optional_package() -> None:
    """An optional package loaded next to economize is reported. A stand-in module is placed in ``sys.modules``, so
    the check does not depend on what is installed."""
    code = "import economize, sys, types\nsys.modules['flash_attn'] = types.ModuleType('flash_attn')"
    assert _pulled(code) == {"flash_attn"}
