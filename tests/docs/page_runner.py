"""Runs the Python blocks of a documentation page in order, on a CPU in float32 with small step counts.

A page's blocks share one namespace, as a reader running the page top to bottom would. Three comments steer a run
and are invisible on the rendered page:

* ``<!-- not run: reason -->`` right before a block skips it (it needs several GPUs, a download of a large model, a
  serving engine, ...);
* ``<!-- test-substitute: PATTERN => REPLACEMENT -->`` anywhere on the page adds a regular-expression rewrite to the
  :data:`SUBSTITUTIONS` of that page (to shrink a size the shared rewrites do not know);
* a line ``--8<-- "docs/snippets/<file>.md"`` includes a snippet, as ``pymdownx.snippets`` does when the site is built
  (paths are relative to the repository root).

The rewrites change sizes and devices, never an economize call.
"""

from __future__ import annotations

import os
import re
import sys
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

_BLOCK = re.compile(r"(?P<marker><!-- not run: (?P<reason>[^>]*) -->\s*)?```(?P<lang>\w*)\n(?P<code>.*?)```", re.S)
_SNIPPET = re.compile(r'^--8<-- "(?P<path>[^"]+)"\s*$', re.M)
_SUBSTITUTE = re.compile(r"<!-- test-substitute: (?P<pattern>.*?) => (?P<replacement>.*?) -->")

#: Rewrites that make a block run on a CPU in seconds. They change sizes and devices, never the API calls.
SUBSTITUTIONS: list[tuple[str, str]] = [
    (r"dtype=torch\.bfloat16", "dtype=torch.float32"),
    (r',\s*device_map="(cuda|auto)"', ""),
    (r"\.cuda\(\)", ""),
    (r'device="cuda"', 'device="cpu"'),
    (r'autocast="bfloat16"', "autocast=None"),
    (r'dtype="bfloat16"', 'dtype="float32"'),
    (r"steps=2000", "steps=2"),
    (r"max_steps=2000", "max_steps=2"),
    (r"warmup_steps=100", "warmup_steps=1"),
    (r"save_every=500", "save_every=1"),
    (r"save_steps=500", "save_steps=1"),
    (r"seq_len=1024", "seq_len=64"),
    (r"batch_size=8", "batch_size=2"),
    (r"fixed_windows\(\"val\.bin\", 64, 1024\)", 'fixed_windows("val.bin", 2, 192)'),
    (r"view\(128, 2048\)", "view(4, 64)"),
    (r"\[: 128 \* 2048\]", "[: 4 * 64]"),
    (r"max_new_tokens=\d+", "max_new_tokens=6"),
    (r"(\.decode\([^,()]+(?:\[[^\]]*\])?(?:\.cuda\(\))?), \d+", r"\1, 6"),
    (r"decode_batch\(prompts, \[64, 32, 48\]\)", "decode_batch(prompts, [6, 4, 5])"),
    (r"decode_batch\(prompts, 32", "decode_batch(prompts, 6"),
    (r"limit=4\)", "limit=2)"),
    (r"select\(range\(64\)\)", "select(range(2))"),
    (r"texts\[:64\]", "texts[:4]"),
    (
        r'load_dataset\("openai/gsm8k", "main", split="train"\)',
        'load_dataset("openai/gsm8k", "main", split="train[:300]")',
    ),
    (r"num_skip=8", "num_skip=2"),
    (r"TealConfig\(sparsity=0\.4\)", "TealConfig(sparsity=0.4, layers=(1,), greedy_step=0.2)"),
    (r"vocab_size=32768", "vocab_size=4096"),
    (r"draft_vocab=32768", "draft_vocab=4096"),
    (r"\[:32768\]", "[:4096]"),
    (r"Qwen/Qwen2\.5-7B-Instruct", "Qwen/Qwen2.5-0.5B"),
    (r"Qwen/Qwen2\.5-7B", "Qwen/Qwen2.5-0.5B-Instruct"),
]


@dataclass(frozen=True)
class Block:
    """One Python block of a page.

    Attributes:
        line: Line of the block's first code line in the page (snippets expanded).
        code: The code as written on the page.
        skip: The reason of its ``not run`` marker, or ``None`` for a block that runs.
    """

    line: int
    code: str
    skip: str | None


def expand_snippets(text: str, root: Path = ROOT) -> str:
    """``text`` with every ``--8<-- "path"`` line replaced by the file it names (relative to ``root``)."""
    return _SNIPPET.sub(lambda match: expand_snippets((root / match["path"]).read_text(), root), text)


def blocks(page: Path) -> list[Block]:
    """The Python blocks of ``page`` (snippets expanded)."""
    text = expand_snippets(page.read_text())
    found = []
    for match in _BLOCK.finditer(text):
        if match["lang"] != "python":
            continue
        line = text.count("\n", 0, match.start("code")) + 1
        found.append(Block(line, match["code"], match["reason"]))
    return found


def page_substitutions(page: Path) -> list[tuple[str, str]]:
    """The ``test-substitute`` rewrites of ``page``, in page order."""
    return [(match["pattern"], match["replacement"]) for match in _SUBSTITUTE.finditer(page.read_text())]


def adapt(code: str, extra: list[tuple[str, str]] | None = None) -> str:
    """``code`` with the page's rewrites (``extra``, applied first) and :data:`SUBSTITUTIONS` applied."""
    for pattern, replacement in [*(extra or []), *SUBSTITUTIONS]:
        code = re.sub(pattern, replacement, code)
    return code


def run_page(page: Path, workdir: Path, keep_going: bool = False) -> list[tuple[int, str]]:
    """Run the blocks of ``page`` in ``workdir``; return ``(line, traceback)`` of the failed blocks."""
    os.chdir(workdir)
    # A real module, so that classes defined on the page (dataclasses, registered techniques) resolve their module.
    module = types.ModuleType(f"__docs_{page.stem}__")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    extra = page_substitutions(page)
    failures = []
    for block in blocks(page):
        where = f"{page.name}:{block.line}"
        if block.skip is not None:
            print(f"{where}: skipped ({block.skip})", flush=True)
            continue
        start = time.time()
        try:
            exec(compile(adapt(block.code, extra), where, "exec"), namespace)
        except Exception:
            failures.append((block.line, traceback.format_exc()))
            print(f"{where}: FAILED\n{failures[-1][1]}", flush=True)
            if not keep_going:
                break
        else:
            print(f"{where}: ok ({time.time() - start:.1f}s)", flush=True)
    return failures


def main(argv: list[str]) -> int:
    """``python tests/docs/page_runner.py PAGE [WORKDIR]``: run every block of one page and report all failures."""
    page = Path(argv[1]).resolve()
    work = Path(argv[2] if len(argv) > 2 else ".").resolve()
    work.mkdir(parents=True, exist_ok=True)
    return 1 if run_page(page, work, keep_going=True) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
