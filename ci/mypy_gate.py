"""The type gate of CI: mypy may report no finding that is not in the committed baseline.

economize is not yet clean under mypy with the current torch (newer torch annotates ``nn.Module.__getattr__`` as
``Tensor | Module``, so every untyped submodule access is a finding). Until it is, the gate is a ratchet: the findings
of ``ci/mypy_baseline.txt`` are tolerated, any other finding fails, and a fixed finding is reported so that it can be
deleted from the baseline. Findings are compared without their line numbers, so that an edit elsewhere in a file
does not invalidate the baseline.

Usage (from the repository root, with the environment of ``.github/workflows/ci.yml``)::

    python ci/mypy_gate.py                 # run the gate
    python ci/mypy_gate.py --write         # rewrite the baseline after fixing findings (never to add new ones)
    python ci/mypy_gate.py --mypy "uvx mypy@2.4.0 --python-executable $(which python)"
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).resolve().parent / "mypy_baseline.txt"
#: The pinned mypy (the same version as the ``dev`` extra and docs/CONTRIBUTING.md).
MYPY = f"uvx mypy@2.4.0 --python-executable {shlex.quote(sys.executable)}"

_FINDING = re.compile(r"^(?P<path>[^:\s]+\.py):\d+(?::\d+)?: error: (?P<text>.*)$")
_LINE_REFERENCE = re.compile(r"\bline \d+\b")


def findings(output: str) -> Counter[str]:
    """The findings in mypy's output as ``"<path>: <message> [<code>]"``, with line numbers removed."""
    found: Counter[str] = Counter()
    for line in output.splitlines():
        match = _FINDING.match(line.strip())
        if match:
            found[f"{match['path']}: {_LINE_REFERENCE.sub('line N', match['text'])}"] += 1
    return found


def load_baseline(path: Path = BASELINE) -> Counter[str]:
    """The tolerated findings (one per line; ``#`` starts a comment)."""
    if not path.exists():
        return Counter()
    lines = (line.strip() for line in path.read_text().splitlines())
    return Counter(line for line in lines if line and not line.startswith("#"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mypy", default=MYPY, help="the mypy command (default: %(default)s)")
    parser.add_argument("--write", action="store_true", help="replace the baseline with the current findings")
    args = parser.parse_args(argv)
    result = subprocess.run([*shlex.split(args.mypy), "src/economize"], cwd=ROOT, capture_output=True, text=True)
    current = findings(result.stdout)
    if result.returncode not in (0, 1) or (result.returncode == 1 and not current):
        print(result.stdout, result.stderr, sep="\n")
        print(f"mypy did not run (exit code {result.returncode})")
        return 2
    if args.write:
        header = "# Tolerated mypy findings, see ci/mypy_gate.py. Shrink this file; never add to it by hand.\n"
        BASELINE.write_text(header + "".join(f"{f}\n" for f in sorted(current.elements())))
        print(f"wrote {sum(current.values())} findings to {BASELINE.relative_to(ROOT)}")
        return 0
    baseline = load_baseline()
    new, fixed = current - baseline, baseline - current
    for finding in sorted(fixed.elements()):
        print(f"fixed (delete from the baseline): {finding}")
    for finding in sorted(new.elements()):
        print(f"NEW: {finding}")
    print(f"mypy: {sum(current.values())} findings, {sum(baseline.values())} tolerated, {sum(new.values())} new, "
          f"{sum(fixed.values())} fixed")
    return 1 if new else 0


if __name__ == "__main__":
    raise SystemExit(main())
