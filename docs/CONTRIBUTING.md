# Contributing to economize

## Checks before a change lands

Run them from the `economize/` directory; `.github/workflows/ci.yml` runs the same checks on every push and pull
request. The lint and type configuration lives in `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`); `uvx` runs both
tools without installing them into the project's environment. Both are pinned (the same versions as the `dev` extra of
`pyproject.toml`), because a new ruff or mypy release adds, removes or changes rules and would make the gate depend on
the day it runs. A version bump is a change of its own that updates the pins in both places and fixes what the new
version reports. The torch and transformers builds of CI are pinned in `ci/constraints.txt` for the same reason.

```bash
# Lint: must report "All checks passed!" for the library, the tests, the vendored Hub recipe, the CI scripts and the
# docs hooks.
uvx ruff@0.16.10 check src tests ci docs/hooks

# Types: must report "0 new". mypy needs the environment that has torch and transformers installed, so the gate
# points uvx's mypy at the current interpreter. Install that environment as CI does:
#   pip install -c ci/constraints.txt -e ".[test]" --extra-index-url https://download.pytorch.org/whl/cpu
python ci/mypy_gate.py

# Tests (CPU; GPU-only tests skip themselves, and so do the tests of an extra that is not installed: CI installs
# ".[test,train,peft]"). Run from economize/ or no tests are collected.
CUDA_VISIBLE_DEVICES= python -m pytest -q tests

# Docs (a change to docs/, mkdocs.yml or a docstring): must build without a warning. Needs the docs extra.
#   pip install -c ci/constraints.txt -e ".[docs]" --extra-index-url https://download.pytorch.org/whl/cpu
CUDA_VISIBLE_DEVICES= mkdocs build --strict
```

The type gate is a ratchet. Newer torch releases annotate `nn.Module.__getattr__` as `Tensor | Module`, which turns
every untyped submodule access into a mypy finding, and economize is not yet clean under them. `ci/mypy_baseline.txt`
lists the findings that are tolerated (by file and message, without line numbers); the gate fails on any other
finding and reports every baseline entry that no longer occurs. A change that fixes findings runs
`python ci/mypy_gate.py --write` to shrink the baseline; a change never adds to it, and a bump of the pinned torch or
mypy regenerates it in a change of its own. The baseline is only valid for the builds of `ci/constraints.txt`: other
torch versions report other findings.

A change that alters a technique's numbers on purpose regenerates its golden tensors
(`PYTHONPATH=src:tests python tests/golden/regenerate.py <case>`) and says why in the commit message; a new technique
adds a case to `tests/golden/catalog.py`.

The lint gate is `ruff check` only. economize does not run `ruff format`, so a change never reformats code it does not
touch. Safe fixes for import order and sorted `__all__` can be applied with
`uvx ruff@0.16.10 check --fix --select I001,RUF022 src tests`; every other finding is fixed by hand. A rule is switched
off only in `pyproject.toml`, with a comment that says why, and a single line is exempted only with a
`# noqa: <code>` that names the code (ruff's `RUF100` reports markers that no longer suppress anything).
Never auto-fix `RUF100` together with a narrow `--select`: ruff then counts every marker for an unselected code as
unused and deletes it.
A `# type: ignore[<code>]` likewise names its code and is reported by mypy once it is unused
(`warn_unused_ignores`).

## Design rules

These rules are explained in `docs/ARCHITECTURE.md`; the tests named here enforce them.

- **Extension, not modification.** A technique, architecture, format, verifier, exporter, objective, ... is one
  registered class (`docs/ARCHITECTURE.md` 3.12). A new one never edits an existing module. Exporters register with
  the decorator `@economize.backends.register_exporter(backend, technique)`.
- **Layering.** A module imports only from its own layer and the layers below it (`docs/ARCHITECTURE.md` 2.1,
  `tests/unit/test_import_layering.py`). `import economize` never loads `economize.integrations` or an optional library.
- **No private access across objects.** Code calls the public methods of other economize objects. When a third-party
  library offers no public route, the access to its private symbol checks that the symbol exists (or the library
  version) and carries a comment naming the upstream symbol and the versions it was checked against.
- **Deprecation, not breakage.** A moved or renamed public name keeps working at its old path through a module
  `__getattr__` that calls `economize._deprecation.warn_moved(old, new)` and returns the new object. Tests of the old
  path assert the `DeprecationWarning`.
- **Numerics and formats are fixed.** Refactors keep logits bit-identical (the oracle tests under `tests/oracle/`)
  and keep every checkpoint format readable.
- **Errors** derive from `economize.errors.EconomizeError` (and from the built-in exception they refine, such as
  `ValueError` for `ConfigError`), so callers can catch economize's errors as a group.
