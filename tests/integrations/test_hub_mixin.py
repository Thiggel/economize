"""``economize.integrations.hub``: checkpoint files, model card, local round trip, and the Hub calls against a fake API.

Nothing here talks to the real Hub: ``HfApi`` and ``snapshot_download`` are replaced by local fakes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from economize.core.hub import REPO_PATTERNS
from helpers import make_tiny_model, perturb

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::FutureWarning")

REPO = "economize-tests/tiny-qwen2-jump"


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny_base")
    make_tiny_model().save_pretrained(path)
    return path


def _base(path: Path, name: str | None = None) -> Any:
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    base.requires_grad_(False)
    if name is not None:
        base.config._name_or_path = name
    return base


def _states(model: Any) -> dict[str, torch.Tensor]:
    return {f"{t.name}.{k}": v for t in model.techniques.values() for k, v in t.state_dict().items()}


def _assert_same_techniques(a: Any, b: Any) -> None:
    left, right = _states(a), _states(b)
    assert left.keys() == right.keys() and left
    assert all(torch.equal(left[key], right[key]) for key in left)
    assert [t.config.to_dict() for t in a.techniques.values()] == [t.config.to_dict() for t in b.techniques.values()]


class FakeHfApi:
    """Stands in for ``huggingface_hub.HfApi`` inside ``ModelHubMixin.push_to_hub``; uploads copy into ``remote``."""

    remote: Path
    calls: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, token: Any = None) -> None:
        self.token = token

    def create_repo(self, repo_id: str, **kwargs: Any) -> Any:
        FakeHfApi.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))
        return type("RepoUrl", (), {"repo_id": repo_id})()

    def upload_folder(self, *, repo_id: str, folder_path: str | Path, **kwargs: Any) -> str:
        FakeHfApi.calls.append(("upload_folder", {"repo_id": repo_id, **kwargs}))
        shutil.copytree(folder_path, FakeHfApi.remote, dirs_exist_ok=True)
        return f"https://huggingface.co/{repo_id}/commit/0123abcd"


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> type[FakeHfApi]:
    import huggingface_hub.hub_mixin

    FakeHfApi.remote = tmp_path / "remote"
    FakeHfApi.calls = []
    monkeypatch.setattr(huggingface_hub.hub_mixin, "HfApi", FakeHfApi)
    return FakeHfApi


# ---- files -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("adaptation", ["frozen", "lora"])
def test_save_pretrained_writes_economize_files_and_card(tiny_dir: Path, applied_jump: Any, tmp_path: Path,
                                                      adaptation: str) -> None:
    from economize.core.serialization import WEIGHTS_FILE
    from economize.integrations import hub

    model = applied_jump(_base(tiny_dir), adaptation=adaptation)
    assert hub.save_pretrained(model, tmp_path) is None
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["README.md", "economize_config.json", WEIGHTS_FILE])
    meta = json.loads((tmp_path / "economize_config.json").read_text())
    assert meta["base_model"] == str(tiny_dir)
    assert meta["techniques"][0]["technique"] == "jump"
    assert meta["techniques"][0]["config"]["adaptation"] == adaptation
    assert meta["teacher_fingerprint"] == hub.teacher_fingerprint(model) and meta["base_quantization"] is None
    assert "base_model_revision" in meta and "decoding" in meta
    from safetensors import safe_open

    with safe_open(str(tmp_path / WEIGHTS_FILE), "pt") as f:
        keys = list(f.keys())
    assert keys and all(key.startswith("jump.") for key in keys)  # technique weights only, no base weights
    assert not any("embed_tokens" in key or key.startswith("jump.model.") for key in keys)


def test_local_round_trip_restores_techniques(tiny_dir: Path, applied_jump: Any, tmp_path: Path) -> None:
    from economize.integrations import hub

    model = applied_jump(_base(tiny_dir), adaptation="lora")
    hub.save_pretrained(model, tmp_path)
    onto_module = hub.from_pretrained(tmp_path, _base(tiny_dir))
    _assert_same_techniques(model, onto_module)
    by_name = hub.from_pretrained(tmp_path, dtype=torch.float32)  # loads the base the checkpoint names
    _assert_same_techniques(model, by_name)
    ids = torch.arange(1, 9)[None]
    with torch.no_grad():
        assert torch.equal(model.base_model(ids).logits, by_name.base_model(ids).logits)
    assert type(by_name).__name__ == type(model).__name__


def test_loading_onto_different_weights_warns(tiny_dir: Path, applied_jump: Any, tmp_path: Path) -> None:
    from economize.integrations import hub

    hub.save_pretrained(applied_jump(_base(tiny_dir)), tmp_path)
    other = _base(tiny_dir)
    perturb(other, std=1e-3)
    with pytest.warns(UserWarning, match="weights differ"):
        hub.from_pretrained(tmp_path, other)
    with warnings_as_errors():
        hub.from_pretrained(tmp_path, _base(tiny_dir))
        hub.from_pretrained(tmp_path, other, check_base=False)


class warnings_as_errors:
    """Fail on any ``UserWarning`` inside the block."""

    def __enter__(self) -> None:
        import warnings

        self._catch = warnings.catch_warnings()
        self._catch.__enter__()
        warnings.simplefilter("error", UserWarning)

    def __exit__(self, *exc: Any) -> None:
        self._catch.__exit__(*exc)


def test_teacher_fingerprint_tracks_weights_and_wider_casts() -> None:
    from economize.integrations.hub import teacher_fingerprint

    base = make_tiny_model()
    print_ = teacher_fingerprint(base)
    assert print_.startswith("sha256:") and print_ == teacher_fingerprint(make_tiny_model())
    assert print_ == teacher_fingerprint(make_tiny_model().double())
    assert print_ != teacher_fingerprint(make_tiny_model().bfloat16())
    assert print_ != teacher_fingerprint(make_tiny_model(seed=1))


# ---- model card --------------------------------------------------------------------------------------------------
def test_model_card_links_hub_base_and_tags(tiny_dir: Path, applied_jump: Any) -> None:
    from economize.integrations.hub import model_card

    model = applied_jump(_base(tiny_dir, name="Qwen/Qwen2.5-0.5B"))
    card = model_card(model, repo_id=REPO, license="apache-2.0", custom_generate="economize/jump-generate")
    data = card.data.to_dict()
    assert data["base_model"] == "Qwen/Qwen2.5-0.5B" and data["base_model_relation"] == "adapter"
    assert data["library_name"] == "economize" and data["pipeline_tag"] == "text-generation"
    assert {"economize", "economize-jump", "speculative-decoding"} <= set(data["tags"])
    assert data["license"] == "apache-2.0"
    assert f'hub.from_pretrained("{REPO}"' in card.text and 'custom_generate="economize/jump-generate"' in card.text
    assert "| `jump` |" in card.text


def test_model_card_does_not_link_local_bases(tiny_dir: Path, applied_jump: Any) -> None:
    from economize.integrations.hub import model_card

    card = model_card(applied_jump(_base(tiny_dir)))
    assert "base_model" not in card.data.to_dict() and card.data.library_name == "economize"
    assert str(tiny_dir) in card.text


# ---- Hub calls (fake API) ----------------------------------------------------------------------------------------
def test_push_to_hub_uploads_one_folder(tiny_dir: Path, applied_jump: Any, fake_hub: type[FakeHfApi]) -> None:
    from economize.core.serialization import WEIGHTS_FILE
    from economize.integrations import hub

    model = applied_jump(_base(tiny_dir, name="Qwen/Qwen2.5-0.5B"))
    url = hub.push_to_hub(model, REPO, private=True, commit_message="add drafter")
    assert url.startswith(f"https://huggingface.co/{REPO}/commit/")
    (create, create_kwargs), (upload, upload_kwargs) = fake_hub.calls
    assert create == "create_repo" and create_kwargs["private"] is True
    assert upload == "upload_folder" and upload_kwargs["commit_message"] == "add drafter"
    files = sorted(p.name for p in fake_hub.remote.iterdir())
    assert files == sorted(["README.md", "economize_config.json", WEIGHTS_FILE])
    readme = (fake_hub.remote / "README.md").read_text()
    assert "base_model: Qwen/Qwen2.5-0.5B" in readme and "library_name: economize" in readme
    assert f'hub.from_pretrained("{REPO}"' in readme


def test_hub_model_view_pushes_the_wrapped_model(applied_jump: Any, fake_hub: type[FakeHfApi]) -> None:
    from economize.integrations.hub import EconomizeHubMixin, HubModel

    view = HubModel(applied_jump())
    assert isinstance(view, EconomizeHubMixin)
    view.push_to_hub(REPO)
    assert (fake_hub.remote / "economize_config.json").is_file()
    assert not (fake_hub.remote / "config.json").exists()


def test_from_pretrained_downloads_only_checkpoint_files(tiny_dir: Path, applied_jump: Any, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    from economize.integrations import hub

    model = applied_jump(_base(tiny_dir))
    hub.save_pretrained(model, tmp_path / "snapshot")
    requests: list[dict[str, Any]] = []

    def fake_snapshot_download(repo_id: str, **kwargs: Any) -> str:
        requests.append({"repo_id": repo_id, **kwargs})
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    loaded = hub.from_pretrained(REPO, _base(tiny_dir), revision="v1", token=False)
    _assert_same_techniques(model, loaded)
    (request,) = requests
    assert request["repo_id"] == REPO and request["revision"] == "v1" and request["library_name"] == "economize"
    assert set(request["allow_patterns"]) == set(REPO_PATTERNS) and "*.safetensors" in REPO_PATTERNS


def test_recorded_base_revision_is_requested(tiny_dir: Path, applied_jump: Any, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    from economize.integrations import hub

    hub.save_pretrained(applied_jump(_base(tiny_dir)), tmp_path)
    meta = json.loads((tmp_path / "economize_config.json").read_text())
    meta["base_model_revision"] = "abc123"
    (tmp_path / "economize_config.json").write_text(json.dumps(meta))
    seen: dict[str, Any] = {}
    original = transformers.AutoModelForCausalLM.from_pretrained

    def spy(name: Any, **kwargs: Any) -> Any:
        seen.update(kwargs, name=name)
        kwargs.pop("revision", None)  # a local folder has no revisions
        return original(name, **kwargs)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", spy)
    hub.from_pretrained(tmp_path, check_base=False)
    assert seen["revision"] == "abc123" and seen["name"] == str(tiny_dir)
    hub.from_pretrained(tmp_path, str(tiny_dir), base_revision="main", check_base=False)
    assert seen["revision"] == "main"
