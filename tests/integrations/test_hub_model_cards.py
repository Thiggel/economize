"""Model cards and uploads of ``economize.integrations.hub.push_to_hub`` (fake ``HfApi``, no network)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::FutureWarning")

REPO = "economize-tests/tiny-audit"


class _FakeHfApi:
    remote: Path

    def __init__(self, token: Any = None) -> None:
        self.token = token

    def create_repo(self, repo_id: str, **kwargs: Any) -> Any:
        return type("RepoUrl", (), {"repo_id": repo_id})()

    def upload_folder(self, *, repo_id: str, folder_path: str | Path, **kwargs: Any) -> str:
        shutil.copytree(folder_path, _FakeHfApi.remote, dirs_exist_ok=True)
        return f"https://huggingface.co/{repo_id}/commit/0123abcd"


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> type[_FakeHfApi]:
    import huggingface_hub.hub_mixin

    _FakeHfApi.remote = tmp_path / "remote"
    monkeypatch.setattr(huggingface_hub.hub_mixin, "HfApi", _FakeHfApi)
    return _FakeHfApi


def test_push_to_hub_accepts_model_card_kwargs_none(applied_jump: Any, fake_hub: type[_FakeHfApi]) -> None:
    from economize.integrations import hub

    url = hub.push_to_hub(applied_jump(), REPO, model_card_kwargs=None)
    assert url.startswith(f"https://huggingface.co/{REPO}/commit/")
    assert f'hub.from_pretrained("{REPO}"' in (fake_hub.remote / "README.md").read_text()


def test_push_to_hub_leaves_the_callers_card_kwargs_alone(applied_jump: Any, fake_hub: type[_FakeHfApi]) -> None:
    from economize.integrations import hub

    card_kwargs: dict[str, Any] = {}
    hub.push_to_hub(applied_jump(), REPO, model_card_kwargs=card_kwargs)
    assert card_kwargs == {}


def test_hub_model_view_names_its_repo_in_the_card(applied_jump: Any, fake_hub: type[_FakeHfApi]) -> None:
    from economize.integrations.hub import HubModel

    HubModel(applied_jump()).push_to_hub(REPO)
    readme = (fake_hub.remote / "README.md").read_text()
    assert f'hub.from_pretrained("{REPO}"' in readme and "<this repo" not in readme
