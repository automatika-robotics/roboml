"""CI-safe tests: validate checkpoint source resolution (huggingface/modelscope)."""

import inspect
import sys
import types

import pytest

from roboml.utils import (
    CheckpointSource,
    MODELSCOPE_ALTERNATIVES,
    get_checkpoint_source,
    has_huggingface_credentials,
    has_modelscope_credentials,
    is_checkpoint_gated,
    resolve_checkpoint,
)


def _fake_hub_config(monkeypatch, token):
    """Install a fake modelscope_hub.config module with a given cached token."""
    fake_pkg = types.ModuleType("modelscope_hub")
    fake_cfg = types.ModuleType("modelscope_hub.config")

    def hub_config():
        return types.SimpleNamespace(token=token)

    fake_cfg.HubConfig = hub_config
    fake_pkg.config = fake_cfg
    monkeypatch.setitem(sys.modules, "modelscope_hub", fake_pkg)
    monkeypatch.setitem(sys.modules, "modelscope_hub.config", fake_cfg)


@pytest.fixture
def clean_env(monkeypatch):
    """Ensure ROBOML_SOURCE is not set."""
    monkeypatch.delenv("ROBOML_SOURCE", raising=False)


@pytest.fixture
def no_gating(monkeypatch):
    """Disable the gating hub lookup so resolver tests stay offline."""
    monkeypatch.setattr("roboml.utils.is_checkpoint_gated", lambda *a, **k: False)


@pytest.fixture
def fake_modelscope(monkeypatch):
    """Inject a fake modelscope module that records download calls."""
    fake = types.ModuleType("modelscope")
    calls = []

    def snapshot_download(checkpoint):
        calls.append(checkpoint)
        return f"/fake/cache/{checkpoint}"

    fake.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", fake)
    return calls


class TestGetCheckpointSource:
    def test_default_is_huggingface(self, clean_env):
        assert get_checkpoint_source(None) == CheckpointSource.HUGGINGFACE.value

    def test_env_var_sets_source(self, clean_env, monkeypatch):
        monkeypatch.setenv("ROBOML_SOURCE", "modelscope")
        assert get_checkpoint_source(None) == CheckpointSource.MODELSCOPE.value

    def test_param_overrides_env(self, clean_env, monkeypatch):
        monkeypatch.setenv("ROBOML_SOURCE", "modelscope")
        assert get_checkpoint_source("huggingface") == "huggingface"

    def test_invalid_param_raises(self, clean_env):
        with pytest.raises(ValueError, match="Invalid checkpoint source"):
            get_checkpoint_source("not_a_hub")

    def test_invalid_env_var_raises(self, clean_env, monkeypatch):
        monkeypatch.setenv("ROBOML_SOURCE", "not_a_hub")
        with pytest.raises(ValueError, match="ROBOML_SOURCE"):
            get_checkpoint_source(None)


class TestResolveCheckpoint:
    def test_huggingface_passthrough(self, clean_env, no_gating):
        assert resolve_checkpoint("Qwen/Qwen3-0.6B", "huggingface") == "Qwen/Qwen3-0.6B"

    def test_default_source_passthrough(self, clean_env, no_gating):
        assert resolve_checkpoint("Qwen/Qwen3-0.6B") == "Qwen/Qwen3-0.6B"

    def test_modelscope_returns_local_dir(self, clean_env, no_gating, fake_modelscope):
        result = resolve_checkpoint("Qwen/Qwen3-0.6B", "modelscope")
        assert result == "/fake/cache/Qwen/Qwen3-0.6B"
        assert fake_modelscope == ["Qwen/Qwen3-0.6B"]

    def test_modelscope_source_from_env(
        self, clean_env, no_gating, monkeypatch, fake_modelscope
    ):
        monkeypatch.setenv("ROBOML_SOURCE", "modelscope")
        result = resolve_checkpoint("Qwen/Qwen3-0.6B")
        assert result == "/fake/cache/Qwen/Qwen3-0.6B"

    def test_modelscope_not_installed(self, clean_env, monkeypatch):
        monkeypatch.setitem(sys.modules, "modelscope", None)
        with pytest.raises(ImportError, match=r"roboml\[modelscope\]"):
            resolve_checkpoint("Qwen/Qwen3-0.6B", "modelscope")

    def test_download_failure_raises(self, clean_env, no_gating, monkeypatch):
        fake = types.ModuleType("modelscope")

        def failing_download(checkpoint):
            raise Exception("404 not found")

        fake.snapshot_download = failing_download
        monkeypatch.setitem(sys.modules, "modelscope", fake)

        with pytest.raises(RuntimeError, match="Failed to download"):
            resolve_checkpoint("some/unknown-model", "modelscope")

    def test_download_failure_hints_alternative(
        self, clean_env, no_gating, monkeypatch
    ):
        fake = types.ModuleType("modelscope")

        def failing_download(checkpoint):
            raise Exception("404 not found")

        fake.snapshot_download = failing_download
        monkeypatch.setitem(sys.modules, "modelscope", fake)

        # defaults known to be missing from ModelScope should suggest alternatives
        with pytest.raises(RuntimeError, match="microsoft/speecht5_tts"):
            resolve_checkpoint("suno/bark-small", "modelscope")
        with pytest.raises(RuntimeError, match="facebook/detr-resnet-50"):
            resolve_checkpoint("PekingU/rtdetr_r50vd_coco_o365", "modelscope")

    def test_alternatives_cover_known_missing_defaults(self):
        assert "suno/bark-small" in MODELSCOPE_ALTERNATIVES
        assert "PekingU/rtdetr_r50vd_coco_o365" in MODELSCOPE_ALTERNATIVES


class TestCheckpointGating:
    def test_local_path_not_gated(self, clean_env, tmp_path):
        assert is_checkpoint_gated(str(tmp_path), "huggingface") is False

    def test_hf_gated(self, clean_env, monkeypatch):
        monkeypatch.setattr(
            "huggingface_hub.model_info",
            lambda repo: types.SimpleNamespace(gated="manual"),
        )
        assert is_checkpoint_gated("some/gated-model", "huggingface") is True

    def test_hf_ungated(self, clean_env, monkeypatch):
        monkeypatch.setattr(
            "huggingface_hub.model_info",
            lambda repo: types.SimpleNamespace(gated=False),
        )
        assert is_checkpoint_gated("some/open-model", "huggingface") is False

    def test_hf_api_error_fails_open(self, clean_env, monkeypatch):
        def failing_info(repo):
            raise Exception("offline")

        monkeypatch.setattr("huggingface_hub.model_info", failing_info)
        assert is_checkpoint_gated("some/unknown-model", "huggingface") is False

    def test_modelscope_restricted(self, clean_env, monkeypatch):
        monkeypatch.setattr(
            "requests.get",
            lambda url, timeout=None: types.SimpleNamespace(
                json=lambda: {"Data": {"Visibility": 1}}
            ),
        )
        assert is_checkpoint_gated("some/restricted-model", "modelscope") is True

    def test_modelscope_public(self, clean_env, monkeypatch):
        monkeypatch.setattr(
            "requests.get",
            lambda url, timeout=None: types.SimpleNamespace(
                json=lambda: {"Data": {"Visibility": 5}}
            ),
        )
        assert is_checkpoint_gated("some/public-model", "modelscope") is False

    def test_modelscope_api_error_fails_open(self, clean_env, monkeypatch):
        def failing_get(url, timeout=None):
            raise Exception("offline")

        monkeypatch.setattr("requests.get", failing_get)
        assert is_checkpoint_gated("some/unknown-model", "modelscope") is False


class TestGatedPreflight:
    def test_gated_hf_without_credentials_raises(self, clean_env, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
        monkeypatch.setattr("roboml.utils.is_checkpoint_gated", lambda *a, **k: True)
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            resolve_checkpoint("BAAI/RoboBrain2.0-3B", "huggingface")

    def test_gated_hf_with_credentials_passes(self, clean_env, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "some-token")
        monkeypatch.setattr("roboml.utils.is_checkpoint_gated", lambda *a, **k: True)
        result = resolve_checkpoint("BAAI/RoboBrain2.0-3B", "huggingface")
        assert result == "BAAI/RoboBrain2.0-3B"

    def test_ungated_hf_skips_credential_check(self, clean_env, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
        monkeypatch.setattr("roboml.utils.is_checkpoint_gated", lambda *a, **k: False)
        result = resolve_checkpoint("BAAI/RoboBrain2.5-4B", "huggingface")
        assert result == "BAAI/RoboBrain2.5-4B"

    def test_gated_modelscope_without_credentials_raises(
        self, clean_env, monkeypatch, fake_modelscope
    ):
        monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
        _fake_hub_config(monkeypatch, token=None)
        monkeypatch.setattr("roboml.utils.is_checkpoint_gated", lambda *a, **k: True)
        with pytest.raises(RuntimeError, match="MODELSCOPE_API_TOKEN"):
            resolve_checkpoint("BAAI/RoboBrain2.0-3B", "modelscope")

    def test_gated_modelscope_with_credentials_downloads(
        self, clean_env, monkeypatch, fake_modelscope
    ):
        monkeypatch.setenv("MODELSCOPE_API_TOKEN", "some-token")
        monkeypatch.setattr("roboml.utils.is_checkpoint_gated", lambda *a, **k: True)
        result = resolve_checkpoint("BAAI/RoboBrain2.0-3B", "modelscope")
        assert result == "/fake/cache/BAAI/RoboBrain2.0-3B"


class TestHuggingFaceCredentials:
    def test_env_token(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "some-token")
        assert has_huggingface_credentials() is True

    def test_cached_login_token(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr("huggingface_hub.get_token", lambda: "cached-token")
        assert has_huggingface_credentials() is True

    def test_no_credentials(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
        assert has_huggingface_credentials() is False

    def test_undeterminable_defaults_to_true(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        assert has_huggingface_credentials() is True


class TestModelScopeCredentials:
    def test_env_token(self, monkeypatch):
        monkeypatch.setenv("MODELSCOPE_API_TOKEN", "some-token")
        assert has_modelscope_credentials() is True

    def test_cached_login_token(self, monkeypatch):
        monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
        _fake_hub_config(monkeypatch, token="cached-token")
        assert has_modelscope_credentials() is True

    def test_no_credentials(self, monkeypatch):
        monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
        _fake_hub_config(monkeypatch, token=None)
        assert has_modelscope_credentials() is False

    def test_undeterminable_defaults_to_true(self, monkeypatch):
        monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
        # simulate an older modelscope without the modelscope_hub package
        monkeypatch.setitem(sys.modules, "modelscope_hub", None)
        monkeypatch.setitem(sys.modules, "modelscope_hub.config", None)
        assert has_modelscope_credentials() is True


class TestWhisperAliasMapping:
    def test_size_alias_maps_to_repo_id(self):
        from roboml.models.speech_to_text import _map_size_alias

        assert _map_size_alias("small.en") == "Systran/faster-whisper-small.en"

    def test_repo_id_passes_through(self):
        from roboml.models.speech_to_text import _map_size_alias

        assert (
            _map_size_alias("Systran/faster-whisper-large-v3")
            == "Systran/faster-whisper-large-v3"
        )


class TestModelInitSignatures:
    def test_all_models_accept_source(self):
        from roboml import models

        for model_cls in (
            models.TransformersLLM,
            models.TransformersMLLM,
            models.RoboBrain2,
            models.Whisper,
            models.TransformersTTS,
            models.VisionModel,
        ):
            params = inspect.signature(model_cls._initialize).parameters
            assert "source" in params, (
                f"{model_cls.__name__}._initialize is missing the source param"
            )
            assert params["source"].default is None
