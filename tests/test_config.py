"""Tests for configuration loading."""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config.settings import AgentConfig, ModelConfig, load_config


def test_model_config_from_dict():
    data = {
        "name": "test",
        "provider": "openai",
        "model_id": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 2048,
        "temperature": 0.5,
    }
    mc = ModelConfig.from_dict(data)
    assert mc.name == "test"
    assert mc.provider == "openai"
    assert mc.model_id == "gpt-4o"
    assert mc.max_tokens == 2048
    assert mc.temperature == 0.5


def test_load_config_defaults(tmp_path):
    """load_config with a minimal YAML returns defaults."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "models:\n"
        "  - name: default\n"
        "    provider: openai\n"
        "    model_id: gpt-4o\n"
        "    api_key_env: OPENAI_API_KEY\n"
    )
    cfg = load_config(str(cfg_file))
    assert cfg.default_model == "default"
    assert "default" in cfg.models
    assert cfg.models["default"].model_id == "gpt-4o"


def test_env_override_model(tmp_path, monkeypatch):
    """HERMES_MODEL env var overrides default_model."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "models:\n"
        "  - name: default\n"
        "    provider: openai\n"
        "    model_id: gpt-4o\n"
        "    api_key_env: OPENAI_API_KEY\n"
        "  - name: fast\n"
        "    provider: openai\n"
        "    model_id: gpt-4o-mini\n"
        "    api_key_env: OPENAI_API_KEY\n"
    )
    monkeypatch.setenv("HERMES_MODEL", "fast")
    cfg = load_config(str(cfg_file))
    assert cfg.active_model.name == "fast"


def test_active_model_unknown_raises(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "models:\n"
        "  - name: default\n"
        "    provider: openai\n"
        "    model_id: gpt-4o\n"
        "    api_key_env: OPENAI_API_KEY\n"
    )
    monkeypatch.setenv("HERMES_MODEL", "nonexistent")
    cfg = load_config(str(cfg_file))
    with pytest.raises(KeyError, match="nonexistent"):
        _ = cfg.active_model


def test_env_override_iterations(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "models:\n"
        "  - name: default\n"
        "    provider: openai\n"
        "    model_id: gpt-4o\n"
        "    api_key_env: OPENAI_API_KEY\n"
    )
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "42")
    cfg = load_config(str(cfg_file))
    assert cfg.max_iterations == 42
