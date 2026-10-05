"""
Configuration loader for Hermes Agent.
Reads from config/config.yaml and overrides with environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for a single LLM backend."""

    name: str                          # logical name, e.g. "default", "fast"
    provider: str                      # openai | anthropic | ollama | openai-compatible
    model_id: str                      # e.g. "gpt-4o", "claude-3-5-sonnet-20241022"
    api_base: Optional[str] = None     # override endpoint (Ollama, LM Studio, etc.)
    api_key_env: str = "OPENAI_API_KEY"  # env var that holds the key
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    timeout: int = 120                 # seconds

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(
            name=data["name"],
            provider=data["provider"],
            model_id=data["model_id"],
            api_base=data.get("api_base"),
            api_key_env=data.get("api_key_env", "OPENAI_API_KEY"),
            max_tokens=int(data.get("max_tokens", 4096)),
            temperature=float(data.get("temperature", 0.7)),
            top_p=float(data.get("top_p", 1.0)),
            timeout=int(data.get("timeout", 120)),
        )


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Top-level agent configuration."""

    # Which model config (by name) to use by default
    default_model: str = "default"

    # All registered model configs keyed by name
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    # System prompt file path (relative to project root)
    system_prompt_file: str = "prompts/system.md"

    # Maximum agentic loop iterations before stopping
    max_iterations: int = 10

    # Whether to enable streaming responses
    stream: bool = False

    # Tool settings
    tools_enabled: bool = True
    tool_timeout: int = 30  # seconds per tool call

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/hermes.log"

    # Memory / history
    max_history_turns: int = 20

    @property
    def active_model(self) -> ModelConfig:
        """Return the active ModelConfig, respecting HERMES_MODEL env override."""
        override = os.environ.get("HERMES_MODEL", self.default_model)
        if override not in self.models:
            raise KeyError(
                f"Model '{override}' not found. Available: {list(self.models.keys())}"
            )
        return self.models[override]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "config" / "config.yaml"


def load_config(path: Optional[str] = None) -> AgentConfig:
    """Load configuration from YAML file, then apply env-var overrides."""

    config_path = Path(path) if path else Path(
        os.environ.get("HERMES_CONFIG", str(_DEFAULT_CONFIG_PATH))
    )

    raw: Dict[str, Any] = {}
    if config_path.exists():
        with config_path.open() as fh:
            raw = yaml.safe_load(fh) or {}

    agent_raw = raw.get("agent", {})
    models_raw: List[Dict[str, Any]] = raw.get("models", [])

    # Build model registry
    models: Dict[str, ModelConfig] = {}
    for m in models_raw:
        mc = ModelConfig.from_dict(m)
        models[mc.name] = mc

    # If no models defined, create a sensible default
    if not models:
        models["default"] = ModelConfig(
            name="default",
            provider="openai",
            model_id=os.environ.get("HERMES_MODEL_ID", "gpt-4o"),
            api_key_env="OPENAI_API_KEY",
        )

    cfg = AgentConfig(
        default_model=agent_raw.get("default_model", "default"),
        models=models,
        system_prompt_file=agent_raw.get("system_prompt_file", "prompts/system.md"),
        max_iterations=int(agent_raw.get("max_iterations", 10)),
        stream=bool(agent_raw.get("stream", False)),
        tools_enabled=bool(agent_raw.get("tools_enabled", True)),
        tool_timeout=int(agent_raw.get("tool_timeout", 30)),
        log_level=agent_raw.get("log_level", os.environ.get("LOG_LEVEL", "INFO")),
        log_file=agent_raw.get("log_file", "logs/hermes.log"),
        max_history_turns=int(agent_raw.get("max_history_turns", 20)),
    )

    # Env-var overrides for common knobs
    if os.environ.get("HERMES_MAX_ITERATIONS"):
        cfg.max_iterations = int(os.environ["HERMES_MAX_ITERATIONS"])
    if os.environ.get("HERMES_STREAM"):
        cfg.stream = os.environ["HERMES_STREAM"].lower() in ("1", "true", "yes")
    if os.environ.get("HERMES_TOOLS_ENABLED"):
        cfg.tools_enabled = os.environ["HERMES_TOOLS_ENABLED"].lower() in ("1", "true", "yes")

    return cfg
