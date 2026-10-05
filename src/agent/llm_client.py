"""
LLM client abstraction for Hermes Agent.
Supports OpenAI, Anthropic, Ollama, and any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Generator, List, Optional

from src.config.settings import ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def user_message(content: str) -> Dict[str, str]:
    return {"role": "user", "content": content}


def assistant_message(content: str) -> Dict[str, str]:
    return {"role": "assistant", "content": content}


def system_message(content: str) -> Dict[str, str]:
    return {"role": "system", "content": content}


def tool_result_message(tool_use_id: str, content: str) -> Dict[str, Any]:
    """Anthropic-style tool result."""
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}],
    }


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------

class LLMResponse:
    """Unified response object across providers."""

    def __init__(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        raw: Any = None,
        finish_reason: str = "stop",
    ):
        self.content = content
        self.tool_calls: List[Dict[str, Any]] = tool_calls or []
        self.raw = raw
        self.finish_reason = finish_reason

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def __repr__(self) -> str:
        return (
            f"LLMResponse(finish_reason={self.finish_reason!r}, "
            f"tool_calls={len(self.tool_calls)}, "
            f"content={self.content[:60]!r}...)"
        )


# ---------------------------------------------------------------------------
# Base client
# ---------------------------------------------------------------------------

class BaseLLMClient:
    def __init__(self, model_config: ModelConfig):
        self.cfg = model_config

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> LLMResponse:
        raise NotImplementedError

    def stream_complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[str, None, None]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible client
# ---------------------------------------------------------------------------

class OpenAIClient(BaseLLMClient):
    """Works for openai and openai-compatible providers."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc

        kwargs: Dict[str, Any] = {
            "api_key": model_config.api_key or "sk-no-key",
            "timeout": model_config.timeout,
        }
        if model_config.api_base:
            kwargs["base_url"] = model_config.api_base

        self._client = OpenAI(**kwargs)

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": self.cfg.model_id,
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        logger.debug("OpenAI request model=%s msgs=%d", self.cfg.model_id, len(messages))
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(
                    {"id": tc.id, "name": tc.function.name, "arguments": args}
                )

        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            raw=resp,
            finish_reason=choice.finish_reason or "stop",
        )

    def stream_complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[str, None, None]:
        kwargs: Dict[str, Any] = {
            "model": self.cfg.model_id,
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        for chunk in self._client.chat.completions.create(**kwargs):
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

class AnthropicClient(BaseLLMClient):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise ImportError("Install anthropic: pip install anthropic") from exc

        self._client = anthropic.Anthropic(
            api_key=model_config.api_key or "",
            timeout=model_config.timeout,
        )

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> LLMResponse:
        # Separate system message from the rest
        system_content = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                chat_messages.append(m)

        # Convert OpenAI-style tools to Anthropic format
        anthropic_tools = []
        if tools:
            for t in tools:
                fn = t.get("function", t)
                anthropic_tools.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )

        kwargs: Dict[str, Any] = {
            "model": self.cfg.model_id,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "messages": chat_messages,
        }
        if system_content:
            kwargs["system"] = system_content
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        logger.debug("Anthropic request model=%s msgs=%d", self.cfg.model_id, len(chat_messages))
        resp = self._client.messages.create(**kwargs)

        text_content = ""
        tool_calls = []

        for block in resp.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {"id": block.id, "name": block.name, "arguments": block.input}
                )

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            raw=resp,
            finish_reason=resp.stop_reason or "stop",
        )

    def stream_complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[str, None, None]:
        system_content = ""
        chat_messages = [m for m in messages if m["role"] != "system"]
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]

        kwargs: Dict[str, Any] = {
            "model": self.cfg.model_id,
            "max_tokens": self.cfg.max_tokens,
            "messages": chat_messages,
        }
        if system_content:
            kwargs["system"] = system_content

        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text


# ---------------------------------------------------------------------------
# Ollama client (uses OpenAI-compatible API)
# ---------------------------------------------------------------------------

class OllamaClient(OpenAIClient):
    """Ollama exposes an OpenAI-compatible /v1 endpoint."""

    def __init__(self, model_config: ModelConfig):
        if not model_config.api_base:
            model_config.api_base = "http://localhost:11434/v1"
        elif not model_config.api_base.endswith("/v1"):
            model_config.api_base = model_config.api_base.rstrip("/") + "/v1"
        super().__init__(model_config)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_MAP = {
    "openai": OpenAIClient,
    "openai-compatible": OpenAIClient,
    "anthropic": AnthropicClient,
    "ollama": OllamaClient,
}


def create_llm_client(model_config: ModelConfig) -> BaseLLMClient:
    """Instantiate the right client for the given provider."""
    cls = _PROVIDER_MAP.get(model_config.provider.lower())
    if cls is None:
        raise ValueError(
            f"Unknown provider '{model_config.provider}'. "
            f"Supported: {list(_PROVIDER_MAP.keys())}"
        )
    return cls(model_config)
