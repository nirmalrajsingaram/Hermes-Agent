"""
Hermes Agent — agentic loop with tool use, configurable models, and memory.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from src.agent.llm_client import (
    LLMResponse,
    assistant_message,
    create_llm_client,
    system_message,
    user_message,
    tool_result_message,
)
from src.config.settings import AgentConfig
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class HermesAgent:
    """
    Agentic loop:
      1. Receive user input
      2. Call LLM (with tools)
      3. If tool_calls → execute tools → feed results back → repeat
      4. Return final assistant message
    """

    def __init__(self, config: AgentConfig, tool_registry: Optional[ToolRegistry] = None):
        self.config = config
        self.tool_registry = tool_registry or ToolRegistry()
        self._llm = create_llm_client(config.active_model)
        self._history: List[Dict[str, Any]] = []
        self._system_prompt: str = self._load_system_prompt()

        logger.info(
            "HermesAgent initialised — model=%s provider=%s",
            config.active_model.model_id,
            config.active_model.provider,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> str:
        """Single-turn chat; manages history and agentic loop internally."""
        self._history.append(user_message(user_input))
        response = self._run_loop()
        self._history.append(assistant_message(response.content))
        self._trim_history()
        return response.content

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """Streaming single-turn chat (no tool calls in stream mode)."""
        self._history.append(user_message(user_input))
        messages = self._build_messages()
        full_content = ""
        for chunk in self._llm.stream_complete(messages):
            full_content += chunk
            yield chunk
        self._history.append(assistant_message(full_content))
        self._trim_history()

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []
        logger.debug("Conversation history cleared.")

    def switch_model(self, model_name: str) -> None:
        """Hot-swap to a different registered model."""
        if model_name not in self.config.models:
            raise KeyError(
                f"Model '{model_name}' not found. Available: {list(self.config.models.keys())}"
            )
        self.config.default_model = model_name
        self._llm = create_llm_client(self.config.active_model)
        logger.info("Switched to model: %s", model_name)

    def list_models(self) -> List[str]:
        """Return names of all registered models."""
        return list(self.config.models.keys())

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> LLMResponse:
        tools = (
            self.tool_registry.openai_schema()
            if self.config.tools_enabled and self.tool_registry.has_tools()
            else None
        )

        for iteration in range(self.config.max_iterations):
            messages = self._build_messages()
            logger.debug("Loop iteration %d/%d", iteration + 1, self.config.max_iterations)

            response = self._llm.complete(
                messages=messages,
                tools=tools,
                stream=False,
            )

            # No tool calls → final answer
            if not response.has_tool_calls:
                logger.debug("No tool calls — returning final response.")
                return response

            # Execute tool calls
            logger.debug("Tool calls requested: %s", [tc["name"] for tc in response.tool_calls])
            self._history.append(
                {"role": "assistant", "content": response.content or "", "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                    }
                    for tc in response.tool_calls
                ]}
            )

            for tool_call in response.tool_calls:
                result = self._execute_tool(tool_call)
                # Append tool result in OpenAI format
                self._history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    }
                )

        logger.warning("Max iterations (%d) reached.", self.config.max_iterations)
        return LLMResponse(
            content="I reached my maximum number of reasoning steps. Please rephrase or simplify your request.",
            finish_reason="max_iterations",
        )

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_call: Dict[str, Any]) -> str:
        name = tool_call["name"]
        args = tool_call["arguments"]
        logger.info("Executing tool: %s args=%s", name, args)
        start = time.monotonic()

        try:
            result = self.tool_registry.call(name, args, timeout=self.config.tool_timeout)
            elapsed = time.monotonic() - start
            logger.debug("Tool '%s' completed in %.2fs", name, elapsed)
            return json.dumps(result) if not isinstance(result, str) else result
        except Exception as exc:
            logger.error("Tool '%s' failed: %s", name, exc, exc_info=True)
            return json.dumps({"error": str(exc)})

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(self) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self._system_prompt:
            messages.append(system_message(self._system_prompt))
        messages.extend(self._history)
        return messages

    def _trim_history(self) -> None:
        max_turns = self.config.max_history_turns * 2  # each turn = user + assistant
        if len(self._history) > max_turns:
            self._history = self._history[-max_turns:]

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _load_system_prompt(self) -> str:
        prompt_path = Path(self.config.system_prompt_file)
        if not prompt_path.is_absolute():
            # Resolve relative to project root (two levels up from this file)
            prompt_path = Path(__file__).parents[2] / self.config.system_prompt_file

        if prompt_path.exists():
            content = prompt_path.read_text(encoding="utf-8").strip()
            logger.debug("Loaded system prompt from %s", prompt_path)
            return content

        logger.warning("System prompt file not found: %s — using built-in default.", prompt_path)
        return (
            "You are Hermes, a helpful and capable AI assistant. "
            "You have access to tools that help you answer questions accurately. "
            "Always think step-by-step and be concise."
        )
