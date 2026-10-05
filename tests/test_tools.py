"""Tests for the tool registry and built-in tools."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.tools.registry import ToolRegistry, tool
from src.tools.builtin_tools import calculator, format_json, get_current_datetime


def test_register_and_call():
    registry = ToolRegistry()

    @tool(description="Multiply two numbers", parameters={
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
        },
        "required": ["x", "y"],
    })
    def multiply(x: float, y: float) -> float:
        return x * y

    registry.register(multiply)
    assert "multiply" in registry.names()
    result = registry.call("multiply", {"x": 3, "y": 4})
    assert result == 12


def test_registry_openai_schema():
    registry = ToolRegistry()

    @tool(description="Say hello", parameters={
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    })
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    registry.register(greet)
    schema = registry.openai_schema()
    assert len(schema) == 1
    assert schema[0]["type"] == "function"
    assert schema[0]["function"]["name"] == "greet"


def test_unknown_tool_raises():
    registry = ToolRegistry()
    with pytest.raises(KeyError, match="no_such_tool"):
        registry.call("no_such_tool", {})


def test_calculator_basic():
    assert calculator(expression="2 + 2") == "4"
    assert calculator(expression="sqrt(16)") == "4.0"
    assert calculator(expression="2 ** 10") == "1024"


def test_calculator_safe():
    result = calculator(expression="__import__('os').system('echo hi')")
    assert "Error" in result


def test_format_json_valid():
    result = format_json(json_string='{"a":1,"b":2}')
    assert '"a": 1' in result


def test_format_json_invalid():
    result = format_json(json_string="not json")
    assert "Invalid JSON" in result


def test_get_current_datetime():
    result = get_current_datetime()
    # Should be a valid ISO-8601 string
    assert "T" in result or "-" in result
