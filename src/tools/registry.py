"""
Tool registry for Hermes Agent.
Tools are plain Python functions decorated with @tool.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool descriptor
# ---------------------------------------------------------------------------

class ToolDefinition:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        fn: Callable,
        return_direct: bool = False,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON Schema object
        self.fn = fn
        self.return_direct = return_direct  # if True, agent returns tool output immediately

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------

def tool(
    name: Optional[str] = None,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
    return_direct: bool = False,
):
    """
    Decorator to register a function as a Hermes tool.

    Usage:
        @tool(description="Add two numbers", parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        })
        def add(a: float, b: float) -> float:
            return a + b
    """

    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        tool_desc = description or (inspect.getdoc(fn) or "")
        tool_params = parameters or _infer_parameters(fn)

        fn._tool_definition = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            parameters=tool_params,
            fn=fn,
            return_direct=return_direct,
        )
        return fn

    return decorator


def _infer_parameters(fn: Callable) -> Dict[str, Any]:
    """Build a minimal JSON Schema from Python type annotations."""
    sig = inspect.signature(fn)
    hints = {}
    try:
        hints = fn.__annotations__
    except AttributeError:
        pass

    _py_to_json = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
    }

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        hint = hints.get(param_name)
        json_type = "string"
        if hint is not None:
            json_type = _py_to_json.get(getattr(hint, "__name__", ""), "string")

        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, fn: Callable) -> None:
        """Register a function that has been decorated with @tool."""
        if not hasattr(fn, "_tool_definition"):
            raise ValueError(
                f"Function '{fn.__name__}' must be decorated with @tool before registering."
            )
        td: ToolDefinition = fn._tool_definition
        self._tools[td.name] = td
        logger.debug("Tool registered: %s", td.name)

    def register_all(self, *fns: Callable) -> None:
        for fn in fns:
            self.register(fn)

    def has_tools(self) -> bool:
        return bool(self._tools)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def openai_schema(self) -> List[Dict[str, Any]]:
        return [td.to_openai_schema() for td in self._tools.values()]

    def call(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered. Available: {self.names()}")

        td = self._tools[name]

        if timeout:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(td.fn, **arguments)
                return future.result(timeout=timeout)
        else:
            return td.fn(**arguments)
