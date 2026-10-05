"""
Built-in tools shipped with Hermes Agent.
Add your own tools here or in a separate module, then register them in main.py.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import subprocess
from typing import Any, Dict, Optional

import httpx

from src.tools.registry import tool


# ---------------------------------------------------------------------------
# Date / Time
# ---------------------------------------------------------------------------

@tool(
    description="Get the current date and time in ISO-8601 format.",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Optional timezone name (e.g. 'UTC', 'US/Eastern'). Defaults to local.",
            }
        },
        "required": [],
    },
)
def get_current_datetime(timezone: Optional[str] = None) -> str:
    """Return current datetime, optionally in a specific timezone."""
    try:
        import zoneinfo

        tz = zoneinfo.ZoneInfo(timezone) if timezone else None
        now = datetime.datetime.now(tz=tz)
    except Exception:
        now = datetime.datetime.utcnow()
    return now.isoformat()


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

@tool(
    description="Evaluate a safe mathematical expression. Supports +, -, *, /, **, sqrt, log, sin, cos, tan, pi, e.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate, e.g. '2 ** 10' or 'sqrt(144)'.",
            }
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    allowed_names: Dict[str, Any] = {
        k: getattr(math, k)
        for k in dir(math)
        if not k.startswith("_")
    }
    allowed_names["abs"] = abs
    allowed_names["round"] = round

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


# ---------------------------------------------------------------------------
# Web fetch
# ---------------------------------------------------------------------------

@tool(
    description="Fetch the text content of a URL. Returns up to 4000 characters.",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch."},
            "timeout": {"type": "integer", "description": "Request timeout in seconds (default 15)."},
        },
        "required": ["url"],
    },
)
def web_fetch(url: str, timeout: int = 15) -> str:
    """Fetch and return plain text from a URL."""
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text[:4000]
        return text
    except Exception as exc:
        return f"Error fetching {url}: {exc}"


# ---------------------------------------------------------------------------
# Shell command (sandboxed)
# ---------------------------------------------------------------------------

@tool(
    description="Run a safe, non-interactive shell command and return stdout. Only whitelisted commands are allowed.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run."},
        },
        "required": ["command"],
    },
)
def shell_command(command: str) -> str:
    """Execute a whitelisted shell command."""
    _ALLOWED_PREFIXES = (
        "ls", "pwd", "echo", "cat", "grep", "find", "wc",
        "python", "pip", "curl", "wget", "df", "du",
    )
    first_word = command.strip().split()[0] if command.strip() else ""
    if first_word not in _ALLOWED_PREFIXES:
        return f"Command '{first_word}' is not whitelisted. Allowed: {_ALLOWED_PREFIXES}"

    try:
        result = subprocess.run(
            command,
            shell=True,  # noqa: S602
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout or result.stderr
        return output[:3000]
    except subprocess.TimeoutExpired:
        return "Command timed out after 15 seconds."
    except Exception as exc:
        return f"Error running command: {exc}"


# ---------------------------------------------------------------------------
# JSON pretty-print helper
# ---------------------------------------------------------------------------

@tool(
    description="Parse and pretty-print a JSON string.",
    parameters={
        "type": "object",
        "properties": {
            "json_string": {"type": "string", "description": "Raw JSON string to format."},
        },
        "required": ["json_string"],
    },
)
def format_json(json_string: str) -> str:
    """Pretty-print JSON."""
    try:
        data = json.loads(json_string)
        return json.dumps(data, indent=2)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"


# ---------------------------------------------------------------------------
# All built-in tools (used in main.py for registration)
# ---------------------------------------------------------------------------

BUILTIN_TOOLS = [
    get_current_datetime,
    calculator,
    web_fetch,
    shell_command,
    format_json,
]
