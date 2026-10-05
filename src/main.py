"""
Hermes Agent — main entry point.
Supports interactive REPL, single-shot query, and server mode.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is on sys.path when run directly
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agent.hermes import HermesAgent
from src.agent.logging_setup import setup_logging
from src.config.settings import load_config
from src.tools.builtin_tools import BUILTIN_TOOLS
from src.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes",
        description="Hermes Agent — configurable multi-model AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              hermes                                    # interactive REPL
              hermes -q "What is 42 * 73?"              # single query
              hermes --model claude -q "Summarise..."   # use specific model
              hermes --list-models                       # show available models
              hermes --no-tools -q "Plain chat"          # disable tools
            """
        ),
    )

    parser.add_argument(
        "-c", "--config",
        help="Path to config YAML (default: config/config.yaml or HERMES_CONFIG env var)",
        default=None,
    )
    parser.add_argument(
        "-m", "--model",
        help="Model name to use (overrides config default_model and HERMES_MODEL env var)",
        default=None,
    )
    parser.add_argument(
        "-q", "--query",
        help="Single query — run once and exit",
        default=None,
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable tool use",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all configured models and exit",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override log level",
    )

    return parser


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent(args: argparse.Namespace) -> HermesAgent:
    cfg = load_config(args.config)

    if args.model:
        cfg.default_model = args.model
    if args.no_tools:
        cfg.tools_enabled = False
    if args.stream:
        cfg.stream = True
    if args.log_level:
        cfg.log_level = args.log_level

    setup_logging(level=cfg.log_level, log_file=cfg.log_file)

    registry = ToolRegistry()
    if cfg.tools_enabled:
        registry.register_all(*BUILTIN_TOOLS)

    return HermesAgent(config=cfg, tool_registry=registry)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

BANNER = """
╔═══════════════════════════════════════════════════════╗
║         H E R M E S   A G E N T                      ║
║  Configurable multi-model AI assistant                ║
╚═══════════════════════════════════════════════════════╝
Type your message and press Enter.
Commands:  /quit  /reset  /model <name>  /models  /help
"""

HELP_TEXT = """
Commands:
  /quit             Exit the session
  /reset            Clear conversation history
  /model <name>     Switch to a different model
  /models           List available models
  /tools            Show registered tools
  /help             Show this help message
"""


def run_repl(agent: HermesAgent) -> None:
    print(BANNER)
    print(f"  Active model : {agent.config.active_model.model_id}")
    print(f"  Provider     : {agent.config.active_model.provider}")
    print(f"  Tools        : {'enabled' if agent.config.tools_enabled else 'disabled'}")
    if agent.config.tools_enabled:
        print(f"  Tool list    : {', '.join(agent.tool_registry.names())}")
    print()

    while True:
        try:
            raw = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        # ---------- built-in commands ----------
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break

            elif cmd == "/reset":
                agent.reset()
                print("[History cleared]")

            elif cmd == "/models":
                models = agent.list_models()
                active = agent.config.default_model
                for m in models:
                    marker = " ← active" if m == active else ""
                    mc = agent.config.models[m]
                    print(f"  {m:20s}  {mc.provider:20s}  {mc.model_id}{marker}")

            elif cmd == "/model":
                if len(parts) < 2:
                    print("Usage: /model <name>")
                else:
                    try:
                        agent.switch_model(parts[1].strip())
                        mc = agent.config.active_model
                        print(f"[Switched to {mc.name} → {mc.model_id} ({mc.provider})]")
                    except KeyError as e:
                        print(f"Error: {e}")

            elif cmd == "/tools":
                if agent.tool_registry.has_tools():
                    for name in agent.tool_registry.names():
                        print(f"  • {name}")
                else:
                    print("  (no tools registered)")

            elif cmd == "/help":
                print(HELP_TEXT)

            else:
                print(f"Unknown command '{cmd}'. Type /help for options.")

            continue

        # ---------- agent call ----------
        try:
            if agent.config.stream:
                print("Hermes> ", end="", flush=True)
                for chunk in agent.stream_chat(raw):
                    print(chunk, end="", flush=True)
                print()
            else:
                response = agent.chat(raw)
                print(f"Hermes> {response}")
        except Exception as exc:
            print(f"[Error] {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # List models without needing keys
    if args.list_models:
        cfg = load_config(args.config)
        print("Configured models:")
        for name, mc in cfg.models.items():
            default_marker = " (default)" if name == cfg.default_model else ""
            print(f"  {name:20s}  {mc.provider:20s}  {mc.model_id}{default_marker}")
        return 0

    agent = build_agent(args)

    if args.query:
        # Single-shot mode
        if agent.config.stream:
            for chunk in agent.stream_chat(args.query):
                print(chunk, end="", flush=True)
            print()
        else:
            print(agent.chat(args.query))
        return 0

    # Interactive REPL
    run_repl(agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
