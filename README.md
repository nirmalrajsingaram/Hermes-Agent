# Hermes Agent

A Docker-ready, multi-model AI agent with a configurable model backend, tool-use support, and an interactive REPL.

> **Build verified** — `hermes-agent:latest` builds successfully (~69.5 MB image). All 13 tests pass.

---

## Features

| Feature | Details |
|---|---|
| **Configurable models** | OpenAI, Anthropic, Ollama, any OpenAI-compatible endpoint |
| **Agentic loop** | Autonomous tool-use with configurable max iterations |
| **Built-in tools** | DateTime, Calculator, Web Fetch, Shell, JSON formatter |
| **Extensible tools** | `@tool` decorator — add your own in minutes |
| **Streaming** | Streaming output for all supported providers |
| **Interactive REPL** | `/model`, `/reset`, `/tools`, `/models` commands |
| **Docker-first** | Multi-stage Dockerfile + Compose with Ollama profile |
| **Config hierarchy** | YAML file → env vars (env wins) |

---

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/your-org/hermes-agent
cd hermes-agent
cp .env.example .env
# Edit .env and add your API key(s)
```

### 2. Build the Docker image

```bash
docker compose build
# or directly:
docker build -t hermes-agent:latest .
```

### 3. Run with Docker Compose

```bash
# Interactive REPL
docker compose run --rm hermes

# Single query
docker compose run --rm hermes --query "What is 42 * 73?"

# Use a specific model
docker compose run --rm hermes --model claude --query "Summarise quantum computing in 3 bullets"

# With streaming
docker compose run --rm hermes --model gpt-4o-mini --stream --query "Tell me a short story"

# List all available models
docker compose run --rm hermes --list-models

# Disable tools (plain chat)
docker compose run --rm hermes --no-tools --query "Hello!"
```

### 4. Run locally (without Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python -m src.main
```

---

## Configurable Models

All models are defined in `config/config.yaml`. Switch at runtime via:

| Method | Example |
|---|---|
| CLI flag | `--model claude` |
| Env var | `HERMES_MODEL=ollama-llama3` |
| REPL command | `/model gpt-4o-mini` |
| Python API | `agent.switch_model("claude")` |

### Preset Models

| Name | Provider | Model ID |
|---|---|---|
| `default` | openai | gpt-4o |
| `gpt-4o-mini` | openai | gpt-4o-mini |
| `claude` | anthropic | claude-3-5-sonnet-20241022 |
| `claude-haiku` | anthropic | claude-3-haiku-20240307 |
| `ollama-llama3` | ollama | llama3.2 (local) |
| `compatible` | openai-compatible | Mixtral (Together AI) |

Add more by appending to the `models` list in `config/config.yaml`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HERMES_MODEL` | `default` | Active model name |
| `HERMES_CONFIG` | `config/config.yaml` | Path to config YAML |
| `HERMES_MAX_ITERATIONS` | `10` | Max agentic loop steps |
| `HERMES_STREAM` | `false` | Enable streaming |
| `HERMES_TOOLS_ENABLED` | `true` | Enable/disable tools |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `TOGETHER_API_KEY` | — | Together AI key |

---

## Adding a Custom Tool

```python
# src/tools/my_tools.py
from src.tools.registry import tool

@tool(
    description="Convert Celsius to Fahrenheit.",
    parameters={
        "type": "object",
        "properties": {
            "celsius": {"type": "number", "description": "Temperature in Celsius"},
        },
        "required": ["celsius"],
    },
)
def celsius_to_fahrenheit(celsius: float) -> str:
    return f"{celsius * 9/5 + 32:.1f}°F"
```

Then register it in `src/main.py`:

```python
from src.tools.my_tools import celsius_to_fahrenheit
registry.register(celsius_to_fahrenheit)
```

---

## Local LLM with Ollama

```bash
# Start Ollama alongside Hermes
docker compose --profile ollama up -d ollama

# Pull a model
docker compose exec ollama ollama pull llama3.2

# Run Hermes with Ollama
docker compose run --rm -e HERMES_MODEL=ollama-llama3 hermes
```

---

## Project Structure

```
hermes-agent/
├── config/
│   └── config.yaml          # Model & agent configuration
├── prompts/
│   └── system.md            # System prompt (editable without rebuild)
├── src/
│   ├── agent/
│   │   ├── hermes.py        # HermesAgent — agentic loop
│   │   ├── llm_client.py    # LLM client abstraction (all providers)
│   │   └── logging_setup.py # Logging configuration
│   ├── config/
│   │   └── settings.py      # Config dataclasses + loader
│   ├── tools/
│   │   ├── registry.py      # @tool decorator + ToolRegistry
│   │   └── builtin_tools.py # Built-in tools
│   └── main.py              # CLI entrypoint
├── tests/
│   ├── test_config.py
│   └── test_tools.py
├── logs/                    # Runtime logs (git-ignored)
├── .env.example             # Environment variable template
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Running Tests

```bash
pip install pytest httpx pyyaml
pytest tests/ -v
```

Expected output (all 13 tests pass):
```
tests/test_config.py::test_model_config_from_dict        PASSED
tests/test_config.py::test_load_config_defaults          PASSED
tests/test_config.py::test_env_override_model            PASSED
tests/test_config.py::test_active_model_unknown_raises   PASSED
tests/test_config.py::test_env_override_iterations       PASSED
tests/test_tools.py::test_register_and_call              PASSED
tests/test_tools.py::test_registry_openai_schema         PASSED
tests/test_tools.py::test_unknown_tool_raises            PASSED
tests/test_tools.py::test_calculator_basic               PASSED
tests/test_tools.py::test_calculator_safe                PASSED
tests/test_tools.py::test_format_json_valid              PASSED
tests/test_tools.py::test_format_json_invalid            PASSED
tests/test_tools.py::test_get_current_datetime           PASSED
13 passed in 0.08s
```

---

## REPL Commands

Once inside the interactive REPL (`docker compose run --rm hermes`):

| Command | Description |
|---|---|
| `/models` | List all configured models |
| `/model <name>` | Switch to a different model live |
| `/tools` | Show registered tools |
| `/reset` | Clear conversation history |
| `/help` | Show all commands |
| `/quit` | Exit |

---

## License

MIT
