# OpenSearch GenAI SDK

OTEL-native tracing and scoring for LLM applications. Instrument your AI workflows with standard OpenTelemetry spans and submit evaluation scores — all routed to OpenSearch through a single OTLP pipeline.

## Features

- **One-line setup** — `register()` configures the full OTEL pipeline (TracerProvider, exporter, auto-instrumentation)
- **Decorators** — `@workflow`, `@task`, `@agent`, `@tool` wrap functions as OTEL spans with [GenAI semantic convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/) attributes
- **Auto-instrumentation** — automatically discovers and activates installed instrumentor packages (OpenAI, Anthropic, Bedrock, LangChain, etc.)
- **Scoring** — `score()` emits evaluation metrics as OTEL spans at span, trace, or session level
- **AWS SigV4** — built-in SigV4 signing for AWS-hosted OpenSearch and Data Prepper endpoints
- **Zero lock-in** — remove a decorator and your code still works; everything is standard OTEL

## Requirements

- **Python**: 3.10, 3.11, 3.12, or 3.13
- **OpenTelemetry SDK**: ≥1.20.0, <2

## Installation

```bash
pip install opensearch-genai-sdk-py
```

With AWS SigV4 support:

```bash
pip install opensearch-genai-sdk-py[aws]
```

## Quick Start

```python
from opensearch_genai_sdk import register, workflow, agent, tool, score

# 1. Initialize tracing (one line)
register(endpoint="http://localhost:4318/v1/traces")

# 2. Decorate your functions
@tool("get_weather")
def get_weather(city: str) -> dict:
    """Fetch weather data for a city."""
    return {"city": city, "temp": 22, "condition": "sunny"}

@agent("weather_assistant")
def assistant(query: str) -> str:
    data = get_weather("Paris")
    return f"{data['condition']}, {data['temp']}C"

@workflow("weather_query")
def run(query: str) -> str:
    return assistant(query)

result = run("What's the weather?")

# 3. Submit scores (after workflow completes)
score(name="relevance", value=0.95, trace_id="...", source="llm-judge")
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Your Application                    │
│                                                      │
│  @workflow ─→ @agent ─→ @tool    score()            │
│     │            │         │        │                │
│     └────────────┴─────────┴────────┘                │
│                     │                                │
│            opensearch-genai-sdk-py                    │
├─────────────────────────────────────────────────────┤
│  register()                                          │
│  ┌─────────────────────────────────────────────┐    │
│  │  TracerProvider                              │    │
│  │  ├── Resource (service.name)                 │    │
│  │  ├── BatchSpanProcessor                      │    │
│  │  │   └── OTLPSpanExporter (HTTP or gRPC)     │    │
│  │  │       └── SigV4 signing (AWS endpoints)   │    │
│  │  └── Auto-instrumentation                    │    │
│  │      ├── openai, anthropic, bedrock, ...     │    │
│  │      ├── langchain, llamaindex, haystack     │    │
│  │      └── chromadb, pinecone, qdrant, ...     │    │
│  └─────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────┘
                       │ OTLP (HTTP/gRPC)
                       ▼
              ┌─────────────────┐
              │  Data Prepper /  │
              │  OTEL Collector  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   OpenSearch     │
              │  ├── traces      │
              │  └── scores      │
              └─────────────────┘
```

## API Reference

### `register()`

Configures the OTEL tracing pipeline. Call once at startup.

```python
register(
    endpoint="https://pipeline.us-east-1.osis.amazonaws.com/v1/traces",
    service_name="my-app",
    auth="auto",           # "auto" | "sigv4" | "none"
    batch=True,            # BatchSpanProcessor (True) or Simple (False)
    auto_instrument=True,  # discover installed instrumentor packages
)
```

**Endpoint formats:**

| URL scheme | Transport |
|---|---|
| `http://` / `https://` | HTTP (default) |
| `grpc://` | gRPC (insecure) |
| `grpcs://` | gRPC (TLS) |

**Auth:** Use `auth="sigv4"` for AWS endpoints requiring SigV4 signing. `auth="auto"` (default) uses plain authentication.

### Decorators

Four decorators for tracing application logic. Each creates an OTEL span with `gen_ai.*` semantic convention attributes.

| Decorator | Use for | Operation name | Span name format |
|---|---|---|---|
| `@workflow("name")` | Top-level orchestration | `invoke_agent` | `name` |
| `@task("name")` | Units of work | `invoke_agent` | `name` |
| `@agent("name")` | Autonomous agent logic | `invoke_agent` | `invoke_agent name` |
| `@tool("name")` | Tool/function calls | `execute_tool` | `execute_tool name` |

All decorators accept `name` (defaults to function's `__qualname__`) and `version`.

**Attributes set automatically:**

| Attribute | Set by |
|---|---|
| `gen_ai.operation.name` | All decorators |
| `gen_ai.agent.name` / `gen_ai.tool.name` | All decorators |
| `gen_ai.input.messages` / `gen_ai.output.messages` | `@workflow`, `@task`, `@agent` |
| `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` | `@tool` |
| `gen_ai.tool.type` | `@tool` (always `"function"`) |
| `gen_ai.tool.description` | `@tool` (from docstring, if present) |
| `gen_ai.agent.version` | All decorators (when `version` is set) |

**Supported function types:** sync, async, generators, async generators. Errors are captured as span status + exception events.

```python
@agent("research_agent", version=2)
async def research(query: str) -> str:
    """Agents create invoke_agent spans with gen_ai.agent.* attributes."""
    result = await search_tool(query)
    return summarize(result)

@tool("search")
def search_tool(query: str) -> list:
    """Docstring becomes gen_ai.tool.description. Input/output use gen_ai.tool.call.* attributes."""
    return api.search(query)
```

### `score()`

Submits evaluation scores as OTEL spans. Use any evaluation framework you prefer (autoevals, RAGAS, custom) and submit the results through `score()`.

**Three scoring levels:**

```python
# Span-level: score a specific LLM call or tool execution
score(
    name="accuracy",
    value=0.95,
    trace_id="abc123",
    span_id="def456",
    explanation="Weather data matches ground truth",
    source="heuristic",
)

# Trace-level: score an entire workflow
score(
    name="relevance",
    value=0.92,
    trace_id="abc123",
    explanation="Response addresses the user's query",
    source="llm-judge",
)

# Session-level: score across multiple traces in a conversation
score(
    name="user_satisfaction",
    value=0.88,
    conversation_id="session-123",
    label="satisfied",
    source="human",
)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Metric name (e.g., `"relevance"`, `"factuality"`) |
| `value` | `float` | Numeric score |
| `trace_id` | `str` | Trace being scored (span/trace-level) |
| `span_id` | `str` | Span being scored (span-level) |
| `conversation_id` | `str` | Session being scored (session-level) |
| `label` | `str` | Human-readable label (`"pass"`, `"relevant"`) |
| `explanation` | `str` | Evaluator justification (truncated to 500 chars) |
| `response_id` | `str` | LLM completion ID for correlation |
| `source` | `str` | Score origin: `"sdk"`, `"human"`, `"llm-judge"`, `"heuristic"` |
| `metadata` | `dict` | Arbitrary key-value metadata |

Scores are emitted as `gen_ai.evaluation.result` spans with `gen_ai.evaluation.*` attributes, following the OTEL GenAI semantic conventions.

## Auto-Instrumented Libraries

`register()` automatically discovers and activates installed instrumentor packages. No code changes needed — just install the package and calls are traced.

**LLM Providers:**
OpenAI, Anthropic, Google Generative AI, Cohere, Mistral AI, Groq, Ollama, Together, Replicate, Writer, Voyage AI, Aleph Alpha

**Cloud AI Services:**
AWS Bedrock, AWS SageMaker, Google Vertex AI, IBM watsonx

**Frameworks:**
LangChain, LlamaIndex, Haystack, CrewAI, Agno, MCP, Transformers, OpenAI Agents

**Vector Databases:**
ChromaDB, Pinecone, Qdrant, Weaviate, Milvus, LanceDB, Marqo

## Configuration

| Environment Variable | Description | Default |
|---|---|---|
| `OPENSEARCH_OTEL_ENDPOINT` | OTLP endpoint URL | `http://localhost:21890/opentelemetry/v1/traces` |
| `OTEL_SERVICE_NAME` | Service name for spans | `"default"` |
| `OPENSEARCH_PROJECT` | Project/service name (fallback) | `"default"` |
| `AWS_DEFAULT_REGION` | AWS region for SigV4 | auto-detected |

## License

Apache-2.0
