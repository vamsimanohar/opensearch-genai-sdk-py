"""Weather research agent — Anthropic + OpenAI auto-instrumentation demo.

Sends real traces to the lightweight trace collector (no Docker needed).
Start it first from the repo root:

    python ../trace_collector.py      # listens on http://localhost:4318

Then run:

    ANTHROPIC_API_KEY=sk-ant-... OPENAI_API_KEY=sk-... python examples/08_anthropic_openai_agent.py

What this demo does
-------------------
1. register() — points the SDK at the local OTEL Collector on port 4318.
   Auto-instrumentation activates for both the anthropic and openai libraries.
2. @tool("get_weather")  — simulates a weather API (no real HTTP call).
3. @tool("summarize_anthropic") — calls claude-haiku-4-5 to summarise weather data.
4. @tool("summarize_openai")    — calls gpt-4o-mini to summarise weather data.
5. @agent("weather_agent")      — orchestrates the two LLM tools.
6. @workflow("weather_workflow") — top-level span + score() calls.

Spans visible in trace_collector.py stdout:
  weather_workflow
  +-- invoke_agent weather_agent
  |   +-- execute_tool get_weather
  |   +-- execute_tool summarize_anthropic  (+ child LLM span from auto-instr.)
  |   +-- execute_tool summarize_openai     (+ child LLM span from auto-instr.)
  +-- gen_ai.evaluation.result  (relevance)
  +-- gen_ai.evaluation.result  (accuracy)
"""

from __future__ import annotations

import os
import sys

from opentelemetry import trace as otel_trace

from opensearch_genai_sdk_py import agent, register, score, tool, workflow

# ---------------------------------------------------------------------------
# 1. Point SDK at the local OTEL Collector
# ---------------------------------------------------------------------------
register(
    endpoint="http://localhost:4318/v1/traces",
    service_name="weather-agent-demo",
    auto_instrument=True,
)

# ---------------------------------------------------------------------------
# 2. Import LLM clients AFTER register() so auto-instrumentation hooks fire
# ---------------------------------------------------------------------------
try:
    import anthropic as _anthropic

    anthropic_client = _anthropic.Anthropic()
    HAS_ANTHROPIC = True
except Exception:
    HAS_ANTHROPIC = False

try:
    import openai as _openai

    openai_client = _openai.OpenAI()
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False


# ---------------------------------------------------------------------------
# 3. Tools
# ---------------------------------------------------------------------------
@tool("get_weather")
def get_weather(city: str) -> dict:
    """Return simulated weather data for a city."""
    data = {
        "Paris": {"temp_c": 18, "condition": "partly cloudy", "humidity": 65},
        "Tokyo": {"temp_c": 24, "condition": "sunny", "humidity": 55},
        "New York": {"temp_c": 12, "condition": "rainy", "humidity": 80},
    }
    return data.get(city, {"temp_c": 20, "condition": "unknown", "humidity": 60})


@tool("summarize_anthropic")
def summarize_with_anthropic(city: str, weather: dict) -> str:
    """Ask Claude to write a one-sentence weather summary (auto-instrumented)."""
    if not HAS_ANTHROPIC:
        return f"[Anthropic not available] {city}: {weather['condition']}, {weather['temp_c']}°C"

    prompt = (
        f"Write one sentence describing the weather in {city}: "
        f"{weather['condition']}, {weather['temp_c']}°C, humidity {weather['humidity']}%."
    )
    try:
        msg = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        return f"[Anthropic error: {exc}] {city}: {weather['condition']}"


@tool("summarize_openai")
def summarize_with_openai(city: str, weather: dict) -> str:
    """Ask GPT to write a one-sentence weather summary (auto-instrumented)."""
    if not HAS_OPENAI:
        return f"[OpenAI not available] {city}: {weather['condition']}, {weather['temp_c']}°C"

    prompt = (
        f"Write one sentence describing the weather in {city}: "
        f"{weather['condition']}, {weather['temp_c']}°C, humidity {weather['humidity']}%."
    )
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"[OpenAI error: {exc}] {city}: {weather['condition']}"


# ---------------------------------------------------------------------------
# 4. Agent
# ---------------------------------------------------------------------------
@agent("weather_agent")
def weather_agent(city: str) -> dict:
    """Fetch weather and get summaries from both Anthropic and OpenAI."""
    weather = get_weather(city)
    anthropic_summary = summarize_with_anthropic(city, weather)
    openai_summary = summarize_with_openai(city, weather)
    return {
        "city": city,
        "weather": weather,
        "anthropic_summary": anthropic_summary,
        "openai_summary": openai_summary,
    }


# ---------------------------------------------------------------------------
# 5. Workflow
# ---------------------------------------------------------------------------
@workflow("weather_workflow")
def weather_workflow(city: str) -> dict:
    """Run weather agent then score the output."""
    result = weather_agent(city)

    # Attach scores to the current (workflow) span
    ctx = otel_trace.get_current_span().get_span_context()
    trace_id = format(ctx.trace_id, "032x")
    span_id = format(ctx.span_id, "016x")

    score(
        name="relevance",
        value=0.95,
        trace_id=trace_id,
        span_id=span_id,
        source="heuristic",
        explanation="Weather data and LLM summaries are directly relevant to the city query.",
    )
    score(
        name="accuracy",
        value=0.90,
        trace_id=trace_id,
        span_id=span_id,
        source="heuristic",
        explanation="Temperature and condition data match expected ranges.",
    )
    return result


# ---------------------------------------------------------------------------
# 6. Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "Paris"

    print(f"Running weather workflow for: {city}")
    print(f"Anthropic available: {HAS_ANTHROPIC}")
    print(f"OpenAI available:    {HAS_OPENAI}")
    print(f"Sending traces to:   http://localhost:4318/v1/traces  (trace_collector.py)")
    print("-" * 60)

    result = weather_workflow(city)

    print(f"\nCity:     {result['city']}")
    print(f"Weather:  {result['weather']}")
    print(f"Anthropic: {result['anthropic_summary']}")
    print(f"OpenAI:    {result['openai_summary']}")
    print()
    print("Traces sent to collector. Run: docker-compose logs -f (in examples/)")
