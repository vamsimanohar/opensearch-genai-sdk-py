"""Dummy agent → OTEL Collector via HTTP.

Sends spans to http://localhost:4318/v1/traces (OTLP/HTTP).
Watch the collector terminal to see them arrive.

Usage:
    # Terminal 1: docker compose up
    # Terminal 2: python examples/agent_http.py
"""

import random
import time

from opentelemetry import trace

from opensearch_genai_sdk_py import agent, register, score, task, tool, workflow

# --- register() with HTTP endpoint ---
register(
    endpoint="http://localhost:4318/v1/traces",
    service_name="agent-http-demo",
    batch=False,  # SimpleSpanProcessor — flush immediately for demo
)


# --- Tools ---
@tool(name="web_search")
def web_search(query: str) -> list[dict]:
    time.sleep(random.uniform(0.05, 0.1))
    return [
        {"title": f"Result 1 for: {query}", "snippet": "OpenSearch is open-source."},
        {"title": f"Result 2 for: {query}", "snippet": "Supports vector search and observability."},
    ]


@tool(name="calculator")
def calculator(expr: str) -> float:
    time.sleep(0.02)
    return 42.0


# --- Tasks ---
@task(name="plan")
def plan(question: str) -> list[str]:
    time.sleep(0.03)
    return [question, f"{question} features"]


@task(name="summarize")
def summarize(results: list[dict]) -> str:
    time.sleep(0.05)
    snippets = [r["snippet"] for r in results]
    return f"Summary: {'. '.join(snippets)}"


# --- Agent ---
@agent(name="research_agent")
def research(question: str) -> str:
    queries = plan(question)
    all_results = []
    for q in queries:
        all_results.extend(web_search(q))
    return summarize(all_results)


# --- Workflow ---
@workflow(name="qa_workflow")
def run(question: str) -> str:
    answer = research(question)

    ctx = trace.get_current_span().get_span_context()
    tid = format(ctx.trace_id, "032x")
    sid = format(ctx.span_id, "016x")

    score(name="relevance", value=0.95, trace_id=tid, span_id=sid, source="llm-judge")
    score(name="completeness", value=0.88, trace_id=tid, span_id=sid, source="llm-judge")

    return answer


# --- Run ---
if __name__ == "__main__":
    print("Protocol: HTTP  →  http://localhost:4318/v1/traces")
    print("=" * 60)
    result = run("What is OpenSearch?")
    print(f"\nAnswer: {result}")
    print("\n→ Check the collector terminal for spans.")
