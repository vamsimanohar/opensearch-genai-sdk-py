"""A dummy research agent that demonstrates the full SDK.

Uses ConsoleSpanExporter so you can see the OTEL output directly.

Run:
    python examples/dummy_agent.py

This produces a complete span tree printed to stdout as JSON:

    research_workflow                    (workflow)
    +-- research_agent                   (agent)
    |   +-- plan_research                (task)
    |   +-- web_search                   (tool)       x N queries
    |   +-- calculator                   (tool)       optional
    |   +-- summarize_results            (task)
    +-- score.relevance                  (score span)
    +-- score.completeness               (score span)

No collector, no Data Prepper, no network needed.
"""

from __future__ import annotations

import random
import time

# ---------------------------------------------------------------------------
# 1. Set up ConsoleSpanExporter BEFORE importing SDK decorators.
#    This replaces register() for demo/local-dev purposes.
# ---------------------------------------------------------------------------
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider(
    resource=Resource.create({"service.name": "dummy-agent"})
)
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# ---------------------------------------------------------------------------
# 2. Import SDK decorators and score (they pick up the global provider).
# ---------------------------------------------------------------------------
from opensearch_genai_sdk import workflow, task, agent, tool, score

# ---------------------------------------------------------------------------
# Simulated knowledge base for the "web search"
# ---------------------------------------------------------------------------
_FAKE_RESULTS: dict[str, list[dict]] = {
    "opensearch features": [
        {
            "title": "OpenSearch Documentation - Key Features",
            "snippet": "OpenSearch is an open-source search and analytics suite. "
            "It includes full-text search, log analytics, security analytics, "
            "and observability powered by Piped Processing Language (PPL).",
        },
        {
            "title": "OpenSearch vs Elasticsearch comparison",
            "snippet": "OpenSearch offers native alerting, anomaly detection, "
            "and SQL/PPL query support out of the box.",
        },
    ],
    "opensearch observability": [
        {
            "title": "Trace Analytics in OpenSearch",
            "snippet": "OpenSearch Dashboards provides trace analytics for "
            "distributed tracing. Integrate with OTEL collectors and Data Prepper "
            "to ingest spans, service maps, and error rates.",
        },
    ],
    "opensearch ai": [
        {
            "title": "AI/ML in OpenSearch",
            "snippet": "OpenSearch supports vector search via k-NN plugin, "
            "conversational search with RAG, and ML Commons for deploying models. "
            "The GenAI SDK adds observability for LLM-powered applications.",
        },
    ],
}

_DEFAULT_RESULT = [
    {
        "title": "General OpenSearch information",
        "snippet": "OpenSearch is a community-driven, open-source project.",
    },
]


# ---------------------------------------------------------------------------
# 3. @tool — individual tool calls
# ---------------------------------------------------------------------------
@tool(name="web_search")
def web_search(query: str) -> list[dict]:
    """Simulated web search that returns canned results."""
    time.sleep(random.uniform(0.05, 0.15))  # simulate latency
    # Fuzzy match against our fake knowledge base
    for key, results in _FAKE_RESULTS.items():
        if key in query.lower():
            return results
    return _DEFAULT_RESULT


@tool(name="calculator")
def calculator(expression: str) -> float:
    """Simulated calculator tool for numeric computations."""
    time.sleep(0.02)
    # In a real agent this would eval safely; here we just return a fixed number.
    return 42.0


# ---------------------------------------------------------------------------
# 4. @task — individual processing steps
# ---------------------------------------------------------------------------
@task(name="plan_research")
def plan_research(question: str) -> list[str]:
    """Break a question into sub-queries to search for."""
    time.sleep(0.03)  # simulate thinking time

    # Simple heuristic: generate 2-3 search queries from the question
    base = question.lower().replace("?", "").strip()
    queries = [
        f"{base}",
        f"{base} features",
        f"{base} observability",
    ]
    return queries


@task(name="summarize_results")
def summarize_results(question: str, search_results: list[dict]) -> str:
    """Combine search results into a coherent answer."""
    time.sleep(random.uniform(0.05, 0.1))  # simulate LLM call

    snippets = [r["snippet"] for r in search_results]
    combined = " ".join(snippets)

    # Simulated "LLM summary"
    summary = (
        f"Based on {len(search_results)} sources: {combined[:300]}"
        if combined
        else "No relevant information found."
    )
    return summary


# ---------------------------------------------------------------------------
# 5. @agent — the autonomous research loop
# ---------------------------------------------------------------------------
@agent(name="research_agent")
def research_agent(question: str) -> str:
    """Research agent loop: plan -> search -> (optional calc) -> summarize.

    Demonstrates a realistic agent pattern where:
    - The agent plans what information it needs
    - Executes tool calls in a loop
    - Aggregates results
    - Produces a final summary
    """
    # Step 1: Plan
    queries = plan_research(question)

    # Step 2: Search (loop over planned queries)
    all_results: list[dict] = []
    for q in queries:
        results = web_search(q)
        all_results.extend(results)

    # Step 3: Optional tool call (e.g., compute something)
    if "how many" in question.lower() or "calculate" in question.lower():
        calc_result = calculator("len(results) * relevance_factor")
        all_results.append(
            {"title": "Calculation", "snippet": f"Computed value: {calc_result}"}
        )

    # Step 4: Summarize
    answer = summarize_results(question, all_results)
    return answer


# ---------------------------------------------------------------------------
# 6. @workflow — top-level orchestration
# ---------------------------------------------------------------------------
@workflow(name="research_workflow")
def research_workflow(question: str) -> str:
    """Full research workflow: run agent, then score the output.

    This is the entry point. It orchestrates the agent and then
    submits evaluation scores for the produced answer.
    """
    # Run the agent
    answer = research_agent(question)

    # Get the current span's trace ID so we can attach scores to it
    current_span = trace.get_current_span()
    ctx = current_span.get_span_context()
    trace_id = format(ctx.trace_id, "032x")
    span_id = format(ctx.span_id, "016x")

    # Step 7: Score the output
    score(
        name="relevance",
        value=0.92,
        trace_id=trace_id,
        span_id=span_id,
        source="llm-judge",
        rationale="The answer directly addresses the question with specific details "
        "about OpenSearch features, observability, and AI capabilities.",
    )

    score(
        name="completeness",
        value=0.85,
        trace_id=trace_id,
        span_id=span_id,
        source="llm-judge",
        rationale="Covers most aspects but could include more detail on "
        "vector search and neural search capabilities.",
    )

    return answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    question = "What is OpenSearch and how does it support AI observability?"

    print(f"Question: {question}")
    print(f"{'='*70}")
    print("Running research workflow... (spans print below as JSON)\n")

    result = research_workflow(question)

    print(f"\n{'='*70}")
    print(f"Answer: {result}")
    print(f"{'='*70}")
    print()
    print("Span tree produced:")
    print("  research_workflow              (workflow)")
    print("  +-- research_agent             (agent)")
    print("  |   +-- plan_research          (task)")
    print("  |   +-- web_search             (tool) x3 queries")
    print("  |   +-- summarize_results      (task)")
    print("  +-- score.relevance            (score)")
    print("  +-- score.completeness         (score)")
    print()
    print("All spans were printed to stdout by ConsoleSpanExporter.")
    print("No collector or backend needed.")
