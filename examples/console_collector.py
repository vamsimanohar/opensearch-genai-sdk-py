"""Console-based OTEL setup — see spans without a real collector.

This example replaces the need for a running OTEL collector or Data Prepper
instance. Spans are printed to stdout as JSON, which is useful for:

- Local development and debugging
- Verifying that decorators produce the expected span tree
- CI/CD smoke tests

Run:
    python examples/console_collector.py
"""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# --- Set up ConsoleSpanExporter as the global tracer provider ---
# This prints every span to stdout as JSON when the span ends.
# No collector, no Data Prepper, no network needed.
provider = TracerProvider(resource=Resource.create({"service.name": "console-collector-demo"}))
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# --- Now import SDK decorators (they use the global provider) ---
from opensearch_genai_sdk import agent, task, tool, workflow


@tool(name="web_search")
def search(query: str) -> list[dict]:
    """Simulated web search tool."""
    return [{"title": f"Result for: {query}", "url": "https://example.com"}]


@task(name="summarize")
def summarize(text: str) -> str:
    """Simulated LLM summarization."""
    return f"Summary of: {text[:100]}"


@agent(name="research_agent")
def research(query: str) -> str:
    """Agent that searches, then summarizes."""
    results = search(query)
    titles = ", ".join(r["title"] for r in results)
    return summarize(titles)


@workflow(name="qa_pipeline")
def run_pipeline(question: str) -> str:
    """Top-level workflow orchestrating the agent."""
    answer = research(question)
    return answer


# --- Run ---
if __name__ == "__main__":
    result = run_pipeline("What is OpenSearch?")
    print(f"\n{'=' * 60}")
    print(f"Final answer: {result}")
    print(f"{'=' * 60}")
    print("\nAll spans above were printed by ConsoleSpanExporter.")
    print("No collector needed — useful for local dev and debugging.")

    # Produces this span tree (printed as JSON to stdout):
    #
    #   qa_pipeline          (workflow)
    #   +-- research_agent   (agent)
    #       +-- web_search   (tool)
    #       +-- summarize    (task)
