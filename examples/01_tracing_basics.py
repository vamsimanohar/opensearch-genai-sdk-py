"""Basic tracing with opensearch-genai-sdk.

Shows how to register the SDK and trace custom functions
with @workflow, @task, @agent, and @tool decorators.
"""

from opensearch_genai_sdk import register, workflow, task, agent, tool

# --- Setup ---
# Local Data Prepper
register(endpoint="http://localhost:21890/opentelemetry/v1/traces")

# AWS-hosted (SigV4 is auto-detected from the hostname)
# register(endpoint="https://my-pipeline.us-east-1.osis.amazonaws.com/v1/traces")


# --- Decorators ---
@tool(name="web_search")
def search(query: str) -> list[dict]:
    """Simulated web search tool."""
    return [
        {"title": f"Result for: {query}", "url": "https://example.com"},
    ]


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
    """Top-level workflow that orchestrates the agent."""
    answer = research(question)
    return answer


# --- Run ---
if __name__ == "__main__":
    result = run_pipeline("What is OpenSearch?")
    print(result)

    # Produces this span tree:
    #
    #   qa_pipeline          (workflow)
    #   └── research_agent   (agent)
    #       ├── web_search   (tool)
    #       └── summarize    (task)
