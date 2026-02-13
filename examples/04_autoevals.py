"""Using autoevals scorers with opensearch-genai-sdk.

Shows how to plug in autoevals scorers (Factuality, Levenshtein, etc.)
directly into evaluate(). The adapt_score() function handles converting
autoevals result objects to our Score dataclass.

Requires: pip install opensearch-genai-sdk[evals]
"""

from opensearch_genai_sdk import register, evaluate

# --- Setup ---
register(endpoint="http://localhost:21890/opentelemetry/v1/traces")

# --- Dataset ---
dataset = [
    {
        "input": "What is the capital of France?",
        "expected": "Paris is the capital of France.",
    },
    {
        "input": "Explain photosynthesis in one sentence.",
        "expected": "Photosynthesis converts sunlight into chemical energy in plants.",
    },
]


def my_llm(input: str) -> str:
    """Replace with your actual LLM call."""
    return "Paris is the capital of France."


if __name__ == "__main__":
    # autoevals scorers work out of the box
    from autoevals import Factuality, Levenshtein

    results = evaluate(
        name="qa-factuality",
        data=dataset,
        task=my_llm,
        scores=[Factuality(), Levenshtein()],
    )
    print(results)
    # Eval: qa-factuality (2 samples, 0 errors)
    #   Factuality: 0.850
    #   Levenshtein: 0.720
