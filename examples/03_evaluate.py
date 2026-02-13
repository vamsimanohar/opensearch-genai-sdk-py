"""Evaluation with opensearch-genai-sdk.

Shows how to run evaluate() with different scorer types:
- autoevals scorers (Factuality, Levenshtein)
- Custom scorers matching the Scorer protocol
- Plain functions returning floats

evaluate() creates OTEL spans for the entire flow and emits
scores through the same exporter pipeline.
"""

from opensearch_genai_sdk import register, evaluate
from opensearch_genai_sdk.evals import Score

# --- Setup ---
register(endpoint="http://localhost:21890/opentelemetry/v1/traces")


# --- Custom scorer ---
class ExactMatch:
    """Scorer that checks if output exactly matches expected."""

    name = "exact_match"

    def __call__(self, *, input: str, output: str, expected: str = None, **kwargs) -> Score:
        match = output.strip().lower() == (expected or "").strip().lower()
        return Score(
            name=self.name,
            value=1.0 if match else 0.0,
            label="match" if match else "mismatch",
        )


class ContainsKeyword:
    """Scorer that checks if a keyword appears in the output."""

    name = "contains_keyword"

    def __call__(self, *, input: str, output: str, expected: str = None, **kwargs) -> Score:
        keyword = (expected or "").strip().lower()
        found = keyword in output.lower()
        return Score(
            name=self.name,
            value=1.0 if found else 0.0,
            rationale=f"Keyword '{keyword}' {'found' if found else 'not found'} in output",
        )


# --- Dataset ---
dataset = [
    {"input": "What is the capital of France?", "expected": "Paris"},
    {"input": "What is 2 + 2?", "expected": "4"},
    {"input": "Who wrote Hamlet?", "expected": "Shakespeare"},
]


# --- Task function (your LLM call goes here) ---
def my_llm(input: str) -> str:
    """Replace with your actual LLM call."""
    answers = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Who wrote Hamlet?": "William Shakespeare",
    }
    return answers.get(input, "I don't know")


# --- Run evaluation ---
if __name__ == "__main__":
    results = evaluate(
        name="qa-eval",
        data=dataset,
        task=my_llm,
        scores=[ExactMatch(), ContainsKeyword()],
    )
    print(results)
    # Eval: qa-eval (3 samples, 0 errors)
    #   exact_match: 0.667
    #   contains_keyword: 1.000

    # Span tree for each data point:
    #
    #   evaluate                       (eval run)
    #   ├── eval_item [0]              (per data point)
    #   │   ├── eval_task              (task execution)
    #   │   ├── eval_score.exact_match (scorer)
    #   │   └── eval_score.contains_keyword
    #   ├── eval_item [1]
    #   │   └── ...
    #   └── eval_item [2]
    #       └── ...
    #
    # Plus score.exact_match and score.contains_keyword spans
    # emitted for each item (routed to ai-scores by Data Prepper)
