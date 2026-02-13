"""Evaluation framework for OpenSearch AI observability.

Provides the evaluate() orchestrator and Scorer protocol for running
evaluations on LLM outputs. Compatible with autoevals, phoenix-evals,
or any custom scorer that matches the Scorer protocol.
"""

from opensearch_genai_sdk.evals.protocol import Score, Scorer
from opensearch_genai_sdk.evals.evaluate import evaluate

__all__ = ["evaluate", "Score", "Scorer"]
