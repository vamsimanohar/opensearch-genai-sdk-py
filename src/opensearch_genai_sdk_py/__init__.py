"""OpenSearch AI Observability SDK.

OTEL-native tracing and scoring for LLM applications.
"""

from opensearch_genai_sdk_py.decorators import agent, task, tool, workflow
from opensearch_genai_sdk_py.exporters import AWSSigV4OTLPExporter
from opensearch_genai_sdk_py.register import register
from opensearch_genai_sdk_py.score import score

__all__ = [
    # Setup
    "register",
    # Decorators
    "workflow",
    "task",
    "agent",
    "tool",
    # Scoring
    "score",
    # Exporters
    "AWSSigV4OTLPExporter",
]
