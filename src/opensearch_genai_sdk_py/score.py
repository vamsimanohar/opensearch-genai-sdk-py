"""Score submission for OpenSearch AI observability.

Sends evaluation scores as OTEL spans using gen_ai.evaluation.*
semantic convention attributes. Supports three scoring levels:

- **Span-level:** trace_id + span_id — score a specific span
- **Trace-level:** trace_id only — score the entire trace
- **Session-level:** conversation_id — score across traces

This keeps everything in OTEL — no separate OpenSearch client needed
for scoring. Same SigV4 auth, same exporter, same pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace

logger = logging.getLogger(__name__)

_TRACER_NAME = "opensearch-genai-sdk-py-scores"


def score(
    name: str,
    value: float | None = None,
    *,
    trace_id: str | None = None,
    span_id: str | None = None,
    conversation_id: str | None = None,
    label: str | None = None,
    explanation: str | None = None,
    response_id: str | None = None,
    source: str = "sdk",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Submit an evaluation score as an OTEL span.

    Creates a span with gen_ai.evaluation.* attributes that flows
    through the same OTLP exporter as all other traces.

    Three scoring levels:
    - **Span-level:** pass trace_id + span_id to score a specific span.
    - **Trace-level:** pass only trace_id to score the entire trace.
    - **Session-level:** pass conversation_id to score across traces.

    Args:
        name: Evaluation metric name (e.g., "relevance", "factuality").
        value: Numeric score value.
        trace_id: The trace ID being scored. Stored as an attribute
            (does NOT become the span's own trace ID).
        span_id: Span ID for span-level scoring.
        conversation_id: Conversation/session ID for session-level scoring.
        label: Human-readable label (e.g., "pass", "relevant", "satisfied").
        explanation: Evaluator justification or rationale.
        response_id: Completion ID for correlation with a specific response.
        source: Who created the score — "sdk", "human", "llm-judge", "heuristic".
        metadata: Optional arbitrary metadata.

    Example:
        from opensearch_genai_sdk_py import score

        # Span-level scoring
        score(
            name="accuracy",
            value=0.95,
            trace_id="abc123",
            span_id="def456",
            explanation="Weather data is correct",
            source="heuristic",
        )

        # Trace-level scoring
        score(
            name="relevance",
            value=0.92,
            trace_id="abc123",
            explanation="Response addresses the query",
            source="llm-judge",
        )

        # Session-level scoring
        score(
            name="user_satisfaction",
            value=0.88,
            conversation_id="session-123",
            label="satisfied",
            source="human",
        )
    """
    tracer = trace.get_tracer(_TRACER_NAME)

    attrs: dict[str, Any] = {
        "gen_ai.evaluation.name": name,
        "gen_ai.evaluation.source": source,
    }

    if value is not None:
        attrs["gen_ai.evaluation.score.value"] = value
    if trace_id:
        attrs["gen_ai.evaluation.trace_id"] = trace_id
    if span_id:
        attrs["gen_ai.evaluation.span_id"] = span_id
    if conversation_id:
        attrs["gen_ai.conversation.id"] = conversation_id
    if label:
        attrs["gen_ai.evaluation.score.label"] = label
    if explanation:
        attrs["gen_ai.evaluation.explanation"] = explanation[:500]
    if response_id:
        attrs["gen_ai.response.id"] = response_id
    if metadata:
        for k, v in metadata.items():
            attrs[f"gen_ai.evaluation.metadata.{k}"] = str(v)

    with tracer.start_as_current_span("gen_ai.evaluation.result", attributes=attrs):
        logger.debug("Score emitted: %s=%s (trace=%s)", name, value, trace_id)
