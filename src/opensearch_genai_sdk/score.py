"""Score submission for OpenSearch AI observability.

Sends evaluation scores and human feedback as OTEL spans through the
same exporter pipeline as all other traces. Data Prepper routes these
spans to the ai-scores index based on the `opensearch.score` attribute.

This keeps everything in OTEL — no separate OpenSearch client needed
for scoring. Same SigV4 auth, same exporter, same pipeline.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Literal, Optional

from opentelemetry import trace

logger = logging.getLogger(__name__)

_TRACER_NAME = "opensearch-genai-sdk-scores"


def score(
    name: str,
    value: Optional[float] = None,
    *,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    label: Optional[str] = None,
    data_type: Literal["NUMERIC", "CATEGORICAL", "BOOLEAN"] = "NUMERIC",
    source: str = "sdk",
    comment: Optional[str] = None,
    rationale: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    project: Optional[str] = None,
) -> None:
    """Submit a score as an OTEL span.

    Creates a span with score attributes that flows through the same
    OTLP exporter as all other traces. Data Prepper can route these
    to a dedicated index based on the ``opensearch.score`` marker.

    Args:
        name: Score name (e.g., "relevance", "factuality", "toxicity").
        value: Numeric score value (0.0 to 1.0 for NUMERIC).
        trace_id: The trace ID being scored. Stored as an attribute
            (does NOT become the span's own trace ID).
        span_id: Optional span ID for span-level scoring.
        label: Categorical label (for CATEGORICAL scores).
        data_type: Score type — NUMERIC, CATEGORICAL, or BOOLEAN.
        source: Who created the score — "sdk", "human", "llm-judge", "heuristic".
        comment: Optional human-readable comment.
        rationale: Optional explanation from an LLM judge.
        metadata: Optional arbitrary metadata.
        project: Project name. Defaults to OPENSEARCH_PROJECT env var.

    Example:
        from opensearch_genai_sdk import score

        score(
            name="relevance",
            value=0.95,
            trace_id="abc123",
            source="llm-judge",
            rationale="Answer directly addresses the question",
        )
    """
    tracer = trace.get_tracer(_TRACER_NAME)

    attrs: Dict[str, Any] = {
        "opensearch.score": True,
        "score.name": name,
        "score.data_type": data_type,
        "score.source": source,
        "score.project": project or os.environ.get("OPENSEARCH_PROJECT", "default"),
    }

    if value is not None:
        attrs["score.value"] = value
    if trace_id:
        attrs["score.trace_id"] = trace_id
    if span_id:
        attrs["score.span_id"] = span_id
    if label:
        attrs["score.label"] = label
    if comment:
        attrs["score.comment"] = comment
    if rationale:
        attrs["score.rationale"] = rationale[:500]
    if metadata:
        for k, v in metadata.items():
            attrs[f"score.metadata.{k}"] = str(v)

    with tracer.start_as_current_span(f"score.{name}", attributes=attrs):
        logger.debug("Score emitted: %s=%s (trace=%s)", name, value, trace_id)
