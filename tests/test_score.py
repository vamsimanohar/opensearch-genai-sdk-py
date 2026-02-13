"""Tests for opensearch_genai_sdk.score.

Verifies that the score() function creates OTEL spans with the
correct attributes for numeric, categorical, and boolean scores,
including metadata handling and default values.
"""

import os

import pytest
from opentelemetry.trace import StatusCode

from opensearch_genai_sdk.score import score


class TestScoreNumeric:
    """Test numeric score submissions."""

    def test_basic_numeric_score(self, exporter):
        score(name="relevance", value=0.95, trace_id="abc123")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "score.relevance"
        assert span.attributes["opensearch.score"] is True
        assert span.attributes["score.name"] == "relevance"
        assert span.attributes["score.value"] == 0.95
        assert span.attributes["score.trace_id"] == "abc123"
        assert span.attributes["score.data_type"] == "NUMERIC"
        assert span.attributes["score.source"] == "sdk"

    def test_score_with_zero_value(self, exporter):
        score(name="toxicity", value=0.0, trace_id="t1")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["score.value"] == 0.0

    def test_score_with_one_value(self, exporter):
        score(name="perfect", value=1.0, trace_id="t2")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["score.value"] == 1.0

    def test_score_no_value(self, exporter):
        score(name="no_val")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert "score.value" not in span.attributes

    def test_score_source_override(self, exporter):
        score(name="relevance", value=0.8, source="llm-judge")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["score.source"] == "llm-judge"

    def test_score_source_human(self, exporter):
        score(name="quality", value=0.7, source="human")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["score.source"] == "human"


class TestScoreCategorical:
    """Test categorical score submissions."""

    def test_categorical_score(self, exporter):
        score(
            name="sentiment",
            label="positive",
            data_type="CATEGORICAL",
            trace_id="cat1",
        )

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.name == "score.sentiment"
        assert span.attributes["score.name"] == "sentiment"
        assert span.attributes["score.label"] == "positive"
        assert span.attributes["score.data_type"] == "CATEGORICAL"
        assert span.attributes["score.trace_id"] == "cat1"

    def test_categorical_with_value_and_label(self, exporter):
        score(
            name="grade",
            value=0.9,
            label="A",
            data_type="CATEGORICAL",
        )

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["score.value"] == 0.9
        assert span.attributes["score.label"] == "A"


class TestScoreBoolean:
    """Test boolean score submissions."""

    def test_boolean_score_pass(self, exporter):
        score(
            name="factual",
            value=1.0,
            data_type="BOOLEAN",
            trace_id="bool1",
        )

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["score.data_type"] == "BOOLEAN"
        assert span.attributes["score.value"] == 1.0

    def test_boolean_score_fail(self, exporter):
        score(
            name="factual",
            value=0.0,
            data_type="BOOLEAN",
            trace_id="bool2",
        )

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["score.value"] == 0.0


class TestScoreOptionalFields:
    """Test optional fields: span_id, comment, rationale, metadata."""

    def test_span_id(self, exporter):
        score(name="test", value=0.5, trace_id="t1", span_id="s1")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["score.span_id"] == "s1"

    def test_no_span_id(self, exporter):
        score(name="test", value=0.5)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert "score.span_id" not in span.attributes

    def test_comment(self, exporter):
        score(name="test", value=0.5, comment="looks good")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["score.comment"] == "looks good"

    def test_no_comment(self, exporter):
        score(name="test", value=0.5)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert "score.comment" not in span.attributes

    def test_rationale(self, exporter):
        score(
            name="test",
            value=0.9,
            rationale="The answer correctly addresses the question.",
        )

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["score.rationale"] == "The answer correctly addresses the question."

    def test_rationale_truncated_at_500(self, exporter):
        long_rationale = "x" * 1000
        score(name="test", value=0.5, rationale=long_rationale)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert len(span.attributes["score.rationale"]) == 500

    def test_no_rationale(self, exporter):
        score(name="test", value=0.5)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert "score.rationale" not in span.attributes


class TestScoreMetadata:
    """Test metadata handling."""

    def test_metadata_flat_values(self, exporter):
        score(
            name="test",
            value=0.5,
            metadata={"model": "gpt-4", "temperature": 0.7},
        )

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["score.metadata.model"] == "gpt-4"
        assert span.attributes["score.metadata.temperature"] == "0.7"

    def test_no_metadata(self, exporter):
        score(name="test", value=0.5)

        spans = exporter.get_finished_spans()
        span = spans[0]
        # No score.metadata.* keys should exist
        meta_keys = [k for k in span.attributes if k.startswith("score.metadata.")]
        assert meta_keys == []

    def test_metadata_with_nested_value(self, exporter):
        score(
            name="test",
            value=0.5,
            metadata={"details": {"nested": True}},
        )

        spans = exporter.get_finished_spans()
        span = spans[0]
        # Nested values are stringified
        assert span.attributes["score.metadata.details"] == "{'nested': True}"


class TestScoreProject:
    """Test project name handling."""

    def test_explicit_project(self, exporter):
        score(name="test", value=0.5, project="my-project")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["score.project"] == "my-project"

    def test_default_project(self, exporter):
        # Ensure env var is not set for this test
        old = os.environ.pop("OPENSEARCH_PROJECT", None)
        try:
            score(name="test", value=0.5)
            spans = exporter.get_finished_spans()
            span = spans[0]
            assert span.attributes["score.project"] == "default"
        finally:
            if old is not None:
                os.environ["OPENSEARCH_PROJECT"] = old

    def test_project_from_env(self, exporter):
        old = os.environ.get("OPENSEARCH_PROJECT")
        os.environ["OPENSEARCH_PROJECT"] = "env-project"
        try:
            score(name="test", value=0.5)
            spans = exporter.get_finished_spans()
            span = spans[0]
            assert span.attributes["score.project"] == "env-project"
        finally:
            if old is not None:
                os.environ["OPENSEARCH_PROJECT"] = old
            else:
                os.environ.pop("OPENSEARCH_PROJECT", None)


class TestScoreTraceAndSpanIds:
    """Test trace_id and span_id attribute handling."""

    def test_trace_id_stored_as_attribute(self, exporter):
        score(name="test", value=0.5, trace_id="deadbeef")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["score.trace_id"] == "deadbeef"
        # The span's own trace_id is NOT the same as score.trace_id
        own_trace_id = format(span.context.trace_id, "032x")
        assert own_trace_id != "deadbeef"

    def test_no_trace_id(self, exporter):
        score(name="test", value=0.5)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert "score.trace_id" not in span.attributes


class TestScoreSpanName:
    """Test span naming conventions."""

    def test_span_name_pattern(self, exporter):
        score(name="factuality", value=0.8)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.name == "score.factuality"

    def test_different_score_names_produce_different_span_names(self, exporter):
        score(name="relevance", value=0.8)
        score(name="toxicity", value=0.1)

        spans = exporter.get_finished_spans()
        names = {s.name for s in spans}
        assert names == {"score.relevance", "score.toxicity"}


class TestMultipleScoresInSequence:
    """Test that multiple score calls create independent spans."""

    def test_multiple_scores(self, exporter):
        score(name="a", value=0.1)
        score(name="b", value=0.2)
        score(name="c", value=0.3)

        spans = exporter.get_finished_spans()
        assert len(spans) == 3
        values = {s.attributes["score.name"]: s.attributes["score.value"] for s in spans}
        assert values == {"a": 0.1, "b": 0.2, "c": 0.3}
