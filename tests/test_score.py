"""Tests for opensearch_genai_sdk.score.

Verifies that the score() function creates OTEL spans with
gen_ai.evaluation.* semantic convention attributes for span-level,
trace-level, and session-level scoring.
"""


from opensearch_genai_sdk.score import score


class TestSpanLevelScoring:
    """Test span-level scoring (trace_id + span_id)."""

    def test_span_level_score(self, exporter):
        score(name="accuracy", value=0.95, trace_id="abc123", span_id="def456")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "gen_ai.evaluation.result"
        assert span.attributes["gen_ai.evaluation.name"] == "accuracy"
        assert span.attributes["gen_ai.evaluation.score.value"] == 0.95
        assert span.attributes["gen_ai.evaluation.trace_id"] == "abc123"
        assert span.attributes["gen_ai.evaluation.span_id"] == "def456"
        assert span.attributes["gen_ai.evaluation.source"] == "sdk"

    def test_span_level_with_explanation(self, exporter):
        score(
            name="accuracy",
            value=0.95,
            trace_id="abc123",
            span_id="def456",
            explanation="Weather data is correct",
            source="heuristic",
        )

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["gen_ai.evaluation.explanation"] == "Weather data is correct"
        assert span.attributes["gen_ai.evaluation.source"] == "heuristic"


class TestTraceLevelScoring:
    """Test trace-level scoring (trace_id only)."""

    def test_trace_level_score(self, exporter):
        score(name="relevance", value=0.92, trace_id="abc123")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.attributes["gen_ai.evaluation.name"] == "relevance"
        assert span.attributes["gen_ai.evaluation.score.value"] == 0.92
        assert span.attributes["gen_ai.evaluation.trace_id"] == "abc123"
        assert "gen_ai.evaluation.span_id" not in span.attributes

    def test_trace_level_with_explanation(self, exporter):
        score(
            name="relevance",
            value=0.92,
            trace_id="abc123",
            explanation="Response addresses the query",
            source="llm-judge",
        )

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["gen_ai.evaluation.explanation"] == "Response addresses the query"
        assert span.attributes["gen_ai.evaluation.source"] == "llm-judge"


class TestSessionLevelScoring:
    """Test session-level scoring (conversation_id)."""

    def test_session_level_score(self, exporter):
        score(
            name="user_satisfaction",
            value=0.88,
            conversation_id="session-123",
            label="satisfied",
            source="human",
        )

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.attributes["gen_ai.evaluation.name"] == "user_satisfaction"
        assert span.attributes["gen_ai.evaluation.score.value"] == 0.88
        assert span.attributes["gen_ai.conversation.id"] == "session-123"
        assert span.attributes["gen_ai.evaluation.score.label"] == "satisfied"
        assert span.attributes["gen_ai.evaluation.source"] == "human"
        assert "gen_ai.evaluation.trace_id" not in span.attributes
        assert "gen_ai.evaluation.span_id" not in span.attributes


class TestScoreValues:
    """Test numeric score value handling."""

    def test_score_with_zero_value(self, exporter):
        score(name="toxicity", value=0.0, trace_id="t1")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["gen_ai.evaluation.score.value"] == 0.0

    def test_score_with_one_value(self, exporter):
        score(name="perfect", value=1.0, trace_id="t2")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["gen_ai.evaluation.score.value"] == 1.0

    def test_score_no_value(self, exporter):
        score(name="no_val")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert "gen_ai.evaluation.score.value" not in span.attributes

    def test_score_source_override(self, exporter):
        score(name="relevance", value=0.8, source="llm-judge")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["gen_ai.evaluation.source"] == "llm-judge"

    def test_score_source_human(self, exporter):
        score(name="quality", value=0.7, source="human")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["gen_ai.evaluation.source"] == "human"

    def test_score_default_source(self, exporter):
        score(name="test", value=0.5)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["gen_ai.evaluation.source"] == "sdk"


class TestScoreLabel:
    """Test label attribute."""

    def test_label_set(self, exporter):
        score(name="sentiment", label="positive")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["gen_ai.evaluation.score.label"] == "positive"

    def test_label_with_value(self, exporter):
        score(name="grade", value=0.9, label="A")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["gen_ai.evaluation.score.value"] == 0.9
        assert span.attributes["gen_ai.evaluation.score.label"] == "A"

    def test_no_label(self, exporter):
        score(name="test", value=0.5)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert "gen_ai.evaluation.score.label" not in span.attributes


class TestExplanation:
    """Test explanation attribute."""

    def test_explanation(self, exporter):
        score(
            name="test",
            value=0.9,
            explanation="The answer correctly addresses the question.",
        )

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert (
            span.attributes["gen_ai.evaluation.explanation"]
            == "The answer correctly addresses the question."
        )

    def test_explanation_truncated_at_500(self, exporter):
        long_explanation = "x" * 1000
        score(name="test", value=0.5, explanation=long_explanation)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert len(span.attributes["gen_ai.evaluation.explanation"]) == 500

    def test_no_explanation(self, exporter):
        score(name="test", value=0.5)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert "gen_ai.evaluation.explanation" not in span.attributes


class TestResponseId:
    """Test response_id correlation attribute."""

    def test_response_id(self, exporter):
        score(name="test", value=0.9, response_id="chatcmpl-abc123")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["gen_ai.response.id"] == "chatcmpl-abc123"

    def test_no_response_id(self, exporter):
        score(name="test", value=0.5)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert "gen_ai.response.id" not in span.attributes


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
        assert span.attributes["gen_ai.evaluation.metadata.model"] == "gpt-4"
        assert span.attributes["gen_ai.evaluation.metadata.temperature"] == "0.7"

    def test_no_metadata(self, exporter):
        score(name="test", value=0.5)

        spans = exporter.get_finished_spans()
        span = spans[0]
        # No gen_ai.evaluation.metadata.* keys should exist
        meta_keys = [k for k in span.attributes if k.startswith("gen_ai.evaluation.metadata.")]
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
        assert span.attributes["gen_ai.evaluation.metadata.details"] == "{'nested': True}"


class TestScoreTraceAndSpanIds:
    """Test trace_id and span_id attribute handling."""

    def test_trace_id_stored_as_attribute(self, exporter):
        score(name="test", value=0.5, trace_id="deadbeef")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["gen_ai.evaluation.trace_id"] == "deadbeef"
        # The span's own trace_id is NOT the same as the evaluation trace_id
        own_trace_id = format(span.context.trace_id, "032x")
        assert own_trace_id != "deadbeef"

    def test_no_trace_id(self, exporter):
        score(name="test", value=0.5)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert "gen_ai.evaluation.trace_id" not in span.attributes

    def test_span_id_present(self, exporter):
        score(name="test", value=0.5, trace_id="t1", span_id="s1")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["gen_ai.evaluation.span_id"] == "s1"

    def test_no_span_id(self, exporter):
        score(name="test", value=0.5)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert "gen_ai.evaluation.span_id" not in span.attributes


class TestScoreSpanName:
    """Test span naming conventions."""

    def test_span_name_is_evaluation_result(self, exporter):
        score(name="factuality", value=0.8)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.name == "gen_ai.evaluation.result"

    def test_all_scores_use_same_span_name(self, exporter):
        score(name="relevance", value=0.8)
        score(name="toxicity", value=0.1)

        spans = exporter.get_finished_spans()
        names = {s.name for s in spans}
        assert names == {"gen_ai.evaluation.result"}


class TestMultipleScoresInSequence:
    """Test that multiple score calls create independent spans."""

    def test_multiple_scores(self, exporter):
        score(name="a", value=0.1)
        score(name="b", value=0.2)
        score(name="c", value=0.3)

        spans = exporter.get_finished_spans()
        assert len(spans) == 3
        values = {
            s.attributes["gen_ai.evaluation.name"]: s.attributes["gen_ai.evaluation.score.value"]
            for s in spans
        }
        assert values == {"a": 0.1, "b": 0.2, "c": 0.3}
