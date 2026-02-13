"""Tests for opensearch_genai_sdk.evals.evaluate.

Covers the evaluate() orchestrator: mock scorers, EvalResult/EvalSummary
structure, span tree verification, error handling, averages computation,
and score emission.
"""

import pytest
from opentelemetry.trace import StatusCode

from opensearch_genai_sdk.evals.evaluate import evaluate, EvalResult, EvalSummary
from opensearch_genai_sdk.evals.protocol import Score


# ---------------------------------------------------------------------------
# Helpers: mock scorers and tasks
# ---------------------------------------------------------------------------


class AlwaysOneScorer:
    """A scorer that always returns 1.0."""

    name = "always_one"

    def __call__(self, *, input, output, expected=None, **kwargs):
        return Score(name="always_one", value=1.0)


class AlwaysHalfScorer:
    """A scorer that always returns 0.5."""

    name = "always_half"

    def __call__(self, *, input, output, expected=None, **kwargs):
        return Score(name="always_half", value=0.5)


class ExactMatchScorer:
    """A scorer that returns 1.0 if output matches expected, else 0.0."""

    name = "exact_match"

    def __call__(self, *, input, output, expected=None, **kwargs):
        match = 1.0 if output == expected else 0.0
        return Score(name="exact_match", value=match)


class FailingScorer:
    """A scorer that always raises an exception."""

    name = "failing_scorer"

    def __call__(self, *, input, output, expected=None, **kwargs):
        raise RuntimeError("scorer exploded")


class LabelScorer:
    """A scorer that returns a label and rationale."""

    name = "label_scorer"

    def __call__(self, *, input, output, expected=None, **kwargs):
        return Score(
            name="label_scorer",
            value=0.8,
            label="good",
            rationale="Looks good to me.",
        )


def echo_task(input_val):
    """A task that echoes the input as output."""
    return input_val


def uppercase_task(input_val):
    """A task that uppercases the input."""
    return str(input_val).upper()


def failing_task(input_val):
    """A task that always raises an error."""
    raise ValueError("task failed")


# ---------------------------------------------------------------------------
# EvalSummary / EvalResult dataclass tests
# ---------------------------------------------------------------------------


class TestEvalResultDataclass:
    """Test EvalResult structure."""

    def test_defaults(self):
        r = EvalResult(input="hello")
        assert r.input == "hello"
        assert r.output is None
        assert r.expected is None
        assert r.scores == {}
        assert r.error is None

    def test_with_scores(self):
        r = EvalResult(
            input="q",
            output="a",
            expected="a",
            scores={"m": Score(name="m", value=1.0)},
        )
        assert r.scores["m"].value == 1.0


class TestEvalSummaryDataclass:
    """Test EvalSummary structure."""

    def test_defaults(self):
        s = EvalSummary(name="test")
        assert s.name == "test"
        assert s.results == []
        assert s.averages == {}
        assert s.total == 0
        assert s.errors == 0

    def test_str_representation(self):
        s = EvalSummary(
            name="qa-eval",
            total=3,
            errors=1,
            averages={"relevance": 0.85, "fluency": 0.92},
        )
        text = str(s)
        assert "qa-eval" in text
        assert "3 samples" in text
        assert "1 errors" in text
        assert "relevance" in text
        assert "fluency" in text


# ---------------------------------------------------------------------------
# Basic evaluate() tests
# ---------------------------------------------------------------------------


class TestEvaluateBasic:
    """Test basic evaluate() functionality."""

    def test_simple_evaluation(self, exporter):
        data = [
            {"input": "hello", "expected": "hello"},
            {"input": "world", "expected": "world"},
        ]

        summary = evaluate(
            name="echo-eval",
            data=data,
            task=echo_task,
            scores=[AlwaysOneScorer()],
        )

        assert isinstance(summary, EvalSummary)
        assert summary.name == "echo-eval"
        assert summary.total == 2
        assert summary.errors == 0
        assert len(summary.results) == 2

        for result in summary.results:
            assert result.output == result.input
            assert "always_one" in result.scores
            assert result.scores["always_one"].value == 1.0

    def test_averages_computed_correctly(self, exporter):
        data = [
            {"input": "a", "expected": "A"},
            {"input": "b", "expected": "B"},
        ]

        summary = evaluate(
            name="avg-eval",
            data=data,
            task=uppercase_task,
            scores=[ExactMatchScorer()],
        )

        # uppercase_task("a") -> "A" == expected "A" -> 1.0
        # uppercase_task("b") -> "B" == expected "B" -> 1.0
        assert summary.averages["exact_match"] == 1.0

    def test_averages_with_mixed_scores(self, exporter):
        data = [
            {"input": "match", "expected": "MATCH"},
            {"input": "miss", "expected": "wrong"},
        ]

        summary = evaluate(
            name="mixed-eval",
            data=data,
            task=uppercase_task,
            scores=[ExactMatchScorer()],
        )

        # uppercase_task("match") -> "MATCH" == "MATCH" -> 1.0
        # uppercase_task("miss") -> "MISS" != "wrong" -> 0.0
        assert summary.averages["exact_match"] == 0.5

    def test_multiple_scorers(self, exporter):
        data = [{"input": "test", "expected": "test"}]

        summary = evaluate(
            name="multi-scorer",
            data=data,
            task=echo_task,
            scores=[AlwaysOneScorer(), AlwaysHalfScorer()],
        )

        assert len(summary.results[0].scores) == 2
        assert summary.averages["always_one"] == 1.0
        assert summary.averages["always_half"] == 0.5


class TestEvaluateCallableData:
    """Test that data can be a callable."""

    def test_callable_data(self, exporter):
        def get_data():
            return [{"input": "hello", "expected": "hello"}]

        summary = evaluate(
            name="callable-data",
            data=get_data,
            task=echo_task,
            scores=[AlwaysOneScorer()],
        )

        assert summary.total == 1
        assert summary.results[0].output == "hello"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestEvaluateErrorHandling:
    """Test error handling in evaluate()."""

    def test_task_failure_counted_as_error(self, exporter):
        data = [
            {"input": "a"},
            {"input": "b"},
        ]

        summary = evaluate(
            name="fail-eval",
            data=data,
            task=failing_task,
            scores=[AlwaysOneScorer()],
        )

        assert summary.total == 2
        assert summary.errors == 2
        for result in summary.results:
            assert result.error is not None
            assert "task failed" in result.error
            assert result.output is None
            # Scorers should not run when task fails (continue)
            assert result.scores == {}

    def test_scorer_failure_does_not_crash(self, exporter):
        data = [{"input": "a", "expected": "a"}]

        summary = evaluate(
            name="scorer-fail",
            data=data,
            task=echo_task,
            scores=[AlwaysOneScorer(), FailingScorer()],
        )

        assert summary.errors == 0
        # The successful scorer still runs
        assert "always_one" in summary.results[0].scores
        # The failing scorer is not in scores (exception was caught)
        assert "failing_scorer" not in summary.results[0].scores

    def test_mixed_task_and_scorer_failures(self, exporter):
        data = [
            {"input": "good", "expected": "good"},
            {"input": "bad"},  # will fail because failing_task
        ]

        def mixed_task(input_val):
            if input_val == "bad":
                raise ValueError("bad input")
            return input_val

        summary = evaluate(
            name="mixed-fail",
            data=data,
            task=mixed_task,
            scores=[AlwaysOneScorer()],
        )

        assert summary.total == 2
        assert summary.errors == 1
        assert summary.results[0].error is None
        assert summary.results[0].scores["always_one"].value == 1.0
        assert summary.results[1].error is not None


# ---------------------------------------------------------------------------
# Span tree structure tests
# ---------------------------------------------------------------------------


class TestEvaluateSpanTree:
    """Verify the OTEL span tree created by evaluate()."""

    def test_span_tree_structure(self, exporter):
        data = [{"input": "hello", "expected": "hello"}]

        evaluate(
            name="tree-eval",
            data=data,
            task=echo_task,
            scores=[AlwaysOneScorer()],
            emit_scores=False,  # Simplify span tree
        )

        spans = exporter.get_finished_spans()
        span_names = [s.name for s in spans]

        # Expected structure:
        #   evaluate (root)
        #     eval_item (per data point)
        #       eval_task (task execution)
        #       eval_score.always_one (scorer)
        assert "evaluate" in span_names
        assert "eval_item" in span_names
        assert "eval_task" in span_names
        assert "eval_score.always_one" in span_names

    def test_root_span_attributes(self, exporter):
        data = [
            {"input": "a"},
            {"input": "b"},
        ]

        evaluate(
            name="root-attrs",
            data=data,
            task=echo_task,
            scores=[AlwaysOneScorer(), AlwaysHalfScorer()],
            emit_scores=False,
        )

        spans = exporter.get_finished_spans()
        root = next(s for s in spans if s.name == "evaluate")

        assert root.attributes["eval.name"] == "root-attrs"
        assert root.attributes["eval.dataset_size"] == 2
        assert root.attributes["eval.scorer_count"] == 2
        assert root.attributes["eval.errors"] == 0
        assert root.attributes["eval.avg.always_one"] == 1.0
        assert root.attributes["eval.avg.always_half"] == 0.5

    def test_item_span_attributes(self, exporter):
        data = [{"input": "test_input"}]

        evaluate(
            name="item-attrs",
            data=data,
            task=echo_task,
            scores=[AlwaysOneScorer()],
            emit_scores=False,
        )

        spans = exporter.get_finished_spans()
        item_span = next(s for s in spans if s.name == "eval_item")

        assert item_span.attributes["eval.item.index"] == 0
        assert item_span.attributes["eval.item.input"] == "test_input"

    def test_task_span_captures_output(self, exporter):
        data = [{"input": "hello"}]

        evaluate(
            name="task-output",
            data=data,
            task=uppercase_task,
            scores=[AlwaysOneScorer()],
            emit_scores=False,
        )

        spans = exporter.get_finished_spans()
        task_span = next(s for s in spans if s.name == "eval_task")
        assert task_span.attributes["eval.task.output"] == "HELLO"

    def test_score_span_attributes(self, exporter):
        data = [{"input": "hello"}]

        evaluate(
            name="score-attrs",
            data=data,
            task=echo_task,
            scores=[LabelScorer()],
            emit_scores=False,
        )

        spans = exporter.get_finished_spans()
        score_span = next(s for s in spans if s.name == "eval_score.label_scorer")

        assert score_span.attributes["eval.scorer.name"] == "label_scorer"
        assert score_span.attributes["eval.score.value"] == 0.8
        assert score_span.attributes["eval.score.label"] == "good"
        assert score_span.attributes["eval.score.rationale"] == "Looks good to me."

    def test_parent_child_relationships(self, exporter):
        data = [{"input": "hello"}]

        evaluate(
            name="parent-child",
            data=data,
            task=echo_task,
            scores=[AlwaysOneScorer()],
            emit_scores=False,
        )

        spans = exporter.get_finished_spans()
        by_name = {s.name: s for s in spans}

        root = by_name["evaluate"]
        item = by_name["eval_item"]
        task_s = by_name["eval_task"]
        score_s = by_name["eval_score.always_one"]

        # All in the same trace
        trace_id = root.context.trace_id
        assert item.context.trace_id == trace_id
        assert task_s.context.trace_id == trace_id
        assert score_s.context.trace_id == trace_id

        # eval_item -> parent is evaluate
        assert item.parent.span_id == root.context.span_id

        # eval_task -> parent is eval_item
        assert task_s.parent.span_id == item.context.span_id

        # eval_score.* -> parent is eval_item
        assert score_s.parent.span_id == item.context.span_id

    def test_task_failure_span_status(self, exporter):
        data = [{"input": "x"}]

        evaluate(
            name="fail-status",
            data=data,
            task=failing_task,
            scores=[AlwaysOneScorer()],
            emit_scores=False,
        )

        spans = exporter.get_finished_spans()
        task_span = next(s for s in spans if s.name == "eval_task")
        assert task_span.status.status_code == StatusCode.ERROR
        assert "task failed" in task_span.status.description

    def test_scorer_failure_span_status(self, exporter):
        data = [{"input": "x"}]

        evaluate(
            name="scorer-fail-status",
            data=data,
            task=echo_task,
            scores=[FailingScorer()],
            emit_scores=False,
        )

        spans = exporter.get_finished_spans()
        score_span = next(s for s in spans if s.name == "eval_score.failing_scorer")
        assert score_span.status.status_code == StatusCode.ERROR

    def test_multiple_items_produce_multiple_item_spans(self, exporter):
        data = [
            {"input": "a"},
            {"input": "b"},
            {"input": "c"},
        ]

        evaluate(
            name="multi-item",
            data=data,
            task=echo_task,
            scores=[AlwaysOneScorer()],
            emit_scores=False,
        )

        spans = exporter.get_finished_spans()
        item_spans = [s for s in spans if s.name == "eval_item"]
        assert len(item_spans) == 3

        indices = sorted(s.attributes["eval.item.index"] for s in item_spans)
        assert indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# Score emission tests
# ---------------------------------------------------------------------------


class TestEvaluateScoreEmission:
    """Verify that scores are emitted as separate OTEL spans when emit_scores=True."""

    def test_emit_scores_creates_score_spans(self, exporter):
        data = [{"input": "hello", "expected": "hello"}]

        evaluate(
            name="emit-test",
            data=data,
            task=echo_task,
            scores=[AlwaysOneScorer()],
            emit_scores=True,
        )

        spans = exporter.get_finished_spans()
        score_spans = [s for s in spans if s.name.startswith("score.")]
        assert len(score_spans) == 1
        span = score_spans[0]
        assert span.attributes["opensearch.score"] is True
        assert span.attributes["score.name"] == "always_one"
        assert span.attributes["score.value"] == 1.0
        assert span.attributes["score.source"] == "eval"

    def test_emit_scores_false_no_score_spans(self, exporter):
        data = [{"input": "hello"}]

        evaluate(
            name="no-emit",
            data=data,
            task=echo_task,
            scores=[AlwaysOneScorer()],
            emit_scores=False,
        )

        spans = exporter.get_finished_spans()
        score_spans = [s for s in spans if s.name.startswith("score.")]
        assert len(score_spans) == 0

    def test_emit_scores_includes_eval_metadata(self, exporter):
        data = [{"input": "hello"}]

        evaluate(
            name="meta-emit",
            data=data,
            task=echo_task,
            scores=[AlwaysOneScorer()],
            emit_scores=True,
        )

        spans = exporter.get_finished_spans()
        score_spans = [s for s in spans if s.name.startswith("score.")]
        span = score_spans[0]
        assert span.attributes["score.metadata.eval_name"] == "meta-emit"
        assert span.attributes["score.metadata.item_index"] == "0"

    def test_multiple_scorers_emit_multiple_score_spans(self, exporter):
        data = [{"input": "hello"}]

        evaluate(
            name="multi-emit",
            data=data,
            task=echo_task,
            scores=[AlwaysOneScorer(), AlwaysHalfScorer()],
            emit_scores=True,
        )

        spans = exporter.get_finished_spans()
        score_spans = [s for s in spans if s.name.startswith("score.")]
        assert len(score_spans) == 2
        names = {s.attributes["score.name"] for s in score_spans}
        assert names == {"always_one", "always_half"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEvaluateEdgeCases:
    """Test edge cases for evaluate()."""

    def test_empty_dataset(self, exporter):
        summary = evaluate(
            name="empty",
            data=[],
            task=echo_task,
            scores=[AlwaysOneScorer()],
        )

        assert summary.total == 0
        assert summary.errors == 0
        assert summary.results == []
        assert summary.averages == {}

    def test_no_expected_field(self, exporter):
        data = [{"input": "hello"}]

        summary = evaluate(
            name="no-expected",
            data=data,
            task=echo_task,
            scores=[AlwaysOneScorer()],
        )

        assert summary.total == 1
        assert summary.results[0].expected is None

    def test_data_without_input_key(self, exporter):
        """If datum doesn't have 'input' key, the whole datum is used as input."""
        data = [{"query": "hello"}]

        summary = evaluate(
            name="no-input-key",
            data=data,
            task=lambda x: str(x),
            scores=[AlwaysOneScorer()],
        )

        assert summary.total == 1
        # datum.get("input", datum) returns the full dict
        assert summary.results[0].input == {"query": "hello"}

    def test_scorer_name_fallback_to_class_name(self, exporter):
        """When a scorer has no .name attribute, type(scorer).__name__ is used."""

        class NoNameScorer:
            def __call__(self, *, input, output, expected=None, **kwargs):
                return Score(name="anon", value=0.5)

        data = [{"input": "test"}]
        summary = evaluate(
            name="fallback-name",
            data=data,
            task=echo_task,
            scores=[NoNameScorer()],
            emit_scores=False,
        )

        spans = exporter.get_finished_spans()
        score_span = next(s for s in spans if s.name.startswith("eval_score."))
        assert score_span.name == "eval_score.NoNameScorer"
