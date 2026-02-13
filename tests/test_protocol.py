"""Tests for opensearch_genai_sdk.evals.protocol.

Covers adapt_score() with different input types: native Score objects,
autoevals-like objects, phoenix-like objects, plain floats, dicts, and
the fallback path.  Also tests the Score dataclass and Scorer protocol.
"""

import pytest

from opensearch_genai_sdk.evals.protocol import Score, Scorer, adapt_score


# ---------------------------------------------------------------------------
# Score dataclass tests
# ---------------------------------------------------------------------------


class TestScoreDataclass:
    """Test the Score dataclass itself."""

    def test_required_fields(self):
        s = Score(name="test")
        assert s.name == "test"
        assert s.value is None
        assert s.label is None
        assert s.rationale is None
        assert s.metadata == {}

    def test_all_fields(self):
        s = Score(
            name="relevance",
            value=0.95,
            label="high",
            rationale="Very relevant answer",
            metadata={"model": "gpt-4"},
        )
        assert s.name == "relevance"
        assert s.value == 0.95
        assert s.label == "high"
        assert s.rationale == "Very relevant answer"
        assert s.metadata == {"model": "gpt-4"}

    def test_metadata_defaults_to_empty_dict(self):
        s1 = Score(name="a")
        s2 = Score(name="b")
        # Ensure no shared mutable default
        s1.metadata["key"] = "val"
        assert "key" not in s2.metadata


# ---------------------------------------------------------------------------
# Scorer protocol tests
# ---------------------------------------------------------------------------


class TestScorerProtocol:
    """Test that the Scorer protocol can be satisfied by custom classes."""

    def test_valid_scorer(self):
        class MyScorer:
            name = "my_scorer"

            def __call__(self, *, input, output, expected=None, **kwargs):
                return Score(name="my_scorer", value=1.0)

        scorer = MyScorer()
        assert isinstance(scorer, Scorer)

    def test_scorer_without_name_is_not_scorer(self):
        class NoName:
            def __call__(self, *, input, output, expected=None, **kwargs):
                return Score(name="x", value=1.0)

        assert not isinstance(NoName(), Scorer)


# ---------------------------------------------------------------------------
# adapt_score() with native Score
# ---------------------------------------------------------------------------


class TestAdaptScoreNative:
    """Test adapt_score when given a native Score object."""

    def test_returns_same_score(self):
        original = Score(name="test", value=0.8, label="good")
        result = adapt_score("test", original)
        assert result is original

    def test_preserves_all_fields(self):
        original = Score(
            name="test",
            value=0.5,
            label="mid",
            rationale="half",
            metadata={"k": "v"},
        )
        result = adapt_score("test", original)
        assert result.value == 0.5
        assert result.label == "mid"
        assert result.rationale == "half"
        assert result.metadata == {"k": "v"}


# ---------------------------------------------------------------------------
# adapt_score() with autoevals-like objects
# ---------------------------------------------------------------------------


class TestAdaptScoreAutoevals:
    """Test adapt_score with autoevals-style scorer results.

    autoevals returns objects with .score, .metadata, .name, .choice, .rationale.
    """

    def test_basic_autoevals_result(self):
        class AutoevalsResult:
            score = 0.75
            metadata = {"model": "gpt-4"}
            choice = "B"
            rationale = "Better answer"

        result = adapt_score("Factuality", AutoevalsResult())
        assert isinstance(result, Score)
        assert result.name == "Factuality"
        assert result.value == 0.75
        assert result.label == "B"
        assert result.rationale == "Better answer"
        assert result.metadata == {"model": "gpt-4"}

    def test_autoevals_without_choice_uses_label(self):
        class AutoevalsResult:
            score = 0.9
            label = "A"
            metadata = {}

        result = adapt_score("test", AutoevalsResult())
        assert result.label == "A"

    def test_autoevals_no_metadata(self):
        class AutoevalsResult:
            score = 0.5
            metadata = None

        result = adapt_score("test", AutoevalsResult())
        assert result.metadata == {}

    def test_autoevals_zero_score(self):
        class AutoevalsResult:
            score = 0
            metadata = {}

        result = adapt_score("test", AutoevalsResult())
        assert result.value == 0

    def test_autoevals_none_score(self):
        class AutoevalsResult:
            score = None
            metadata = {}

        result = adapt_score("test", AutoevalsResult())
        assert result.value is None

    def test_autoevals_with_choice_and_label_prefers_choice(self):
        """When both .choice and .label exist, choice is preferred."""

        class AutoevalsResult:
            score = 0.6
            choice = "C"
            label = "fallback"
            metadata = {}

        result = adapt_score("test", AutoevalsResult())
        assert result.label == "C"


# ---------------------------------------------------------------------------
# adapt_score() with phoenix-like objects
# ---------------------------------------------------------------------------


class TestAdaptScorePhoenix:
    """Test adapt_score with phoenix-evals-style scorer results.

    phoenix-evals returns objects with .label, .explanation, and optionally .score.
    When a phoenix object has a .score attribute, it is matched by the
    autoevals branch first (which checks for hasattr(result, "score")).
    The phoenix-specific branch (label + explanation) is only reached
    when .score is absent.
    """

    def test_phoenix_with_score_goes_through_autoevals_branch(self):
        """A phoenix result that has .score is handled by the autoevals branch.

        The autoevals branch reads .rationale (not .explanation), so the
        explanation is not captured as rationale.
        """

        class PhoenixResult:
            label = "relevant"
            explanation = "The answer is relevant to the query."
            score = 0.9

        result = adapt_score("Relevance", PhoenixResult())
        assert isinstance(result, Score)
        assert result.name == "Relevance"
        assert result.value == 0.9
        assert result.label == "relevant"
        # The autoevals branch reads .rationale, not .explanation
        assert result.rationale is None

    def test_phoenix_without_score_uses_phoenix_branch(self):
        """A phoenix result without .score goes through the phoenix branch."""

        class PhoenixResult:
            label = "toxic"
            explanation = "Contains harmful language."

        result = adapt_score("Toxicity", PhoenixResult())
        assert result.value is None
        assert result.label == "toxic"
        assert result.rationale == "Contains harmful language."

    def test_phoenix_empty_explanation(self):
        class PhoenixResult:
            label = "good"
            explanation = ""

        result = adapt_score("Quality", PhoenixResult())
        assert result.label == "good"
        assert result.rationale == ""

    def test_phoenix_with_score_and_rationale(self):
        """A phoenix-like object that also has .rationale is captured correctly
        through the autoevals branch."""

        class PhoenixResultWithRationale:
            score = 0.85
            label = "good"
            explanation = "some explanation"
            rationale = "The answer is well-supported."

        result = adapt_score("Quality", PhoenixResultWithRationale())
        assert result.value == 0.85
        assert result.rationale == "The answer is well-supported."


# ---------------------------------------------------------------------------
# adapt_score() with plain floats
# ---------------------------------------------------------------------------


class TestAdaptScoreFloat:
    """Test adapt_score with plain numeric values."""

    def test_float_value(self):
        result = adapt_score("metric", 0.85)
        assert isinstance(result, Score)
        assert result.name == "metric"
        assert result.value == 0.85

    def test_int_value(self):
        result = adapt_score("metric", 1)
        assert isinstance(result, Score)
        assert result.value == 1.0
        assert isinstance(result.value, float)

    def test_zero(self):
        result = adapt_score("metric", 0)
        assert result.value == 0.0

    def test_negative_float(self):
        result = adapt_score("metric", -0.5)
        assert result.value == -0.5


# ---------------------------------------------------------------------------
# adapt_score() with dicts
# ---------------------------------------------------------------------------


class TestAdaptScoreDict:
    """Test adapt_score with dict results."""

    def test_dict_with_value(self):
        result = adapt_score("test", {"value": 0.8, "label": "good"})
        assert result.name == "test"
        assert result.value == 0.8
        assert result.label == "good"

    def test_dict_with_score_key(self):
        result = adapt_score("test", {"score": 0.7})
        assert result.value == 0.7

    def test_dict_with_rationale(self):
        result = adapt_score("test", {"value": 0.5, "rationale": "some reason"})
        assert result.rationale == "some reason"

    def test_dict_with_explanation(self):
        result = adapt_score("test", {"value": 0.5, "explanation": "some reason"})
        assert result.rationale == "some reason"

    def test_dict_extra_keys_become_metadata(self):
        result = adapt_score(
            "test",
            {"value": 0.5, "label": "ok", "model": "gpt-4", "tokens": 100},
        )
        assert result.metadata == {"model": "gpt-4", "tokens": 100}

    def test_dict_no_value_no_score(self):
        result = adapt_score("test", {"label": "unknown"})
        assert result.value is None
        assert result.label == "unknown"

    def test_empty_dict(self):
        result = adapt_score("test", {})
        assert result.value is None
        assert result.label is None
        assert result.rationale is None
        assert result.metadata == {}

    def test_dict_prefers_value_over_score(self):
        """When both 'value' and 'score' keys exist, 'value' is used."""
        result = adapt_score("test", {"value": 0.9, "score": 0.1})
        assert result.value == 0.9


# ---------------------------------------------------------------------------
# adapt_score() fallback
# ---------------------------------------------------------------------------


class TestAdaptScoreFallback:
    """Test adapt_score with unsupported types."""

    def test_string_fallback(self):
        result = adapt_score("test", "some string")
        assert isinstance(result, Score)
        assert result.name == "test"
        assert result.value is None
        assert result.metadata == {"raw": "some string"}

    def test_list_fallback(self):
        result = adapt_score("test", [1, 2, 3])
        assert result.value is None
        assert result.metadata == {"raw": "[1, 2, 3]"}

    def test_none_fallback(self):
        result = adapt_score("test", None)
        assert result.value is None
        assert result.metadata == {"raw": "None"}

    def test_bool_is_treated_as_int(self):
        # In Python, bool is a subclass of int, so True/False go through
        # the (int, float) branch.
        result = adapt_score("test", True)
        assert result.value == 1.0

        result = adapt_score("test", False)
        assert result.value == 0.0

    def test_custom_object_fallback(self):
        class Mystery:
            def __str__(self):
                return "mystery_obj"

        result = adapt_score("test", Mystery())
        assert result.value is None
        assert result.metadata == {"raw": "mystery_obj"}
