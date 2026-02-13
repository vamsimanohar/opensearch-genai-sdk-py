"""Scorer protocol and Score dataclass.

Defines the interface that any scorer must implement to work with
evaluate(). Compatible with autoevals and phoenix-evals scorers
via adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@dataclass
class Score:
    """Result from a scorer evaluation.

    Attributes:
        name: The scorer name (e.g., "Factuality", "Relevance").
        value: Numeric score, typically 0.0 to 1.0.
        label: Optional categorical label (e.g., "A", "relevant").
        rationale: Optional explanation from the scorer.
        metadata: Optional additional data from the scorer.
    """

    name: str
    value: Optional[float] = None
    label: Optional[str] = None
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Scorer(Protocol):
    """Protocol for evaluation scorers.

    Any callable with a `name` attribute that accepts input/output/expected
    keyword arguments and returns a Score-like object is a valid scorer.

    Compatible with:
        - autoevals scorers (Factuality, Levenshtein, etc.)
        - phoenix-evals evaluators
        - Custom scorers

    Example:
        class MyScorer:
            name = "my_check"

            def __call__(self, *, input, output, expected=None, **kwargs):
                passed = "yes" in output.lower()
                return Score(name="my_check", value=1.0 if passed else 0.0)
    """

    name: str

    def __call__(
        self,
        *,
        input: str,
        output: str,
        expected: Optional[str] = None,
        **kwargs,
    ) -> Any: ...


def adapt_score(scorer_name: str, result: Any) -> Score:
    """Convert a scorer result to our Score dataclass.

    Handles results from autoevals, phoenix-evals, and raw dicts/floats.
    """
    if isinstance(result, Score):
        return result

    # autoevals returns objects with .score, .metadata, .name attributes
    if hasattr(result, "score"):
        return Score(
            name=scorer_name,
            value=result.score,
            label=getattr(result, "choice", None) or getattr(result, "label", None),
            rationale=getattr(result, "rationale", None),
            metadata=getattr(result, "metadata", None) or {},
        )

    # phoenix-evals returns Score objects with .label, .score, .explanation
    if hasattr(result, "label") and hasattr(result, "explanation"):
        return Score(
            name=scorer_name,
            value=getattr(result, "score", None),
            label=result.label,
            rationale=result.explanation,
        )

    # Plain float
    if isinstance(result, (int, float)):
        return Score(name=scorer_name, value=float(result))

    # Dict
    if isinstance(result, dict):
        return Score(
            name=scorer_name,
            value=result.get("value") or result.get("score"),
            label=result.get("label"),
            rationale=result.get("rationale") or result.get("explanation"),
            metadata={k: v for k, v in result.items() if k not in ("value", "score", "label", "rationale", "explanation")},
        )

    # Fallback
    return Score(name=scorer_name, value=None, metadata={"raw": str(result)})
