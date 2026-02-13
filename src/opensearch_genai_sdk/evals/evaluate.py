"""Evaluation orchestrator for OpenSearch AI observability.

Runs a task function across a dataset, applies scorers to each output,
creates OTEL spans for the entire eval run, and stores scores in OpenSearch.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from opentelemetry import trace

from opensearch_genai_sdk.evals.protocol import Score, adapt_score
from opensearch_genai_sdk.score import score as submit_score

logger = logging.getLogger(__name__)

_TRACER_NAME = "opensearch-genai-sdk-evals"


@dataclass
class EvalResult:
    """Result of a single data point evaluation.

    Attributes:
        input: The input given to the task.
        output: The task's output.
        expected: The expected output (if provided).
        scores: Dict of scorer name → Score.
        error: Error message if the task failed.
    """

    input: Any
    output: Any = None
    expected: Any = None
    scores: Dict[str, Score] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class EvalSummary:
    """Summary of an evaluation run.

    Attributes:
        name: The evaluation name.
        results: Per-data-point results.
        averages: Average score per scorer.
        total: Number of data points.
        errors: Number of failed data points.
    """

    name: str
    results: List[EvalResult] = field(default_factory=list)
    averages: Dict[str, float] = field(default_factory=dict)
    total: int = 0
    errors: int = 0

    def __str__(self) -> str:
        parts = [f"Eval: {self.name} ({self.total} samples, {self.errors} errors)"]
        for scorer_name, avg in self.averages.items():
            parts.append(f"  {scorer_name}: {avg:.3f}")
        return "\n".join(parts)


def evaluate(
    name: str,
    data: Union[List[Dict[str, Any]], Callable[[], List[Dict[str, Any]]]],
    task: Callable,
    scores: Sequence[Any],
    *,
    emit_scores: bool = True,
) -> EvalSummary:
    """Run an evaluation: dataset -> task -> scorers -> OTEL spans.

    For each data point, runs the task function to produce output,
    then applies all scorers. Creates OTEL spans for the entire flow.
    Scores are emitted as OTEL spans through the same exporter pipeline.

    Args:
        name: Name for this evaluation run.
        data: List of dicts with "input" and optional "expected" keys,
            or a callable that returns such a list.
        task: Function that takes an input and returns output.
            Can be sync or async.
        scores: List of scorer instances. Each must be callable with
            (input=, output=, expected=) and return a Score-like result.
            Compatible with autoevals, phoenix-evals, or custom scorers.
        emit_scores: Whether to emit scores as separate OTEL spans
            for Data Prepper routing. Defaults to True.

    Returns:
        EvalSummary with per-data-point results and averages.

    Example:
        from opensearch_genai_sdk import evaluate
        from autoevals import Factuality, Levenshtein

        results = evaluate(
            name="qa-eval",
            data=[
                {"input": "Capital of France?", "expected": "Paris"},
                {"input": "2+2?", "expected": "4"},
            ],
            task=lambda input: my_llm_call(input),
            scores=[Factuality(), Levenshtein],
        )
        print(results)
        # Eval: qa-eval (2 samples, 0 errors)
        #   Factuality: 0.950
        #   Levenshtein: 1.000
    """
    tracer = trace.get_tracer(_TRACER_NAME)

    # Resolve data if callable
    dataset = data() if callable(data) and not isinstance(data, list) else data

    summary = EvalSummary(name=name, total=len(dataset))

    with tracer.start_as_current_span(
        "evaluate",
        attributes={
            "eval.name": name,
            "eval.dataset_size": len(dataset),
            "eval.scorer_count": len(scores),
        },
    ) as eval_span:

        for i, datum in enumerate(dataset):
            input_val = datum.get("input", datum)
            expected_val = datum.get("expected")

            eval_result = EvalResult(input=input_val, expected=expected_val)

            with tracer.start_as_current_span(
                "eval_item",
                attributes={
                    "eval.item.index": i,
                    "eval.item.input": str(input_val)[:1000],
                },
            ) as item_span:

                # Run the task
                with tracer.start_as_current_span("eval_task") as task_span:
                    try:
                        if inspect.iscoroutinefunction(task):
                            output = asyncio.get_event_loop().run_until_complete(
                                task(input_val)
                            )
                        else:
                            output = task(input_val)
                        eval_result.output = output
                        task_span.set_attribute("eval.task.output", str(output)[:1000])
                    except Exception as exc:
                        eval_result.error = str(exc)
                        summary.errors += 1
                        task_span.set_status(trace.StatusCode.ERROR, str(exc))
                        task_span.record_exception(exc)
                        summary.results.append(eval_result)
                        continue

                # Run each scorer
                for scorer in scores:
                    scorer_name = getattr(scorer, "name", type(scorer).__name__)

                    with tracer.start_as_current_span(
                        f"eval_score.{scorer_name}",
                        attributes={"eval.scorer.name": scorer_name},
                    ) as score_span:
                        try:
                            raw_result = scorer(
                                input=str(input_val),
                                output=str(output),
                                expected=str(expected_val) if expected_val else None,
                            )
                            score_obj = adapt_score(scorer_name, raw_result)
                            eval_result.scores[scorer_name] = score_obj

                            if score_obj.value is not None:
                                score_span.set_attribute("eval.score.value", score_obj.value)
                            if score_obj.label:
                                score_span.set_attribute("eval.score.label", score_obj.label)
                            if score_obj.rationale:
                                score_span.set_attribute(
                                    "eval.score.rationale", score_obj.rationale[:500]
                                )

                        except Exception as exc:
                            logger.warning("Scorer %s failed on item %d: %s", scorer_name, i, exc)
                            score_span.set_status(trace.StatusCode.ERROR, str(exc))
                            score_span.record_exception(exc)

                # Emit scores as OTEL spans
                if emit_scores:
                    trace_id = format(item_span.get_span_context().trace_id, "032x")
                    span_id = format(item_span.get_span_context().span_id, "016x")
                    for scorer_name, score_obj in eval_result.scores.items():
                        try:
                            submit_score(
                                name=scorer_name,
                                value=score_obj.value,
                                trace_id=trace_id,
                                span_id=span_id,
                                label=score_obj.label,
                                rationale=score_obj.rationale,
                                source="eval",
                                metadata={"eval_name": name, "item_index": i},
                            )
                        except Exception as exc:
                            logger.warning("Failed to emit score %s: %s", scorer_name, exc)

                summary.results.append(eval_result)

        # Compute averages
        score_totals: Dict[str, List[float]] = {}
        for result in summary.results:
            for scorer_name, score_obj in result.scores.items():
                if score_obj.value is not None:
                    score_totals.setdefault(scorer_name, []).append(score_obj.value)

        for scorer_name, values in score_totals.items():
            avg = sum(values) / len(values)
            summary.averages[scorer_name] = avg
            eval_span.set_attribute(f"eval.avg.{scorer_name}", avg)

        eval_span.set_attribute("eval.errors", summary.errors)

    return summary
