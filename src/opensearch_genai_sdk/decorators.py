"""Decorators for tracing custom functions as OTEL spans.

Provides @workflow, @task, @agent, and @tool decorators that create
standard OpenTelemetry spans. These are the user-facing API for
tracing custom application logic — the gap that pure auto-instrumentors
don't cover.

All decorators produce standard OTEL spans with gen_ai semantic
convention attributes. Zero lock-in: remove the decorator and
your code still works.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
from typing import Any, Callable, Optional, TypeVar

from opentelemetry import trace

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Span kind values following OTEL GenAI semantic conventions
SPAN_KIND_WORKFLOW = "workflow"
SPAN_KIND_TASK = "task"
SPAN_KIND_AGENT = "agent"
SPAN_KIND_TOOL = "tool"

_TRACER_NAME = "opensearch-genai-sdk"


def workflow(
    name: Optional[str] = None,
    version: Optional[int] = None,
) -> Callable[[F], F]:
    """Trace a function as a workflow span.

    Use for top-level orchestration functions that coordinate
    multiple tasks, agents, or tool calls.

    Args:
        name: Span name. Defaults to the function's qualified name.
        version: Optional version number for tracking changes.

    Example:
        @workflow(name="qa_pipeline")
        def run_pipeline(query: str) -> str:
            plan = plan_steps(query)
            result = execute(plan)
            return result
    """
    return _make_decorator(name=name, version=version, span_kind=SPAN_KIND_WORKFLOW)


def task(
    name: Optional[str] = None,
    version: Optional[int] = None,
) -> Callable[[F], F]:
    """Trace a function as a task span.

    Use for individual units of work within a workflow.

    Args:
        name: Span name. Defaults to the function's qualified name.
        version: Optional version number for tracking changes.

    Example:
        @task(name="summarize")
        def summarize_text(text: str) -> str:
            return llm.generate(f"Summarize: {text}")
    """
    return _make_decorator(name=name, version=version, span_kind=SPAN_KIND_TASK)


def agent(
    name: Optional[str] = None,
    version: Optional[int] = None,
) -> Callable[[F], F]:
    """Trace a function as an agent span.

    Use for autonomous agent logic that makes decisions and
    invokes tools.

    Args:
        name: Span name. Defaults to the function's qualified name.
        version: Optional version number for tracking changes.

    Example:
        @agent(name="research_agent")
        def research(query: str) -> str:
            while not done:
                action = decide_next_action(query)
                result = execute_action(action)
            return result
    """
    return _make_decorator(name=name, version=version, span_kind=SPAN_KIND_AGENT)


def tool(
    name: Optional[str] = None,
    version: Optional[int] = None,
) -> Callable[[F], F]:
    """Trace a function as a tool span.

    Use for tool/function calls invoked by agents.

    Args:
        name: Span name. Defaults to the function's qualified name.
        version: Optional version number for tracking changes.

    Example:
        @tool(name="web_search")
        def search(query: str) -> list[dict]:
            return search_api.query(query)
    """
    return _make_decorator(name=name, version=version, span_kind=SPAN_KIND_TOOL)


def _make_decorator(
    name: Optional[str],
    version: Optional[int],
    span_kind: str,
) -> Callable[[F], F]:
    """Create a decorator that wraps a function in an OTEL span."""

    def decorator(fn: F) -> F:
        span_name = name or fn.__qualname__
        tracer = trace.get_tracer(_TRACER_NAME)

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(span_name) as span:
                    _set_span_attributes(span, span_kind, version, args, kwargs)
                    try:
                        result = await fn(*args, **kwargs)
                        _set_output(span, result)
                        return result
                    except Exception as exc:
                        span.set_status(trace.StatusCode.ERROR, str(exc))
                        span.record_exception(exc)
                        raise

            return async_wrapper  # type: ignore[return-value]

        elif inspect.isgeneratorfunction(fn):

            @functools.wraps(fn)
            def gen_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(span_name) as span:
                    _set_span_attributes(span, span_kind, version, args, kwargs)
                    try:
                        yield from fn(*args, **kwargs)
                    except Exception as exc:
                        span.set_status(trace.StatusCode.ERROR, str(exc))
                        span.record_exception(exc)
                        raise

            return gen_wrapper  # type: ignore[return-value]

        elif inspect.isasyncgenfunction(fn):

            @functools.wraps(fn)
            async def async_gen_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(span_name) as span:
                    _set_span_attributes(span, span_kind, version, args, kwargs)
                    try:
                        async for item in fn(*args, **kwargs):
                            yield item
                    except Exception as exc:
                        span.set_status(trace.StatusCode.ERROR, str(exc))
                        span.record_exception(exc)
                        raise

            return async_gen_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(span_name) as span:
                    _set_span_attributes(span, span_kind, version, args, kwargs)
                    try:
                        result = fn(*args, **kwargs)
                        _set_output(span, result)
                        return result
                    except Exception as exc:
                        span.set_status(trace.StatusCode.ERROR, str(exc))
                        span.record_exception(exc)
                        raise

            return sync_wrapper  # type: ignore[return-value]

    return decorator


def _set_span_attributes(
    span: trace.Span,
    span_kind: str,
    version: Optional[int],
    args: tuple,
    kwargs: dict,
) -> None:
    """Set standard attributes on a span."""
    span.set_attribute("gen_ai.operation.name", span_kind)

    if version is not None:
        span.set_attribute("gen_ai.entity.version", version)

    # Capture input (best-effort, don't fail if serialization fails)
    _set_input(span, args, kwargs)


def _set_input(span: trace.Span, args: tuple, kwargs: dict) -> None:
    """Attempt to capture function input as a span attribute."""
    try:
        if args and not kwargs:
            value = args[0] if len(args) == 1 else list(args)
        elif kwargs and not args:
            value = kwargs
        else:
            value = {"args": list(args), "kwargs": kwargs} if args or kwargs else None

        if value is not None:
            serialized = json.dumps(value, default=str)
            # Truncate to avoid oversized attributes
            if len(serialized) > 10_000:
                serialized = serialized[:10_000] + "...(truncated)"
            span.set_attribute("gen_ai.entity.input", serialized)
    except Exception:
        pass


def _set_output(span: trace.Span, result: Any) -> None:
    """Attempt to capture function output as a span attribute."""
    try:
        if result is None:
            return
        serialized = json.dumps(result, default=str)
        if len(serialized) > 10_000:
            serialized = serialized[:10_000] + "...(truncated)"
        span.set_attribute("gen_ai.entity.output", serialized)
    except Exception:
        pass
