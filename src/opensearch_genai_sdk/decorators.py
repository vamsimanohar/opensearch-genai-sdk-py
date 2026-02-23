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

import functools
import inspect
import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from opentelemetry import trace

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Span kind values following OTEL GenAI semantic conventions
SPAN_KIND_WORKFLOW = "workflow"
SPAN_KIND_TASK = "task"
SPAN_KIND_AGENT = "invoke_agent"
SPAN_KIND_TOOL = "execute_tool"

_TRACER_NAME = "opensearch-genai-sdk"


def workflow(
    name: str | None = None,
    version: int | None = None,
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
    name: str | None = None,
    version: int | None = None,
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
    name: str | None = None,
    version: int | None = None,
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
    name: str | None = None,
    version: int | None = None,
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
    name: str | None,
    version: int | None,
    span_kind: str,
) -> Callable[[F], F]:
    """Create a decorator that wraps a function in an OTEL span."""

    def decorator(fn: F) -> F:
        entity_name = name or fn.__qualname__
        # Agent/tool span names follow semconv: "{operation} {name}"
        if span_kind in (SPAN_KIND_AGENT, SPAN_KIND_TOOL):
            span_name = f"{span_kind} {entity_name}"
        else:
            span_name = entity_name
        fn_doc = fn.__doc__
        tracer = trace.get_tracer(_TRACER_NAME)
        sig = inspect.signature(fn)

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(span_name) as span:
                    _set_span_attributes(
                        span, span_kind, entity_name, version, sig, args, kwargs, fn_doc
                    )
                    try:
                        result = await fn(*args, **kwargs)
                        _set_output(span, span_kind, result)
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
                    _set_span_attributes(
                        span, span_kind, entity_name, version, sig, args, kwargs, fn_doc
                    )
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
                    _set_span_attributes(
                        span, span_kind, entity_name, version, sig, args, kwargs, fn_doc
                    )
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
                    _set_span_attributes(
                        span, span_kind, entity_name, version, sig, args, kwargs, fn_doc
                    )
                    try:
                        result = fn(*args, **kwargs)
                        _set_output(span, span_kind, result)
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
    entity_name: str,
    version: int | None,
    sig: inspect.Signature,
    args: tuple,
    kwargs: dict,
    fn_doc: str | None = None,
) -> None:
    """Set standard attributes on a span."""
    span.set_attribute("gen_ai.operation.name", span_kind)

    # Use type-specific name attributes matching gen_ai semantic conventions
    _NAME_ATTR = {
        SPAN_KIND_WORKFLOW: "gen_ai.workflow.name",
        SPAN_KIND_TASK: "gen_ai.task.name",
        SPAN_KIND_AGENT: "gen_ai.agent.name",
        SPAN_KIND_TOOL: "gen_ai.tool.name",
    }
    span.set_attribute(_NAME_ATTR.get(span_kind, "gen_ai.entity.name"), entity_name)

    if version is not None:
        if span_kind == SPAN_KIND_AGENT:
            span.set_attribute("gen_ai.agent.version", str(version))
        else:
            span.set_attribute("gen_ai.entity.version", version)

    # Tool-specific attributes from semconv
    if span_kind == SPAN_KIND_TOOL:
        span.set_attribute("gen_ai.tool.type", "function")
        if fn_doc:
            span.set_attribute("gen_ai.tool.description", fn_doc)

    # Capture input (best-effort, don't fail if serialization fails)
    _set_input(span, span_kind, sig, args, kwargs)


def _set_input(
    span: trace.Span, span_kind: str, sig: inspect.Signature, args: tuple, kwargs: dict
) -> None:
    """Attempt to capture function input as a span attribute.

    Binds positional and keyword arguments to their parameter names
    so the trace shows {"city": "Paris"} instead of just "Paris".
    """
    try:
        if not args and not kwargs:
            return

        # Bind args to parameter names for readable output
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            value = dict(bound.arguments)
        except TypeError:
            # Fallback if binding fails (e.g., *args/**kwargs signatures)
            value = {"args": list(args), "kwargs": kwargs}

        serialized = json.dumps(value, default=str)
        # Truncate to avoid oversized attributes
        if len(serialized) > 10_000:
            serialized = serialized[:10_000] + "...(truncated)"

        # Tool spans use semconv attribute name
        attr_key = (
            "gen_ai.tool.call.arguments" if span_kind == SPAN_KIND_TOOL else "gen_ai.entity.input"
        )
        span.set_attribute(attr_key, serialized)
    except Exception:
        pass


def _set_output(span: trace.Span, span_kind: str, result: Any) -> None:
    """Attempt to capture function output as a span attribute."""
    try:
        if result is None:
            return
        serialized = json.dumps(result, default=str)
        if len(serialized) > 10_000:
            serialized = serialized[:10_000] + "...(truncated)"

        # Tool spans use semconv attribute name
        attr_key = (
            "gen_ai.tool.call.result" if span_kind == SPAN_KIND_TOOL else "gen_ai.entity.output"
        )
        span.set_attribute(attr_key, serialized)
    except Exception:
        pass
