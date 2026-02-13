"""Tests for opensearch_genai_sdk.decorators.

Covers @workflow, @task, @agent, @tool decorators for sync functions,
async functions, generators, and async generators.  Verifies span
names, semantic attributes, parent-child relationships, error handling,
and input/output capture.
"""

import asyncio
import json

import pytest
from opentelemetry.trace import StatusCode

from opensearch_genai_sdk.decorators import workflow, task, agent, tool


# ---------------------------------------------------------------------------
# Helper functions decorated by the SDK
# ---------------------------------------------------------------------------

@workflow(name="my_workflow")
def sync_workflow(x: int) -> int:
    return x + 1


@task(name="my_task")
def sync_task(x: int) -> int:
    return x * 2


@agent(name="my_agent")
def sync_agent(query: str) -> str:
    return f"answer to {query}"


@tool(name="my_tool")
def sync_tool(a: int, b: int) -> int:
    return a + b


@workflow(name="async_workflow")
async def async_workflow_fn(x: int) -> int:
    return x + 10


@task(name="async_task")
async def async_task_fn(x: int) -> int:
    return x * 10


@agent(name="async_agent")
async def async_agent_fn(query: str) -> str:
    return f"async answer to {query}"


@tool(name="async_tool")
async def async_tool_fn(a: int, b: int) -> int:
    return a + b


@workflow(name="error_workflow")
def error_workflow_fn():
    raise ValueError("something went wrong")


@workflow(name="error_async_workflow")
async def async_error_workflow_fn():
    raise RuntimeError("async boom")


@workflow()
def auto_name_workflow():
    return "auto"


@workflow(name="versioned_workflow", version=3)
def versioned_workflow_fn():
    return "v3"


@workflow(name="parent_workflow")
def parent_workflow_fn():
    return child_task_fn("hello")


@task(name="child_task")
def child_task_fn(msg: str) -> str:
    return msg.upper()


@workflow(name="kwargs_workflow")
def kwargs_workflow_fn(*, key: str, value: int) -> dict:
    return {"key": key, "value": value}


@workflow(name="mixed_args_workflow")
def mixed_args_workflow_fn(a: int, b: int, *, flag: bool = False) -> str:
    return f"{a}+{b} flag={flag}"


@tool(name="gen_tool")
def generator_tool_fn(n: int):
    for i in range(n):
        yield i


@tool(name="async_gen_tool")
async def async_generator_tool_fn(n: int):
    for i in range(n):
        yield i


@tool(name="gen_error_tool")
def generator_error_tool_fn():
    yield 1
    raise ValueError("gen error")


@tool(name="async_gen_error_tool")
async def async_generator_error_tool_fn():
    yield 1
    raise ValueError("async gen error")


# ---------------------------------------------------------------------------
# Sync decorator tests
# ---------------------------------------------------------------------------


class TestSyncDecorators:
    """Test sync function decorators."""

    def test_workflow_creates_span(self, exporter):
        result = sync_workflow(5)
        assert result == 6

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "my_workflow"
        assert span.attributes["gen_ai.operation.name"] == "workflow"

    def test_task_creates_span(self, exporter):
        result = sync_task(5)
        assert result == 10

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "my_task"
        assert span.attributes["gen_ai.operation.name"] == "task"

    def test_agent_creates_span(self, exporter):
        result = sync_agent("test")
        assert result == "answer to test"

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "my_agent"
        assert span.attributes["gen_ai.operation.name"] == "agent"

    def test_tool_creates_span(self, exporter):
        result = sync_tool(3, 4)
        assert result == 7

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "my_tool"
        assert span.attributes["gen_ai.operation.name"] == "tool"

    def test_input_capture_single_arg(self, exporter):
        sync_workflow(42)
        spans = exporter.get_finished_spans()
        span = spans[0]
        # Single positional arg is captured directly
        assert json.loads(span.attributes["gen_ai.entity.input"]) == 42

    def test_input_capture_multiple_args(self, exporter):
        sync_tool(3, 4)
        spans = exporter.get_finished_spans()
        span = spans[0]
        captured = json.loads(span.attributes["gen_ai.entity.input"])
        assert captured == [3, 4]

    def test_input_capture_kwargs_only(self, exporter):
        kwargs_workflow_fn(key="x", value=99)
        spans = exporter.get_finished_spans()
        span = spans[0]
        captured = json.loads(span.attributes["gen_ai.entity.input"])
        assert captured == {"key": "x", "value": 99}

    def test_input_capture_mixed_args_kwargs(self, exporter):
        mixed_args_workflow_fn(1, 2, flag=True)
        spans = exporter.get_finished_spans()
        span = spans[0]
        captured = json.loads(span.attributes["gen_ai.entity.input"])
        assert captured == {"args": [1, 2], "kwargs": {"flag": True}}

    def test_output_capture(self, exporter):
        result = sync_workflow(5)
        assert result == 6
        spans = exporter.get_finished_spans()
        span = spans[0]
        assert json.loads(span.attributes["gen_ai.entity.output"]) == 6

    def test_output_capture_dict(self, exporter):
        result = kwargs_workflow_fn(key="x", value=99)
        assert result == {"key": "x", "value": 99}
        spans = exporter.get_finished_spans()
        span = spans[0]
        captured = json.loads(span.attributes["gen_ai.entity.output"])
        assert captured == {"key": "x", "value": 99}

    def test_error_sets_span_status(self, exporter):
        with pytest.raises(ValueError, match="something went wrong"):
            error_workflow_fn()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert "something went wrong" in span.status.description

    def test_error_records_exception_event(self, exporter):
        with pytest.raises(ValueError):
            error_workflow_fn()

        spans = exporter.get_finished_spans()
        span = spans[0]
        events = span.events
        exc_events = [e for e in events if e.name == "exception"]
        assert len(exc_events) >= 1
        assert "ValueError" in exc_events[0].attributes["exception.type"]

    def test_auto_name_uses_qualname(self, exporter):
        auto_name_workflow()
        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.name == "auto_name_workflow"

    def test_version_attribute(self, exporter):
        versioned_workflow_fn()
        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.attributes["gen_ai.entity.version"] == 3

    def test_no_version_attribute_when_not_set(self, exporter):
        sync_workflow(1)
        spans = exporter.get_finished_spans()
        span = spans[0]
        assert "gen_ai.entity.version" not in span.attributes


class TestParentChildRelationships:
    """Test that nested decorated functions produce correct parent-child spans."""

    def test_nested_workflow_task(self, exporter):
        result = parent_workflow_fn()
        assert result == "HELLO"

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        # Spans are finished in order: child first, then parent
        child_span = next(s for s in spans if s.name == "child_task")
        parent_span = next(s for s in spans if s.name == "parent_workflow")

        # Both belong to the same trace
        assert (
            child_span.context.trace_id
            == parent_span.context.trace_id
        )

        # Child's parent is the workflow span
        assert child_span.parent.span_id == parent_span.context.span_id


# ---------------------------------------------------------------------------
# Async decorator tests
# ---------------------------------------------------------------------------


class TestAsyncDecorators:
    """Test async function decorators."""

    @pytest.mark.asyncio
    async def test_async_workflow(self, exporter):
        result = await async_workflow_fn(5)
        assert result == 15

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "async_workflow"
        assert span.attributes["gen_ai.operation.name"] == "workflow"

    @pytest.mark.asyncio
    async def test_async_task(self, exporter):
        result = await async_task_fn(3)
        assert result == 30

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "async_task"
        assert span.attributes["gen_ai.operation.name"] == "task"

    @pytest.mark.asyncio
    async def test_async_agent(self, exporter):
        result = await async_agent_fn("hello")
        assert result == "async answer to hello"

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "async_agent"
        assert span.attributes["gen_ai.operation.name"] == "agent"

    @pytest.mark.asyncio
    async def test_async_tool(self, exporter):
        result = await async_tool_fn(10, 20)
        assert result == 30

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "async_tool"
        assert span.attributes["gen_ai.operation.name"] == "tool"

    @pytest.mark.asyncio
    async def test_async_output_capture(self, exporter):
        result = await async_workflow_fn(7)
        assert result == 17

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert json.loads(span.attributes["gen_ai.entity.output"]) == 17

    @pytest.mark.asyncio
    async def test_async_input_capture(self, exporter):
        await async_tool_fn(10, 20)
        spans = exporter.get_finished_spans()
        span = spans[0]
        captured = json.loads(span.attributes["gen_ai.entity.input"])
        assert captured == [10, 20]

    @pytest.mark.asyncio
    async def test_async_error_sets_span_status(self, exporter):
        with pytest.raises(RuntimeError, match="async boom"):
            await async_error_workflow_fn()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert "async boom" in span.status.description

    @pytest.mark.asyncio
    async def test_async_error_records_exception(self, exporter):
        with pytest.raises(RuntimeError):
            await async_error_workflow_fn()

        spans = exporter.get_finished_spans()
        span = spans[0]
        exc_events = [e for e in span.events if e.name == "exception"]
        assert len(exc_events) >= 1
        assert "RuntimeError" in exc_events[0].attributes["exception.type"]


# ---------------------------------------------------------------------------
# Generator decorator tests
# ---------------------------------------------------------------------------


class TestGeneratorDecorators:
    """Test generator and async generator function decorators."""

    def test_sync_generator(self, exporter):
        items = list(generator_tool_fn(3))
        assert items == [0, 1, 2]

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "gen_tool"
        assert span.attributes["gen_ai.operation.name"] == "tool"

    @pytest.mark.asyncio
    async def test_async_generator(self, exporter):
        items = []
        async for item in async_generator_tool_fn(3):
            items.append(item)
        assert items == [0, 1, 2]

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "async_gen_tool"
        assert span.attributes["gen_ai.operation.name"] == "tool"

    def test_sync_generator_error(self, exporter):
        gen = generator_error_tool_fn()
        assert next(gen) == 1
        with pytest.raises(ValueError, match="gen error"):
            next(gen)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_async_generator_error(self, exporter):
        agen = async_generator_error_tool_fn()
        first = await agen.__anext__()
        assert first == 1
        with pytest.raises(ValueError, match="async gen error"):
            await agen.__anext__()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# functools.wraps preservation
# ---------------------------------------------------------------------------


class TestFunctoolsWraps:
    """Verify that decorators preserve the original function metadata."""

    def test_sync_preserves_name(self):
        assert sync_workflow.__name__ == "sync_workflow"

    def test_async_preserves_name(self):
        assert async_workflow_fn.__name__ == "async_workflow_fn"

    def test_sync_preserves_module(self):
        assert sync_workflow.__module__ == __name__

    def test_generator_preserves_name(self):
        assert generator_tool_fn.__name__ == "generator_tool_fn"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases for decorators."""

    def test_none_return_no_output_attribute(self, exporter):
        @workflow(name="none_return")
        def returns_none():
            return None

        returns_none()
        spans = exporter.get_finished_spans()
        span = spans[0]
        # _set_output skips None results
        assert "gen_ai.entity.output" not in span.attributes

    def test_no_args_no_input_attribute(self, exporter):
        @workflow(name="no_args")
        def no_args_fn():
            return 42

        no_args_fn()
        spans = exporter.get_finished_spans()
        span = spans[0]
        # _set_input skips when no args and no kwargs
        assert "gen_ai.entity.input" not in span.attributes

    def test_large_input_is_truncated(self, exporter):
        @workflow(name="big_input")
        def big_input_fn(data: str) -> str:
            return "ok"

        big_data = "x" * 20_000
        big_input_fn(big_data)
        spans = exporter.get_finished_spans()
        span = spans[0]
        captured = span.attributes["gen_ai.entity.input"]
        assert len(captured) <= 10_100  # 10000 + truncation marker + quotes
        assert "truncated" in captured

    def test_large_output_is_truncated(self, exporter):
        @workflow(name="big_output")
        def big_output_fn() -> str:
            return "y" * 20_000

        big_output_fn()
        spans = exporter.get_finished_spans()
        span = spans[0]
        captured = span.attributes["gen_ai.entity.output"]
        assert len(captured) <= 10_100
        assert "truncated" in captured

    def test_non_serializable_input_does_not_crash(self, exporter):
        @workflow(name="non_serial_input")
        def non_serial_fn(obj):
            return "ok"

        # A custom object with no JSON serialization -- default=str handles it
        class Weird:
            pass

        non_serial_fn(Weird())
        spans = exporter.get_finished_spans()
        assert len(spans) == 1  # Decorator should not crash

    def test_non_serializable_output_does_not_crash(self, exporter):
        @workflow(name="non_serial_output")
        def returns_weird():
            class Weird:
                pass
            return Weird()

        returns_weird()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1  # Decorator should not crash
