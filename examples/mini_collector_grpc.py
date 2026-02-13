"""Mini OTEL Collector — gRPC receiver, prints spans to terminal.

No Docker needed. Receives OTLP/gRPC on port 4317 and pretty-prints
every span.

Usage:
    # Terminal 1: start the collector
    python examples/mini_collector_grpc.py

    # Terminal 2: run the agent
    python examples/agent_grpc.py
"""

import grpc
from concurrent import futures

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
    TraceServiceServicer,
    add_TraceServiceServicer_to_server,
)

# Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

SPAN_KIND_MAP = {
    "workflow": f"{MAGENTA}workflow{RESET}",
    "agent": f"{CYAN}agent{RESET}",
    "task": f"{GREEN}task{RESET}",
    "tool": f"{YELLOW}tool{RESET}",
}


def print_span(span, resource_attrs: dict):
    """Pretty-print a single span."""
    attrs = {}
    for kv in span.attributes:
        key = kv.key
        val = kv.value
        if val.HasField("string_value"):
            attrs[key] = val.string_value
        elif val.HasField("int_value"):
            attrs[key] = val.int_value
        elif val.HasField("double_value"):
            attrs[key] = val.double_value
        elif val.HasField("bool_value"):
            attrs[key] = val.bool_value
        else:
            attrs[key] = str(val)

    span_kind = attrs.get("traceloop.span.kind", "")
    is_score = attrs.get("opensearch.score", False)
    kind_label = SPAN_KIND_MAP.get(span_kind, f"{RED}score{RESET}" if is_score else "internal")

    trace_id = span.trace_id.hex()
    span_id = span.span_id.hex()
    parent_id = span.parent_span_id.hex() if span.parent_span_id else "none (root)"
    duration_ms = (span.end_time_unix_nano - span.start_time_unix_nano) / 1_000_000
    service = resource_attrs.get("service.name", "unknown")

    print(f"\n{BOLD}{'─' * 70}{RESET}")
    print(f"  {BOLD}Span:{RESET} {span.name}  [{kind_label}]  {DIM}{duration_ms:.1f}ms{RESET}")
    print(f"  {DIM}Trace:  {trace_id}{RESET}")
    print(f"  {DIM}Span:   {span_id}{RESET}")
    print(f"  {DIM}Parent: {parent_id}{RESET}")
    print(f"  {DIM}Service: {service}{RESET}")

    for key in ["traceloop.entity.input", "traceloop.entity.output",
                "score.name", "score.value", "score.source", "score.rationale",
                "score.trace_id"]:
        if key in attrs:
            val_str = str(attrs[key])
            if len(val_str) > 120:
                val_str = val_str[:120] + "..."
            print(f"  {CYAN}{key}{RESET}: {val_str}")


class TraceCollector(TraceServiceServicer):
    def Export(self, request: ExportTraceServiceRequest, context):
        span_count = 0
        for resource_spans in request.resource_spans:
            resource_attrs = {}
            for kv in resource_spans.resource.attributes:
                if kv.value.HasField("string_value"):
                    resource_attrs[kv.key] = kv.value.string_value

            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    print_span(span, resource_attrs)
                    span_count += 1

        if span_count > 0:
            print(f"\n{GREEN}✓ Received {span_count} span(s) via gRPC{RESET}\n")

        return ExportTraceServiceResponse()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    add_TraceServiceServicer_to_server(TraceCollector(), server)
    server.add_insecure_port("0.0.0.0:4317")
    server.start()

    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  Mini OTEL Collector — gRPC on port 4317{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"  Listening: grpc://0.0.0.0:4317")
    print(f"  Protocol:  OTLP/gRPC")
    print()
    print(f"  Test with: python examples/agent_grpc.py")
    print(f"{BOLD}{'=' * 70}{RESET}\n")

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
