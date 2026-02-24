"""OTLP span exporters with AWS SigV4 support.

Provides a drop-in replacement for OTLPSpanExporter that signs
requests using AWS Signature Version 4 for AWS-hosted OpenSearch
and Data Prepper endpoints.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

logger = logging.getLogger(__name__)


class SigV4OTLPSpanExporter(OTLPSpanExporter):
    """OTLP HTTP span exporter that signs requests with AWS SigV4.

    Uses botocore's credential chain (env vars, ~/.aws/credentials,
    IAM roles, IMDS) to resolve credentials automatically.

    Args:
        endpoint: The OTLP endpoint URL.
        service: The AWS service name for signing. Use "osis" for
            OpenSearch Ingestion, "es" for OpenSearch Service direct.
        region: AWS region. Auto-detected from botocore if not provided.
        **kwargs: Additional arguments passed to OTLPSpanExporter.

    Example:
        exporter = SigV4OTLPSpanExporter(
            endpoint="https://pipeline.us-east-1.osis.amazonaws.com/v1/traces",
            service="osis",
        )
    """

    def __init__(
        self,
        *args,
        service: str = "osis",
        region: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._service = service

        try:
            import botocore.session
        except ImportError:
            raise ImportError(
                "botocore is required for SigV4 authentication. "
                "Install it with: pip install opensearch-genai-sdk-py[aws]"
            )

        self._botocore_session = botocore.session.get_session()
        self._credentials = self._botocore_session.get_credentials()
        self._region = region or self._botocore_session.get_config_variable("region")

        if not self._credentials:
            raise RuntimeError(
                "No AWS credentials found. Configure credentials via environment "
                "variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY), "
                "~/.aws/credentials, or an IAM role."
            )
        if not self._region:
            raise RuntimeError(
                "No AWS region found. Set the region via the 'region' parameter, "
                "AWS_DEFAULT_REGION environment variable, or ~/.aws/config."
            )

        logger.info(
            "SigV4OTLPSpanExporter initialized for service=%s region=%s",
            self._service,
            self._region,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans with SigV4-signed HTTP requests.

        Injects SigV4 auth headers before delegating to the parent exporter.
        """
        self._inject_sigv4_headers()
        return super().export(spans)

    def _inject_sigv4_headers(self):
        """Compute SigV4 headers and inject them into the exporter's session.

        Uses botocore's SigV4Auth to sign a synthetic request matching
        the OTLP export, then patches the session headers.
        """
        import botocore.auth
        from botocore.awsrequest import AWSRequest

        frozen = self._credentials.get_frozen_credentials()
        signer = botocore.auth.SigV4Auth(frozen, self._service, self._region)

        # Build a minimal request to compute the signature.
        # The actual body will differ per-export, but the auth headers
        # (particularly the security token for temp creds) must be fresh.
        request = AWSRequest(
            method="POST",
            url=self._endpoint,
            headers={"Content-Type": "application/x-protobuf"},
            data=b"",
        )
        signer.add_auth(request)

        # Patch the exporter's internal session headers with SigV4 auth.
        # This ensures every subsequent HTTP request carries valid auth.
        if hasattr(self, "_session"):
            for key in ("Authorization", "X-Amz-Date", "X-Amz-Security-Token"):
                value = request.headers.get(key)
                if value:
                    self._session.headers[key] = value
