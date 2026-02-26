"""Tests for opensearch_genai_sdk_py.register.

Focused on the auth auto-detection logic and endpoint routing.
"""

from unittest.mock import MagicMock, patch

import pytest

from opensearch_genai_sdk_py.register import _is_aws_endpoint


class TestIsAwsEndpoint:
    """Unit tests for the AWS endpoint detection helper."""

    def test_osis_endpoint(self):
        assert _is_aws_endpoint("https://pipeline.us-east-1.osis.amazonaws.com/v1/traces")

    def test_opensearch_service_endpoint(self):
        assert _is_aws_endpoint("https://search.us-west-2.es.amazonaws.com/v1/traces")

    def test_generic_amazonaws_subdomain(self):
        assert _is_aws_endpoint("https://anything.eu-central-1.amazonaws.com/traces")

    def test_localhost_is_not_aws(self):
        assert not _is_aws_endpoint("http://localhost:4318/v1/traces")

    def test_self_hosted_opensearch_is_not_aws(self):
        assert not _is_aws_endpoint("https://my-opensearch.example.com/v1/traces")

    def test_grpc_aws_endpoint(self):
        assert _is_aws_endpoint("grpcs://pipeline.us-east-1.osis.amazonaws.com:443")

    def test_non_aws_https_is_not_aws(self):
        assert not _is_aws_endpoint("https://otel-collector.internal:4318/v1/traces")


class TestRegisterAuthAutoDetect:
    """Verify that register() picks the right exporter based on auth= and endpoint."""

    def _make_mock_botocore(self):
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = MagicMock()
        mock_session.get_config_variable.return_value = "us-east-1"
        return patch("botocore.session.get_session", return_value=mock_session)

    @patch("opensearch_genai_sdk_py.register._create_http_exporter")
    def test_auto_uses_sigv4_for_aws_endpoint(self, mock_create_http):
        """auth='auto' + AWS endpoint → SigV4."""
        from opensearch_genai_sdk_py.register import _create_exporter

        mock_create_http.return_value = MagicMock()
        aws_endpoint = "https://pipeline.us-east-1.osis.amazonaws.com/v1/traces"
        _create_exporter(aws_endpoint, protocol=None, auth="auto", region=None, service="osis", headers=None)

        mock_create_http.assert_called_once()
        _, kwargs = mock_create_http.call_args
        assert kwargs.get("use_sigv4") is True or mock_create_http.call_args.args[1] is True

    @patch("opensearch_genai_sdk_py.register._create_http_exporter")
    def test_auto_uses_plain_http_for_non_aws_endpoint(self, mock_create_http):
        """auth='auto' + non-AWS endpoint → no SigV4."""
        from opensearch_genai_sdk_py.register import _create_exporter

        mock_create_http.return_value = MagicMock()
        _create_exporter(
            "http://localhost:4318/v1/traces",
            protocol=None, auth="auto", region=None, service="osis", headers=None,
        )

        mock_create_http.assert_called_once()
        use_sigv4_arg = mock_create_http.call_args.args[1]
        assert use_sigv4_arg is False

    @patch("opensearch_genai_sdk_py.register._create_http_exporter")
    def test_none_skips_sigv4_even_for_aws_endpoint(self, mock_create_http):
        """auth='none' always skips SigV4, even for AWS endpoints."""
        from opensearch_genai_sdk_py.register import _create_exporter

        mock_create_http.return_value = MagicMock()
        aws_endpoint = "https://pipeline.us-east-1.osis.amazonaws.com/v1/traces"
        _create_exporter(aws_endpoint, protocol=None, auth="none", region=None, service="osis", headers=None)

        use_sigv4_arg = mock_create_http.call_args.args[1]
        assert use_sigv4_arg is False

    @patch("opensearch_genai_sdk_py.register._create_http_exporter")
    def test_sigv4_always_uses_sigv4(self, mock_create_http):
        """auth='sigv4' always enables SigV4 regardless of endpoint."""
        from opensearch_genai_sdk_py.register import _create_exporter

        mock_create_http.return_value = MagicMock()
        _create_exporter(
            "http://localhost:4318/v1/traces",
            protocol=None, auth="sigv4", region=None, service="osis", headers=None,
        )

        use_sigv4_arg = mock_create_http.call_args.args[1]
        assert use_sigv4_arg is True
