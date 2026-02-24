"""AWS SigV4 authentication with opensearch-genai-sdk-py.

When your Data Prepper or OpenSearch Ingestion pipeline is hosted on AWS,
you must specify auth="sigv4" to enable SigV4 signing for OTLP requests.

Requires: pip install opensearch-genai-sdk-py[aws]
"""

from opensearch_genai_sdk import register, score, workflow

# --- AWS-hosted OpenSearch Ingestion (OSIS) ---
# SigV4 signing must be explicitly enabled with auth="sigv4"
# Uses the default boto3 credential chain (env vars, ~/.aws/credentials, IAM role)
register(
    endpoint="https://my-pipeline.us-east-1.osis.amazonaws.com/v1/traces",
    service_name="my-llm-app",
    auth="sigv4",
)

# Explicit region override if needed
# register(
#     endpoint="https://my-pipeline.us-east-1.osis.amazonaws.com/v1/traces",
#     auth="sigv4",
#     region="us-west-2",
# )


@workflow(name="qa_pipeline")
def run(question: str) -> str:
    return f"Answer to: {question}"


if __name__ == "__main__":
    # Traces flow through SigV4-signed OTLP export
    result = run("What is OpenSearch?")
    print(result)

    # Scores use the same SigV4-signed exporter
    score(
        name="relevance",
        value=0.9,
        trace_id="abc123",
        source="human",
    )

    # What happens under the hood:
    #
    # 1. register() sees auth="sigv4" and creates SigV4OTLPSpanExporter
    # 2. Every export() call signs the request with AWS credentials:
    #    - Authorization: AWS4-HMAC-SHA256 Credential=...
    #    - X-Amz-Date: 20240101T000000Z
    #    - X-Amz-Security-Token: ... (if using temporary credentials)
    # 3. Data Prepper / OSIS accepts the signed request
