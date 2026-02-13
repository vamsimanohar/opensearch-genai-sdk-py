"""AWS SigV4 authentication with opensearch-genai-sdk.

When your Data Prepper or OpenSearch Ingestion pipeline is hosted on AWS,
the SDK auto-detects the AWS endpoint and signs OTLP requests with SigV4.

No extra configuration needed — just pass an AWS endpoint URL.

Requires: pip install opensearch-genai-sdk[aws]
"""

from opensearch_genai_sdk import register, workflow, score

# --- AWS-hosted OpenSearch Ingestion (OSIS) ---
# SigV4 is auto-detected from the hostname (.amazonaws.com, .osis., etc.)
# Uses the default boto3 credential chain (env vars, ~/.aws/credentials, IAM role)
register(
    endpoint="https://my-pipeline.us-east-1.osis.amazonaws.com/v1/traces",
    service_name="my-llm-app",
)

# Explicit region override if needed
# register(
#     endpoint="https://my-pipeline.us-east-1.osis.amazonaws.com/v1/traces",
#     aws_region="us-west-2",
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
    # 1. register() sees ".osis.amazonaws.com" in the endpoint
    # 2. Creates SigV4OTLPSpanExporter instead of plain OTLPSpanExporter
    # 3. Every export() call signs the request with AWS credentials:
    #    - Authorization: AWS4-HMAC-SHA256 Credential=...
    #    - X-Amz-Date: 20240101T000000Z
    #    - X-Amz-Security-Token: ... (if using temporary credentials)
    # 4. Data Prepper / OSIS accepts the signed request
