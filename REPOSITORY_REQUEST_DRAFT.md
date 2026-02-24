# Repository Request Template Responses

## Are you requesting a new GitHub Repository within opensearch-project GitHub Organization?
Yes, I am requesting a new repository named `opensearch-genai-sdk-py`.

Current repository: https://github.com/vamsimanohar/opensearch-genai-sdk-py

## GitHub Repository Proposal
https://github.com/opensearch-project/dashboards-observability/issues/2591

## GitHub Repository Additional Information

### 1. What is the new GitHub repository name?
`opensearch-genai-sdk-py`

### 2. Project description and community value?
The OpenSearch GenAI SDK is an OpenTelemetry-native Python library that enables developers to instrument and evaluate agentic AI applications using OpenSearch as the backend storage system. It provides one-line setup for comprehensive LLM observability with zero vendor lock-in.

**Community Value:**
- Positions OpenSearch as a leader in the growing AI observability market
- Provides open-source alternative to proprietary solutions (LangSmith, Arize)
- Enables easy migration of AI developers to OpenSearch ecosystem
- Reference implementation of OpenTelemetry GenAI semantic conventions

### 3. What user problem are you trying to solve with this new repository?
AI/LLM developers face significant observability setup friction:
- Complex OpenTelemetry configuration for AI workloads
- Manual setup of 40+ different LLM provider instrumentors
- Complex AWS SigV4 authentication for OpenSearch endpoints
- Disconnect between evaluation scoring and observability pipelines
- Vendor lock-in with proprietary observability solutions

The SDK solves this with one-line setup: `register(endpoint="...")` that automatically configures everything needed for production AI observability.

### 4. Why do we create a new repo at this time?
- **Market Timing**: AI observability is rapidly growing with OpenTelemetry GenAI conventions stabilizing
- **Strategic Positioning**: First major open-source backend to offer native AI SDK
- **Production Ready**: Complete implementation already exists with comprehensive testing
- **Community Demand**: Growing need for OpenSearch-native AI tooling

### 5. Is there any existing projects that is similar to your proposal?
**Similar Projects:**
- Traceloop: Proprietary SaaS, vendor lock-in
- LangSmith: Proprietary, LangChain-specific
- Arize: Proprietary, expensive enterprise pricing

**Key Differentiators:**
- ✅ Open source with Apache 2.0 license
- ✅ OpenSearch-native integration and optimization
- ✅ Standards-based (100% OpenTelemetry compliant)
- ✅ Zero vendor lock-in (remove decorators, code still works)
- ✅ Self-hosted option for enterprise compliance

### 6. Should this project be in OpenSearch Core/OpenSearch Dashboards Core? If no, why not?
**No, this should be a separate repository because:**
- **Different Language**: Python SDK vs Java/TypeScript core components
- **Different Release Cycle**: SDK needs frequent releases for AI ecosystem changes
- **Different User Base**: AI developers vs OpenSearch operators
- **Standalone Value**: Can be used independently of Dashboards
- **Cleaner Governance**: Separate maintainers and contribution model

### 7. Is this project an OpenSearch/OpenSearch Dashboards plugin to be included as part of the OpenSearch release?
No, this is a client-side Python SDK library distributed via PyPI. It connects TO OpenSearch but is not part of the OpenSearch release itself.

## GitHub Repository Owners

### 1. Who will be supporting this repo going forward?
**Initial Maintainer**: Vamsi Manohar (@vamsimanohar)
- Primary architect and developer (100% of current implementation)
- Deep expertise in OpenSearch and AI observability
- Committed to long-term maintenance and community building

**Expansion Plan**: Recruit 2-4 additional maintainers within 6 months from:
- OpenSearch community members with Python/AI experience
- Contributors from AI framework communities
- Enterprise users adopting the SDK

### 2. What is your plan (including staffing) to be responsive to the community?
**Commitment:**
- **Issue Response**: <48 hours for initial triage
- **PR Review**: <72 hours for community contributions
- **Security Issues**: <24 hours for critical vulnerabilities
- **Documentation**: Maintain comprehensive guides and examples
- **Community Engagement**: Monthly releases, regular communication

**Staffing Plan:**
- Vamsi Manohar: Full-time commitment as primary maintainer
- Additional maintainers: Progressive onboarding with clear responsibilities
- Community contributors: Active recruitment and mentorship

### 3. Initial Maintainers List (max 3 users, provide GitHub aliases):
1. @vamsimanohar (Primary maintainer)
2. TBD (Will recruit from OpenSearch community)
3. TBD (Will recruit from OpenSearch community)

## GitHub Repository Source Code / License / Libraries

### 1. Please provide the URL to the source code.
https://github.com/vamsimanohar/opensearch-genai-sdk-py

### 2. What is the license for the source code?
Apache License 2.0

### 3. Does the source code include any third-party code that is not compliant with the Apache License 2.0?
No. All dependencies are Apache 2.0 or MIT licensed:
- OpenTelemetry SDK: Apache 2.0
- AWS dependencies (botocore, requests-aws4auth): Apache 2.0
- LLM instrumentors: Apache 2.0 or MIT
- Development tools: MIT/BSD

No GPL or copyleft dependencies are included.

## Publication Target(s)
- PyPI

**Publication Details:**
- Package Name: `opensearch-genai-sdk-py`
- Current Version: 0.2.0 (production ready)
- Automated GitHub Actions → PyPI publishing pipeline
- Request creation of `opensearch-project` PyPI organization
- OIDC trusted publishing for security