# Content Domain

## Environment Notes

- Ensure required API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.) are set via environment variables or a local `.env` file.
- Override configuration or dotenv locations with `ENTERPRISE_AGENT_CONFIG` and `ENTERPRISE_AGENT_DOTENV` when needed.
- Telemetry output respects `TELEMETRY_FILE`; set `TELEMETRY_MAX_BYTES` (bytes) to bound log size.

Use for articles: input a topic, output an SEO-optimized draft.

## Quick Start
```bash
make run --domain=content --input="AI trends article"
```

## Features
- Flesch-Kincaid readability score >= 8
- SEO keyword optimization
- Plagiarism detection (<5% duplication)
- Grammar and style improvements

## Example Prompts
- "Write blog on AI trends"
- "Create technical documentation"
- "Generate marketing copy for product launch"
- "Draft white paper on blockchain"
