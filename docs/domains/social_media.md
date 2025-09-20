# Social Media Domain

## Environment Notes

- Ensure required API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.) are set via environment variables or a local `.env` file.
- Override configuration or dotenv locations with `ENTERPRISE_AGENT_CONFIG` and `ENTERPRISE_AGENT_DOTENV` when needed.
- Telemetry output respects `TELEMETRY_FILE`; set `TELEMETRY_MAX_BYTES` (bytes) to bound log size.

Use for campaigns: Input strategy, output engaging content.

## Quick Start
```bash
make run --domain=social_media --input="Q4 campaign strategy"
```

## Features
- Engagement KPI tracking (min 10 likes)
- Brand tone consistency
- Platform-specific optimization
- Hashtag and emoji suggestions

## Example Prompts
- "Draft Q4 campaign posts for Twitter"
- "Create LinkedIn thought leadership content"
- "Generate Instagram story ideas"
- "Plan TikTok video concepts"