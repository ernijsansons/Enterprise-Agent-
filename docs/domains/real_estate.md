# Real Estate Domain

## Environment Notes

- Ensure required API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.) are set via environment variables or a local `.env` file.
- Override configuration or dotenv locations with `ENTERPRISE_AGENT_CONFIG` and `ENTERPRISE_AGENT_DOTENV` when needed.
- Telemetry output respects `TELEMETRY_FILE`; set `TELEMETRY_MAX_BYTES` (bytes) to bound log size.

Use for properties: Input address, output valuation analysis.

## Quick Start
```bash
make run --domain=real_estate --input="123 Main St valuation"
```

## Features
- Yield threshold validation (cap rate >8%)
- Cash flow analysis (DSCR >1.25)
- Comparable property analysis
- Market trend insights

## Example Prompts
- "Value property at 123 Main St"
- "Analyze rental income potential"
- "Compare investment properties"
- "Generate market report for neighborhood"