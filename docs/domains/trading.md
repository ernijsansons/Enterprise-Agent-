# Trading Domain

## Environment Notes

- Ensure required API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.) are set via environment variables or a local `.env` file.
- Override configuration or dotenv locations with `ENTERPRISE_AGENT_CONFIG` and `ENTERPRISE_AGENT_DOTENV` when needed.
- Telemetry output respects `TELEMETRY_FILE`; set `TELEMETRY_MAX_BYTES` (bytes) to bound log size.

Use for strategies: Input signals, output backtested results.

## Quick Start
```bash
make run --domain=trading --input="MA crossover strategy"
```

## Features
- Risk management (max 10% drawdown)
- Sharpe ratio validation (>1.0)
- PnL optimization
- Signal refinement

## Example Prompts
- "Backtest MA crossover strategy"
- "Analyze momentum indicators"
- "Create mean reversion system"
- "Optimize portfolio allocation"