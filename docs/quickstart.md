# Quick Start

For Coding: `make run --domain=coding --input="Build REST API"`

For UI: `make run --domain=ui --input="Design thermonuclear SaaS dashboard"`

For Trading: `make run --domain=trading --input="MA crossover backtest"`

For Content: `make run --domain=content --input="AI trends article"`

Before running any domain, copy `.env.example` to `.env` (keep it local) or export the required keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.). You can also point `ENTERPRISE_AGENT_CONFIG` or `ENTERPRISE_AGENT_DOTENV` at custom files when experimenting.

See `docs/domains/` for domain-specific guidance.
