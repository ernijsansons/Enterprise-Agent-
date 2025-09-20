# Enterprise Coding Agent v3.4

Multi-domain AI agent for software development, social media, content, trading, real estate, and thermonuclear UI workflows.

## Quick Start

1. Clone: `git clone <repo> && cd enterprise-coding-agent-v3.4`
2. Setup: `make setup` (installs deps, verifies Codex CLI)
3. Config: Copy .env.example to .env, add OPENAI_API_KEY, V0_API_KEY, FIGMA_PERSONAL_ACCESS_TOKEN, etc.
4. Run Coding Domain: `make run --domain=coding --input="Build REST API"`
5. Run UI Domain: `make run --domain=ui --input="Design analytics dashboard"`
6. Multi-Domain Examples: See docs/domains/.

## Multi-Domain Quick Starts

- **Coding**: Decompose spec, generate code, test to 97% coverage.
- **UI**: Generate thermonuclear Next.js/Tailwind/ShadCN interfaces with Mantine charts, 3D depth, accessibility, and micro-interactions.
- **Social Media**: Plan campaigns, draft posts, validate engagement.
- **Content**: Structure articles, generate drafts, check SEO.
- **Trading**: Analyze signals, backtest, alert on risks.
- **Real Estate**: Source properties, value, predict cash flow.

See docs/quickstart.md for details.

## Final Acceptance Checklist
- [x] All phases completed with tests passing.
- [x] `make ci` green on GitHub Actions.
- [x] Demo scripts produce sample outputs for coding, ui, social, content, trading, and real estate domains.
- [x] Coding domain coverage at or above 0.97.
- [x] Cost guardrails enforced and logged.
- [x] HITL gates active for high-risk actions.
- [x] Policies and data retention configured.
- [x] Docs complete: quick start, runbooks, benchmarks, migration, ui domain.
