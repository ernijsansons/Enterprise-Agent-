# Enterprise Coding Agent v3.4

Multi-domain AI agent for software development, social media, content, trading, real estate, and thermonuclear UI workflows.

## Quick Start

1. Clone: `git clone https://github.com/ernijsansons/Enterprise-Agent-.git && cd Enterprise-Agent-`
2. Setup: `make setup` (installs deps, verifies Codex CLI)
3. Config: Copy .env.example to .env, add OPENAI_API_KEY, V0_API_KEY, FIGMA_PERSONAL_ACCESS_TOKEN, etc.
4. Run Coding Domain: `make run --domain=coding --input="Build REST API"`
5. Run UI Domain: `make run --domain=ui --input="Design analytics dashboard"`
6. Multi-Domain Examples: See docs/domains/.

## Claude Code CLI Integration

### Installation
```bash
npm install -g @anthropic/claude-code
claude --version
```

### Authentication
The agent prioritizes Claude models through your Anthropic Max subscription:
1. Ensure Claude Code CLI is authenticated: `claude login`
2. Configure `configs/claude_code_config.yaml` for model preferences
3. Models are routed based on complexity:
   - Simple tasks → Claude 3.5 Sonnet (fast, included in Max)
   - Complex/security tasks → Claude 3 Opus (powerful)
   - Fallback → GPT-3.5 (only if Claude unavailable)

### Session Management
Sessions are automatically managed for context retention:
- Fork sessions for parallel processing
- Configurable retention time in `claude_code_config.yaml`
- Use `claude session list` to view active sessions

## Multi-Domain Quick Starts

- **Coding**: Decompose spec, generate code, test to 97% coverage.
- **UI**: Generate thermonuclear Next.js/Tailwind/ShadCN interfaces with Mantine charts, 3D depth, accessibility, and micro-interactions.
- **Social Media**: Plan campaigns, draft posts, validate engagement.
- **Content**: Structure articles, generate drafts, check SEO.
- **Trading**: Analyze signals, backtest, alert on risks.
- **Real Estate**: Source properties, value, predict cash flow.

See docs/quickstart.md for details.

## Repository Best Practices

### Git Workflow
1. Use feature branches for development
2. Commit with semantic messages (feat:, fix:, docs:, etc.)
3. Line endings: `git config core.autocrlf true` on Windows
4. Run tests before pushing: `make test`

### CI/CD Pipeline
GitHub Actions runs on every push:
- Tests: `pytest` with coverage reporting
- Linting: `ruff check` and `black`
- Security: `bandit` for vulnerability scanning
- See `.github/workflows/ci.yml` for configuration

## Final Acceptance Checklist
- [x] All phases completed with tests passing.
- [x] `make ci` green on GitHub Actions.
- [x] Demo scripts produce sample outputs for coding, ui, social, content, trading, and real estate domains.
- [x] Coding domain coverage at or above 0.97.
- [x] Cost guardrails enforced and logged.
- [x] HITL gates active for high-risk actions.
- [x] Policies and data retention configured.
- [x] Docs complete: quick start, runbooks, benchmarks, migration, ui domain.
