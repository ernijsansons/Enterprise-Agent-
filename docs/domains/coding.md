# Coding Domain

## Environment Notes

- Ensure required API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.) are set via environment variables or a local `.env` file.
- Override configuration or dotenv locations with `ENTERPRISE_AGENT_CONFIG` and `ENTERPRISE_AGENT_DOTENV` when needed.
- Telemetry output respects `TELEMETRY_FILE`; set `TELEMETRY_MAX_BYTES` (bytes) to bound log size.

Use for repos: Input spec, output PR-ready code.

## Quick Start
```bash
make run --domain=coding --input="API spec"
```

## Features
- 97% test coverage requirement
- SonarQube code quality checks
- Automated refactoring suggestions
- Multi-language support (Python, JavaScript, Go, Rust)

## Example Prompts
- "Generate Python REST API from spec"
- "Create React component with TypeScript"
- "Write unit tests for existing code"
- "Refactor legacy code to modern patterns"