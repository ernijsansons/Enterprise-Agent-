# Quick Start: Enterprise Agent v3.4 - Production Ready

## 🚀 Enterprise-Grade AI Agent with Zero API Costs

This guide shows you how to install and use the enhanced Enterprise Agent v3.4 with production-ready features including comprehensive metrics, security, and observability.

### ⚡ What's New in v3.4
- **Production-Ready**: Complete CI/CD pipeline with automated testing
- **Enhanced Security**: Comprehensive error handling and vulnerability scanning
- **Full Observability**: Real-time metrics collection and performance monitoring
- **Privacy-First**: GDPR-compliant telemetry with user consent management
- **Advanced Caching**: Adaptive TTL with compression and multiple eviction policies
- **Robust Validation**: Enhanced input validation with clear error messages

### Prerequisites
- **Python 3.9+** (verified during installation)
- **Node.js 16+** (for Claude Code CLI - zero API costs)
- **Git 2.0+** (for version control)
- **Anthropic Max subscription** ($200/month for CLI access)

### ⚡ One-Line Installation

```bash
# Recommended: Use our enhanced installer with dependency validation
curl -sSL https://raw.githubusercontent.com/yourorg/enterprise-agent/main/install.sh | bash

# Alternative: Clone and install manually
git clone https://github.com/yourorg/enterprise-agent.git
cd enterprise-agent
bash install.sh

# Verify installation with comprehensive status check
enterprise-agent status
```

### 🔧 Setup Claude Code (Zero API Costs)

```bash
# Step 1: Install Claude Code CLI (with version validation)
npm install -g @anthropic-ai/claude-code

# Step 2: Verify installation and check version
claude --version

# Step 3: Login with your Max subscription
claude login

# Step 4: Configure agent for subscription mode
enterprise-agent config claude-code --enable

# Step 5: Validate configuration
enterprise-agent config validate
```

> **🔥 Cost Savings**: This setup eliminates API costs entirely by using your Max subscription's included CLI access, saving $1000s per month.

## Usage in Any Project

### Initialize in Your Project

```bash
# Navigate to your project directory
cd /path/to/your/project

# Initialize Enterprise Agent
enterprise-agent init

# Verify initialization
ls -la .enterprise-agent/
```

This creates `.enterprise-agent/` directory with:
- 📁 `config.yml` - Project-specific settings
- 📁 `templates/` - Custom prompt templates
- 📁 `context/` - Project knowledge and context
- 📁 `history/` - Command history and logs
- 📁 `workflows/` - Cursor IDE integration files

### Quick Commands

```bash
# Code review and analysis
enterprise-agent run --input "Review this codebase for best practices"

# Generate comprehensive tests
enterprise-agent run --input "Create comprehensive tests for the auth module"

# UI components (for frontend projects)
enterprise-agent run --domain ui --input "Create a responsive dashboard component"

# Backend API development
enterprise-agent run --domain coding --input "Create a REST API endpoint for user management"

# Interactive mode for complex tasks
enterprise-agent interactive

# Project analysis
enterprise-agent analyze --input "Analyze project structure and suggest improvements"
```

### Cursor Integration

1. **Copy workflow to your project**:
   ```bash
   mkdir -p .cursor/workflows
   cp ~/.enterprise-agent/.cursor/workflows/enterprise-agent.yml .cursor/workflows/
   ```

2. **Use keyboard shortcuts in Cursor**:
   - `Cmd+Shift+A` - Quick agent prompt
   - `Cmd+Shift+R` - Review current file
   - `Cmd+Shift+T` - Generate tests
   - `Cmd+Shift+I` - Interactive mode

3. **Access via Command Palette** (`Cmd+Shift+P`):
   - "Enterprise Agent: Review Code"
   - "Enterprise Agent: Generate Tests"
   - "Enterprise Agent: Interactive Mode"

## Project-Specific Configuration

### React Project

```yaml
# .enterprise-agent/config.yml
default_domain: ui
use_claude_code: true

context:
  tech_stack: [react, typescript, tailwind]
  testing: jest

models:
  primary: claude-3-5-sonnet-20241022
  fallback: gpt-4o-mini
```

### Python API Project

```yaml
# .enterprise-agent/config.yml
default_domain: coding
use_claude_code: true

context:
  tech_stack: [python, fastapi, postgresql]
  testing: pytest

models:
  primary: claude-3-5-sonnet-20241022
  fallback: gpt-4o-mini
```

## Docker Usage

```bash
# Build image
docker build -t enterprise-agent .

# Run in any project
docker run -it -v $(pwd):/workspace enterprise-agent

# Or create alias
alias ea="docker run -it -v \$(pwd):/workspace enterprise-agent"
```

## Advanced Workflows

### Monorepo Setup

```bash
# Root level
enterprise-agent init

# Frontend
cd frontend/
enterprise-agent init --template react

# Backend
cd ../backend/
enterprise-agent init --template python-api
```

### Custom Templates

Create project-specific templates in `.enterprise-agent/templates/`:

```markdown
# .enterprise-agent/templates/api-endpoint.md
Create a REST API endpoint for {{ feature }} with:
- Input validation
- Database operations
- Error handling
- Comprehensive tests
```

Use with:
```bash
enterprise-agent run --template api-endpoint --vars feature=user-auth
```

## Cost Savings

### With Claude Code CLI (Recommended)
- Max subscription: $200/month
- API usage: **$0** (included in subscription)
- **Total: $200/month**

### Without Claude Code (API)
- Max subscription: $200/month
- API usage: $50-200/month extra
- **Total: $250-400/month**

**Savings: $50-200+/month** 💰

## Common Commands

```bash
# Initialize agent in project
enterprise-agent init

# Run with specific domain
enterprise-agent run --domain coding --input "Add authentication"
enterprise-agent run --domain ui --input "Create a navbar"

# Analyze project structure
enterprise-agent analyze

# Interactive mode
enterprise-agent interactive

# Custom config
enterprise-agent run --config ./custom-config.yml --input "Custom prompt"

# Show version
enterprise-agent version

# Setup Claude Code
enterprise-agent setup
```

## Integration Examples

### In package.json

```json
{
  "scripts": {
    "agent:review": "enterprise-agent run --input 'Review codebase'",
    "agent:test": "enterprise-agent run --input 'Generate missing tests'",
    "agent:docs": "enterprise-agent run --input 'Update documentation'"
  }
}
```

### In Makefile

```makefile
.PHONY: agent-review agent-test

agent-review:
	enterprise-agent run --input "Review code for best practices"

agent-test:
	enterprise-agent run --input "Generate comprehensive tests"
```

### GitHub Actions

```yaml
name: AI Code Review
on: [pull_request]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: npm install -g enterprise-agent-cli
      - run: enterprise-agent run --input "Review this PR"
```

## Troubleshooting

### Agent Not Found
```bash
# Check installation
which enterprise-agent

# Reinstall
npm install -g enterprise-agent-cli
```

### Claude Code Issues
```bash
# Check CLI
claude --version

# Re-login
claude login

# Check config
cat ~/.enterprise-agent/config.yml
```

### Cursor Integration
```bash
# Copy workflow files
cp -r ~/.enterprise-agent/.cursor .cursor/

# Restart Cursor
```

## Next Steps

1. **Install globally**: `npm install -g enterprise-agent-cli`
2. **Setup Claude Code**: `claude login`
3. **Initialize in projects**: `enterprise-agent init`
4. **Configure Cursor**: Copy workflow files
5. **Start coding**: Use keyboard shortcuts or command palette

For detailed documentation, see:
- [MULTI_PROJECT_USAGE.md](./MULTI_PROJECT_USAGE.md) - Complete guide
- [CLAUDE_CODE_SETUP.md](./CLAUDE_CODE_SETUP.md) - Zero-cost setup
- [.cursor/workflows/](./cursor/workflows/) - Cursor integration

**Happy coding with zero API costs!** 🚀