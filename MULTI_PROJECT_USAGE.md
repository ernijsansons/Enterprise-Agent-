# Multi-Project Usage Guide

## 🚀 Using Enterprise Agent Across Different Projects

This guide shows you how to use the Enterprise Agent across multiple projects in different development environments, especially with Cursor and Claude Code terminal integration.

## Installation Options

### Option 1: NPM Global Install (Recommended)

```bash
# Install globally
npm install -g enterprise-agent-cli

# Or use the installer
curl -sSL https://raw.githubusercontent.com/yourorg/enterprise-agent/main/install.sh | bash
```

### Option 2: Manual Setup

```bash
# Clone the repository
git clone https://github.com/yourorg/enterprise-agent.git ~/.enterprise-agent
cd ~/.enterprise-agent

# Install dependencies
pip install -r requirements.txt

# Create symlink
ln -s ~/.enterprise-agent/enterprise_agent_cli.py /usr/local/bin/enterprise-agent
chmod +x /usr/local/bin/enterprise-agent
```

### Option 3: Docker

```bash
# Build the image
docker build -t enterprise-agent .

# Create alias
echo 'alias enterprise-agent="docker run -it -v $(pwd):/workspace enterprise-agent"' >> ~/.bashrc
```

## Project Setup Workflows

### Workflow 1: Per-Project Configuration

Each project gets its own configuration:

```bash
# Navigate to your project
cd /path/to/your/project

# Initialize Enterprise Agent
enterprise-agent init

# This creates:
# .enterprise-agent/
# ├── config.yml          # Project-specific settings
# ├── templates/          # Custom prompts
# ├── context/           # Project context
# └── history/           # Command history
```

### Workflow 2: Global Configuration with Overrides

Use global settings with project-specific overrides:

```bash
# Global config at ~/.enterprise-agent/config.yml
use_claude_code: true
models:
  primary: claude-3-5-sonnet-20241022
  fallback: gpt-4o-mini

# Project override at .enterprise-agent/config.yml
default_domain: ui
context:
  tech_stack: [react, typescript, tailwind]
```

### Workflow 3: Template-Based Initialization

Create project templates for different types:

```bash
# Create templates
mkdir -p ~/.enterprise-agent/project-templates

# React project template
enterprise-agent init --template react

# Python API template
enterprise-agent init --template python-api

# Full-stack template
enterprise-agent init --template fullstack
```

## Cursor Integration

### Setup in Cursor

1. **Install Enterprise Agent globally**:
   ```bash
   npm install -g enterprise-agent-cli
   ```

2. **Copy Cursor workflow** to your project:
   ```bash
   mkdir -p .cursor/workflows
   cp ~/.enterprise-agent/.cursor/workflows/enterprise-agent.yml .cursor/workflows/
   ```

3. **Configure keyboard shortcuts** in Cursor settings:
   ```json
   {
     "keybindings": [
       {
         "key": "cmd+shift+a",
         "command": "enterprise-agent.run"
       },
       {
         "key": "cmd+shift+i",
         "command": "enterprise-agent.interactive"
       }
     ]
   }
   ```

### Cursor Command Palette

Access via Command Palette (`Cmd+Shift+P`):

- `Enterprise Agent: Initialize` - Set up in current project
- `Enterprise Agent: Review Code` - Review current file
- `Enterprise Agent: Generate Tests` - Create tests
- `Enterprise Agent: Interactive Mode` - Start chat session
- `Enterprise Agent: Custom Prompt` - Run custom command

### Context Menu Integration

Right-click on files for quick actions:
- **Review This File** - Code review and suggestions
- **Generate Tests** - Create test files
- **Add Documentation** - Generate docs
- **Refactor Code** - Improve structure

## Claude Code Terminal Integration

### Setup Claude Code CLI

```bash
# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Login with your Max subscription
claude login

# Verify setup
claude --version
```

### Configure for Zero-Cost Usage

Update your config to use Claude Code instead of API:

```yaml
# .enterprise-agent/config.yml
use_claude_code: true
models:
  primary: claude-3-5-sonnet-20241022  # Maps to 'sonnet' in CLI
  fallback: claude-3-haiku             # Maps to 'haiku' in CLI

# This eliminates API costs!
```

### Terminal Workflow

```bash
# Quick commands in any project directory
enterprise-agent run --input "Review this codebase"
enterprise-agent run --domain ui --input "Create a dashboard component"
enterprise-agent analyze

# Interactive mode
enterprise-agent interactive
[coding]> Review the authentication system
[coding]> /domain ui
[ui]> Create a responsive navbar
[ui]> exit
```

## Multi-Project Scenarios

### Scenario 1: Frontend + Backend Monorepo

```bash
# Root level
enterprise-agent init

# Frontend specific
cd frontend/
enterprise-agent init --template react
# Edit .enterprise-agent/config.yml:
# default_domain: ui
# context:
#   tech_stack: [react, typescript, tailwind]

# Backend specific
cd ../backend/
enterprise-agent init --template python-api
# Edit .enterprise-agent/config.yml:
# default_domain: coding
# context:
#   tech_stack: [python, fastapi, postgresql]
```

### Scenario 2: Microservices Architecture

```bash
# Service template
create_service() {
    mkdir -p $1
    cd $1
    enterprise-agent init
    cat > .enterprise-agent/config.yml <<EOF
default_domain: coding
context:
  service_name: $1
  architecture: microservice
  tech_stack: [python, docker, kubernetes]
EOF
    cd ..
}

# Create services
create_service user-service
create_service order-service
create_service payment-service
```

### Scenario 3: Client Projects

```bash
# Template for client work
enterprise-agent init --template client-project

# Configure client-specific settings
cat > .enterprise-agent/config.yml <<EOF
client:
  name: "Acme Corp"
  requirements: ["GDPR compliance", "accessibility"]
  tech_stack: ["react", "node", "postgres"]

templates_dir: ~/.enterprise-agent/client-templates/acme

context:
  coding_standards: "Acme coding guidelines"
  security_level: "enterprise"
EOF
```

## Advanced Usage Patterns

### Pattern 1: Context-Aware Prompts

```bash
# The agent automatically detects:
# - Programming languages in use
# - Framework/library dependencies
# - Project structure
# - Existing documentation

enterprise-agent run --input "Add error handling"
# → Automatically uses project's error handling patterns
```

### Pattern 2: Template-Based Generation

```bash
# Custom templates in .enterprise-agent/templates/
cat > .enterprise-agent/templates/api-endpoint.md <<EOF
Create a REST API endpoint for {{ feature }} with:
- Input validation using {{ validation_library }}
- Database operations with {{ orm }}
- Error handling following {{ error_pattern }}
- Tests using {{ test_framework }}
EOF

# Use template
enterprise-agent run --template api-endpoint --vars feature=user-auth
```

### Pattern 3: Workflow Automation

```bash
# Create automation scripts
cat > .enterprise-agent/workflows/code-review.sh <<EOF
#!/bin/bash
echo "🔍 Starting code review workflow..."

# Review changed files
for file in $(git diff --name-only); do
    echo "Reviewing $file..."
    enterprise-agent run --input "Review $file for best practices"
done

# Generate summary
enterprise-agent run --input "Summarize code review findings"
EOF

chmod +x .enterprise-agent/workflows/code-review.sh
```

## Configuration Examples

### React Project

```yaml
# .enterprise-agent/config.yml
default_domain: ui
use_claude_code: true

context:
  framework: react
  typescript: true
  styling: tailwind
  testing: jest

models:
  primary: claude-3-5-sonnet-20241022
  fallback: gpt-4o-mini

templates:
  component: |
    Create a React component for {{ name }} with:
    - TypeScript interfaces
    - Tailwind CSS styling
    - Jest tests
    - Storybook story

custom_commands:
  component: "run --template component --vars name=${1}"
  test: "run --input 'Generate tests for current file'"
  story: "run --input 'Create Storybook story for current component'"
```

### Python API Project

```yaml
# .enterprise-agent/config.yml
default_domain: coding
use_claude_code: true

context:
  language: python
  framework: fastapi
  database: postgresql
  testing: pytest

models:
  primary: claude-3-5-sonnet-20241022
  fallback: gpt-4o-mini

templates:
  endpoint: |
    Create a FastAPI endpoint for {{ resource }} with:
    - Pydantic models for request/response
    - SQLAlchemy ORM operations
    - Proper error handling
    - Comprehensive tests

custom_commands:
  endpoint: "run --template endpoint --vars resource=${1}"
  model: "run --input 'Create SQLAlchemy model for ${1}'"
  test: "run --input 'Generate pytest tests for current file'"
```

## Best Practices

### 1. Project Organization

```
your-project/
├── .enterprise-agent/
│   ├── config.yml           # Project settings
│   ├── templates/          # Custom prompts
│   │   ├── component.md
│   │   ├── api-endpoint.md
│   │   └── test-suite.md
│   ├── context/            # Project knowledge
│   │   ├── architecture.md
│   │   ├── conventions.md
│   │   └── dependencies.md
│   └── history/            # Command history
├── src/
└── README.md
```

### 2. Configuration Layering

1. **Global**: `~/.enterprise-agent/config.yml`
2. **Project**: `.enterprise-agent/config.yml`
3. **Environment**: Environment variables
4. **Command**: CLI flags

### 3. Template Management

```bash
# Global templates
~/.enterprise-agent/templates/
├── react-component.md
├── python-class.md
├── api-endpoint.md
└── test-suite.md

# Project templates
.enterprise-agent/templates/
├── specific-component.md
└── project-specific.md
```

### 4. History and Analytics

```bash
# View command history
enterprise-agent history

# Export for analysis
enterprise-agent history --export --format json > usage.json

# Usage statistics
enterprise-agent stats
```

## Troubleshooting

### Common Issues

1. **Agent not found in project**:
   ```bash
   # Check global installation
   which enterprise-agent

   # Reinstall if needed
   npm install -g enterprise-agent-cli
   ```

2. **Claude Code not working**:
   ```bash
   # Check installation
   claude --version

   # Check login
   claude login

   # Verify config
   cat ~/.enterprise-agent/config.yml
   ```

3. **Cursor integration not working**:
   ```bash
   # Copy workflow files
   cp -r ~/.enterprise-agent/.cursor .cursor/

   # Restart Cursor
   ```

### Debug Mode

```bash
# Enable debug logging
export EA_DEBUG=true
enterprise-agent run --input "debug test"

# Check logs
tail -f ~/.enterprise-agent/logs/debug.log
```

## Migration from Single Project

If you're currently using the agent in a single project:

```bash
# Step 1: Install globally
npm install -g enterprise-agent-cli

# Step 2: Migrate existing config
mv .env .enterprise-agent/config.yml
# Convert .env format to YAML

# Step 3: Test
enterprise-agent run --input "test migration"

# Step 4: Update other projects
cd ../other-project
enterprise-agent init
```

## Conclusion

The Enterprise Agent is designed to work seamlessly across multiple projects with:

- ✅ **Zero-cost Claude Code integration**
- ✅ **Project-specific configurations**
- ✅ **Cursor IDE integration**
- ✅ **Template-based workflows**
- ✅ **Cross-platform compatibility**
- ✅ **Docker containerization**

This setup allows you to maintain consistency across projects while customizing behavior for specific needs.

Happy coding! 🚀