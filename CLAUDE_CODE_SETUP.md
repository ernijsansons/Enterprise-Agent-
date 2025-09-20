# Claude Code CLI Integration for Enterprise Agent

## Save Money with Your Anthropic Max Subscription! 💰

If you have an Anthropic Max subscription ($200/month), you're already paying for Claude Code CLI access. This guide shows you how to use Claude Code instead of the API, **eliminating all API charges** while keeping the same functionality.

> **Note**: This guide assumes you have an active Anthropic Max subscription. If you don't have one, you can sign up at [console.anthropic.com](https://console.anthropic.com).

## The Problem

With Anthropic Max subscription:
- ❌ Using the API still costs money (on top of your $200/month subscription)
- ❌ API usage is NOT covered by your Max plan
- ❌ You end up paying twice: $200 for Max + API costs

## The Solution

Use Claude Code CLI (included in your Max subscription):
- ✅ Zero additional costs beyond your $200/month subscription
- ✅ Same Claude models (Sonnet, Opus, Haiku)
- ✅ Full Enterprise Agent functionality
- ✅ Session management for context retention

## Quick Setup

### Automated Setup (Recommended)

```bash
# Run the automated setup script
python setup_claude_code.py
```

This script will:
1. ✅ Install Claude Code CLI via npm
2. ✅ Help you log in with your Max subscription
3. ✅ Configure your environment variables
4. ✅ Create/update .env file with proper settings
5. ✅ Verify everything is working correctly

> **Prerequisites**: Make sure you have Node.js and npm installed. If not, install from [nodejs.org](https://nodejs.org/).

### Manual Setup

#### 1. Install Claude Code CLI

```bash
# Install globally via npm
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version
```

#### 2. Log in with your Max subscription

```bash
# Start authentication process
claude login
```

This opens a browser for authentication. Log in with the account that has your Max subscription.

> **Important**: Use the same account that has your Anthropic Max subscription.

#### 3. Set up long-lived token (optional but recommended)

```bash
# Create a persistent token to avoid frequent re-authentication
claude setup-token
```

> **Note**: This step is optional but recommended for automated workflows.

#### 4. Configure Environment

**IMPORTANT:** Remove or comment out `ANTHROPIC_API_KEY` to avoid API charges!

Create a `.env` file:

```env
# Use Claude Code CLI instead of API
USE_CLAUDE_CODE=true

# DO NOT SET THIS if you want to use your subscription!
# ANTHROPIC_API_KEY=sk-ant-...  # COMMENTED OUT

# Still need OpenAI for embeddings
OPENAI_API_KEY=your_openai_key_here

# Model selection
PRIMARY_MODEL=claude-3-5-sonnet-20241022
FALLBACK_MODEL=gpt-4o-mini
```

#### 5. Run the Agent

```bash
python src/agent_orchestrator.py
```

## Configuration Options

### Basic Configuration

```env
USE_CLAUDE_CODE=true                # Enable Claude Code CLI
CLAUDE_CODE_MODEL=sonnet            # Default model (sonnet/opus/haiku)
CLAUDE_CODE_TIMEOUT=60              # Timeout in seconds
```

### Advanced Configuration

```env
# Session Management (for context retention)
ENABLE_SESSIONS=true
SESSION_RETENTION_MINUTES=300

# Cache Settings (reduce redundant calls)
ENABLE_CACHE=true
CACHE_TTL_PLANNER=1800
CACHE_TTL_CODER=900

# Usage Monitoring (Max plan limits)
TRACK_USAGE=true
MAX_PROMPTS_PER_WINDOW=800         # Max 20x plan limit
WARNING_THRESHOLD=80                # Warn at 80% usage
AUTO_PAUSE_AT_LIMIT=true

# Error Handling
RETRY_ENABLED=true
MAX_RETRIES=3
EXPONENTIAL_BACKOFF=true
```

### Full Configuration

See `configs/claude_code_config.yaml` for all available options.

## Usage Examples

### Basic Usage

```python
# The agent automatically uses Claude Code when configured
from src.agent_orchestrator import AgentOrchestrator

agent = AgentOrchestrator()
result = agent.run("Analyze this codebase and suggest improvements")
```

### With Session Management

```python
# Sessions maintain context across multiple calls
agent = AgentOrchestrator()

# First call creates a session
result1 = agent.run("Analyze the authentication system", session_id="auth-review")

# Second call continues in the same context
result2 = agent.run("Now suggest security improvements", session_id="auth-review")
```

## Troubleshooting

### "Not logged in to Claude Code"

Run `claude login` and authenticate with your Max subscription account.

### "ANTHROPIC_API_KEY detected"

Remove the API key from:
- Environment variables: `unset ANTHROPIC_API_KEY`
- .env file: Comment out or remove the line
- Shell config: Remove from .bashrc/.zshrc

### "Claude Code CLI not found"

Install with npm:
```bash
npm install -g @anthropic-ai/claude-code
```

### "Command timed out"

Increase timeout in .env:
```env
CLAUDE_CODE_TIMEOUT=120  # 2 minutes
```

## Cost Comparison

### With API (Old Way)
- Max subscription: $200/month
- API usage: ~$50-200/month extra
- **Total: $250-400/month**

### With Claude Code (New Way)
- Max subscription: $200/month
- API usage: $0
- **Total: $200/month**

### Savings: $50-200+ per month! 🎉

## Limitations

Claude Code CLI has some differences from the API:

1. **Rate Limits**: Max 20x plan allows ~200-800 prompts per 5 hours
2. **No Streaming**: Response streaming is limited
3. **Session-based**: Works best with session management for context
4. **Interactive Mode**: Some features require terminal interaction

## Architecture

```
Enterprise Agent
    ├── AgentOrchestrator
    │   ├── Claude Code Provider (NEW)
    │   │   ├── CLI Wrapper
    │   │   ├── Session Manager
    │   │   └── Auth Manager
    │   └── API Provider (Fallback)
    └── Roles (Planner, Coder, Validator, etc.)
```

## Testing

Run the test suite:

```bash
# Test Claude Code provider
pytest tests/test_claude_code_provider.py

# Test authentication manager
pytest tests/test_auth_manager.py

# Run all tests
pytest
```

## Migration from API

### Gradual Migration

Set gradual rollout in .env:
```env
GRADUAL_ROLLOUT=true
CLI_PERCENTAGE=50  # Start with 50% CLI, 50% API
```

### Immediate Migration

```env
USE_CLAUDE_CODE=true
GRADUAL_ROLLOUT=false
CLI_PERCENTAGE=100
```

## Support

- **Issues**: Check the logs in `logs/` directory
- **Debug Mode**: Set `VERBOSE=true` and `DEBUG_CATEGORIES=api,hooks`
- **Help**: Run `python setup_claude_code.py` for guided setup

## Security Notes

- Never commit API keys to version control
- Use .env file for sensitive configuration
- Claude Code uses browser-based authentication (secure)
- Sessions are isolated per project

## Next Steps

1. Complete the setup using `setup_claude_code.py`
2. Verify with `claude --version` and `claude login`
3. Run a test query to ensure everything works
4. Monitor usage to stay within Max plan limits
5. Enjoy zero API costs!

## FAQ

**Q: Do I still need an OpenAI API key?**
A: Yes, for embeddings and fallback. But OpenAI costs are minimal compared to Claude.

**Q: Can I use both CLI and API?**
A: Yes, set `CLAUDE_CODE_FALLBACK_TO_API=true` for automatic fallback.

**Q: What happens if I hit the Max plan limits?**
A: The agent will pause and notify you. Limits reset every 5 hours.

**Q: Is Claude Code CLI as good as the API?**
A: Yes, it uses the same models. The only difference is the interface.

## Conclusion

By switching to Claude Code CLI, you're getting the full value of your Max subscription without paying extra for API usage. The Enterprise Agent is fully compatible and optimized for this setup.

Happy coding with zero API costs! 🚀