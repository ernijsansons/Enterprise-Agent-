# Security Policy

## Reporting Vulnerabilities
Report security issues to: security@example.com

## Security Best Practices

### Secrets Management
- **Never commit secrets**: API keys, tokens, or credentials must never be in code
- **Use environment variables**: Store sensitive data in `.env` files (never commit these)
- **Reference .env.example**: Document required variables without exposing values
- **Credential rotation**: Rotate API keys and tokens regularly

### Claude API Security
- **Subscription mode preferred**: Use Claude Code CLI with Max subscription to avoid API charges
- **API key protection**: If using API mode, store `ANTHROPIC_API_KEY` securely
- **Rate limiting**: Implement timeouts and retries to prevent abuse
- **Cost monitoring**: Track token usage through the CostEstimator module

### Data Protection
- **PII scrubbing**: All logs automatically scrub personally identifiable information
- **Retention policy**: Logs retained for 30 days by default
- **Secure storage**: Use encrypted storage for sensitive data
- **Access control**: Limit repository access to authorized personnel only

### Code Security
- **Dependency scanning**: Regularly update dependencies with `poetry update`
- **Static analysis**: Run `bandit` for security vulnerability detection
- **Code review**: All PRs require review before merging to main
- **Testing**: Maintain >95% code coverage with security-focused tests

### CI/CD Security
- **GitHub Secrets**: Use GitHub Secrets for CI/CD credentials
- **Protected branches**: Enable branch protection rules for main
- **Signed commits**: Consider requiring GPG-signed commits
- **Audit logging**: Enable GitHub audit logs for security events
