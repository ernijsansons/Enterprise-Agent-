# Pull Request

## Description
<!-- Provide a brief description of the changes in this PR -->

## Type of Change
<!-- Mark the relevant option with an [x] -->
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Refactoring (no functional changes, no api changes)
- [ ] ⚡ Performance improvement
- [ ] 🔒 Security improvement
- [ ] 🧪 Test improvement
- [ ] 🚀 CI/CD improvement

## Related Issues
<!-- Link to related issues using keywords like "Fixes #123" or "Relates to #456" -->
- Fixes #
- Relates to #

## Changes Made
<!-- List the main changes made in this PR -->
-
-
-

## Testing
<!-- Describe the testing performed -->
- [ ] Unit tests pass (`poetry run pytest`)
- [ ] Integration tests pass (`python test_metrics_system.py`)
- [ ] Smoke tests pass (`python smoke_test.py`)
- [ ] Code quality checks pass (`poetry run black --check . && poetry run isort --check-only . && poetry run ruff check .`)
- [ ] Type checking passes (`poetry run mypy src/`)
- [ ] Security scan passes (`poetry run bandit -r src/`)

### Test Coverage
<!-- If applicable, include test coverage information -->
- Current coverage: __%
- Coverage change: +/- __%

## Performance Impact
<!-- If applicable, describe any performance implications -->
- [ ] No performance impact expected
- [ ] Performance improvement expected
- [ ] Performance regression possible (explain below)

**Performance details:**

## Breaking Changes
<!-- If this is a breaking change, describe what breaks and how to migrate -->
- [ ] No breaking changes
- [ ] Breaking changes (describe below)

**Breaking change details:**

## Security Considerations
<!-- Describe any security implications -->
- [ ] No security implications
- [ ] Security improvement
- [ ] Potential security impact (describe below)

**Security details:**

## Configuration Changes
<!-- List any configuration changes required -->
- [ ] No configuration changes
- [ ] Configuration changes required (describe below)

**Configuration changes:**

## Documentation
<!-- Check all that apply -->
- [ ] Code is self-documenting
- [ ] Docstrings updated
- [ ] README updated
- [ ] Configuration documentation updated
- [ ] API documentation updated
- [ ] No documentation changes needed

## Deployment Notes
<!-- Any special deployment considerations -->
- [ ] No special deployment requirements
- [ ] Special deployment requirements (describe below)

**Deployment requirements:**

## Checklist
<!-- Verify all items before submitting -->
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Screenshots/Videos
<!-- If applicable, add screenshots or videos to help explain your changes -->

## Additional Context
<!-- Add any other context about the pull request here -->

---

**Reviewer Guidelines:**
- Verify all checklist items are completed
- Check that tests provide adequate coverage
- Ensure documentation is updated appropriately
- Validate security and performance implications
- Test locally if the changes are significant