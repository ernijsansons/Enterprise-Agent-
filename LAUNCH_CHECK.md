# 🚀 Enterprise Agent Launch Check Guide

## Quick Start

Always activate the virtual environment first, then run the launch check:

```powershell
# Activate the Python 3.13 virtual environment
.\.venv\Scripts\Activate

# Run the launch readiness check
.\check_launch.ps1
```

Or as a one-liner:
```powershell
.\.venv\Scripts\Activate; .\check_launch.ps1
```

## Why This Matters

- **Python Version**: Ensures Python 3.13 is used (not system Python 3.12)
- **Dependencies**: Uses pytest, pyyaml, and all packages from the virtual environment
- **Snyk Integration**: Works with your project ID `39fe3efb-d65a-4f29-a662-7659fffc4476`
- **Security**: All vulnerabilities resolved (bandit 1.7.7, anyio 4.10.0)

## Current Status ✅

- **Tests**: 30/30 passing
- **Security**: No vulnerable paths found
- **Code Quality**: Black, Flake8, isort all clean
- **Dependencies**: 79 packages tested successfully
- **Snyk**: Fully integrated and working

## Remember

If you open a new PowerShell window, always run the activation step first:
```powershell
.\.venv\Scripts\Activate
```

The `(.venv)` prefix in your prompt confirms the virtual environment is active.
