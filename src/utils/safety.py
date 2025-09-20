import logging
import re
import subprocess
from typing import List

PII_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

logger = logging.getLogger(__name__)


def scrub_pii(text: str) -> str:
    """Redact PII like emails."""
    return PII_PATTERN.sub("[REDACTED]", text)


def sandboxed_shell(cmd: List[str], allowed_commands: set = {"codex", "pytest"}) -> str:
    """Run shell with allow-list."""
    if not cmd:
        raise ValueError("No command provided.")
    if cmd[0] not in allowed_commands:
        raise ValueError(f"Command {cmd[0]} not allowed.")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=300
        )
        return scrub_pii(result.stdout)
    except FileNotFoundError:
        logger.warning("Command %s is not available on PATH.", cmd[0])
        return f"{cmd[0]} command unavailable"
    except subprocess.CalledProcessError as exc:
        logger.warning("Command %s failed: %s", cmd[0], exc)
        output = exc.stdout or exc.stderr or f"{cmd[0]} command failed"
        return scrub_pii(output)


# WHY: Prevents PII leaks and command injection in CLI calls while handling missing tools gracefully.
