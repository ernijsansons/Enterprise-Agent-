import subprocess

import pytest

from src.utils.safety import sandboxed_shell, scrub_pii


def test_scrub_pii():
    assert scrub_pii("Email: test@example.com") == "Email: [REDACTED]"


def test_scrub_pii_multiple():
    text = "Contact: john@example.com and jane@test.org for details"
    expected = "Contact: [REDACTED] and [REDACTED] for details"
    assert scrub_pii(text) == expected


def test_scrub_pii_no_emails():
    text = "This text has no email addresses"
    assert scrub_pii(text) == text


def test_sandboxed_shell_missing_command(monkeypatch):
    def fake_run(*args, **kwargs):  # noqa: ANN001
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = sandboxed_shell(["codex"], {"codex"})
    assert "codex command unavailable" in result


def test_sandboxed_shell_failure(monkeypatch):
    def fake_run(*args, **kwargs):  # noqa: ANN001
        raise subprocess.CalledProcessError(1, args[0], output="bad", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = sandboxed_shell(["pytest"], {"pytest"})
    assert "bad" in result or "err" in result


def test_sandboxed_shell_disallowed_command():
    with pytest.raises(ValueError):
        sandboxed_shell(["rm"], {"pytest"})
