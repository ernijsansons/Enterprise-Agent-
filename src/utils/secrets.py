"""Secrets loading with graceful fallbacks."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

try:  # Optional dependency for local development
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - acceptable fallback

    def load_dotenv(*_args, **_kwargs):
        return None


logger = logging.getLogger(__name__)

_STUB_DEFAULT = "STUBBED_FALLBACK"
_STUB_OVERRIDES = {
    "BAIDU_API_KEY": _STUB_DEFAULT,
}


def _stub_value(key: str) -> Optional[str]:
    return _STUB_OVERRIDES.get(key, _STUB_DEFAULT)


def load_secrets() -> Dict[str, Optional[str]]:
    """Load API keys from .env, substituting safe stubs when missing."""

    project_root = Path(__file__).resolve().parents[2]
    dotenv_override = os.getenv("ENTERPRISE_AGENT_DOTENV")
    dotenv_candidates = []
    if dotenv_override:
        override_path = Path(dotenv_override)
        if not override_path.is_absolute():
            override_path = project_root / override_path
        dotenv_candidates.append(override_path)
    dotenv_candidates.append(project_root / ".env")
    dotenv_candidates.append(project_root / ".env.local")
    for candidate in dotenv_candidates:
        if candidate.exists():
            load_dotenv(str(candidate))

    key_names = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "XAI_API_KEY",
        "PINECONE_API_KEY",
        "SNYK_TOKEN",
        "SNYK_PROJECT_ID",
        "SONARQUBE_TOKEN",
        "SOURCEGRAPH_API_KEY",
        "BAIDU_API_KEY",
        "FIGMA_API_KEY",
        "V0_API_KEY",
    ]
    secrets: Dict[str, Optional[str]] = {}
    for key in key_names:
        value = os.getenv(key)
        if value:
            secrets[key] = value
            continue
        stub = _stub_value(key)
        if stub:
            logger.warning("Missing %s; using stub value.", key)
            secrets[key] = stub
        else:
            logger.warning("Missing %s; continuing without value.", key)
            secrets[key] = None
    return secrets


__all__ = ["load_secrets"]
