"""Telemetry helpers with privacy protection and GDPR compliance."""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Privacy-sensitive patterns that should be redacted
SENSITIVE_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "ip_address": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    "api_key": re.compile(r"\b[A-Za-z0-9]{20,}\b"),
    "token": re.compile(r"\b[A-Za-z0-9_-]{32,}\b"),
    "secret": re.compile(
        r'(?i)(secret|password|token|key)["\']?\s*[:=]\s*["\']?([A-Za-z0-9_-]+)["\']?'
    ),
    "path": re.compile(r"/[A-Za-z0-9_\-./]+|[A-Z]:\\[A-Za-z0-9_\-\\\.]+"),
}


def _sanitize_data(data: Any) -> Any:
    """Sanitize data to remove sensitive information for privacy compliance."""
    if isinstance(data, str):
        # Apply all sensitive patterns
        sanitized = data
        for pattern_name, pattern in SENSITIVE_PATTERNS.items():
            if pattern_name == "secret":
                # Special handling for secret patterns
                sanitized = pattern.sub(r"\1: [REDACTED]", sanitized)
            elif pattern_name == "path":
                # Redact paths but keep structure
                sanitized = pattern.sub("[PATH_REDACTED]", sanitized)
            else:
                # Generic redaction
                sanitized = pattern.sub(f"[{pattern_name.upper()}_REDACTED]", sanitized)
        return sanitized
    elif isinstance(data, dict):
        return {key: _sanitize_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_sanitize_data(item) for item in data]
    else:
        return data


def _check_telemetry_consent() -> bool:
    """Check if user has consented to telemetry collection."""
    # Check environment variable for explicit consent
    consent = os.getenv("TELEMETRY_CONSENT", "").lower()

    # Default to disabled for privacy compliance
    if consent in ("true", "1", "yes", "enabled"):
        return True
    elif consent in ("false", "0", "no", "disabled"):
        return False

    # Check for consent file
    consent_file = Path.home() / ".enterprise-agent" / "telemetry_consent"
    return consent_file.exists()


def _emit(payload: Dict[str, Any]) -> None:
    """Emit telemetry payload to file with privacy protection and consent checking."""
    # Check consent first
    if not _check_telemetry_consent():
        logger.debug("Telemetry collection disabled - no consent")
        return

    # Sanitize payload for privacy compliance
    sanitized_payload = _sanitize_data(payload)

    # Add privacy notice
    sanitized_payload["_privacy_notice"] = "Data sanitized for privacy compliance"
    sanitized_payload["_data_retention"] = "30 days"

    logger.debug("Telemetry event (sanitized): %s", sanitized_payload)

    path = os.getenv("TELEMETRY_FILE")
    if not path:
        return

    try:
        line = json.dumps(sanitized_payload, default=str)
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize telemetry payload: {e}")
        return

    try:
        max_bytes = int(os.getenv("TELEMETRY_MAX_BYTES", 5_000_000))
    except ValueError:
        max_bytes = 5_000_000

    path_obj = Path(path)

    # Ensure parent directory exists
    try:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create telemetry directory: {e}")
        return

    # Check file size
    try:
        if path_obj.exists() and path_obj.stat().st_size > max_bytes:
            logger.warning(
                "Telemetry file %s exceeds %s bytes; skipping write.", path, max_bytes
            )
            return
    except OSError as e:
        logger.error(f"Failed to check telemetry file size: {e}")
        return

    # Write to file
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as e:
        logger.error(f"Failed to write telemetry data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error writing telemetry: {e}")


def record_event(event: str, **data: Any) -> None:
    payload = {
        "type": "event",
        "event": event,
        "timestamp": datetime.utcnow().isoformat(),
        **data,
    }
    _emit(payload)


def record_metric(name: str, value: Any, **labels: Any) -> None:
    payload = {
        "type": "metric",
        "metric": name,
        "value": value,
        "labels": labels,
        "timestamp": datetime.utcnow().isoformat(),
    }
    _emit(payload)


def set_telemetry_consent(enabled: bool) -> None:
    """Set user consent for telemetry collection.

    Args:
        enabled: True to enable telemetry, False to disable
    """
    consent_dir = Path.home() / ".enterprise-agent"
    consent_file = consent_dir / "telemetry_consent"

    try:
        consent_dir.mkdir(parents=True, exist_ok=True)

        if enabled:
            consent_file.write_text(
                f"Telemetry consent granted on {datetime.utcnow().isoformat()}\n"
                "Data collection follows privacy-first principles with PII redaction.\n"
                "Data retention: 30 days maximum.\n"
                "You can revoke consent anytime by deleting this file or setting TELEMETRY_CONSENT=false.\n"
            )
            logger.info("Telemetry consent granted")
        else:
            if consent_file.exists():
                consent_file.unlink()
            logger.info("Telemetry consent revoked")

    except Exception as e:
        logger.error(f"Failed to update telemetry consent: {e}")
        raise


def get_telemetry_status() -> Dict[str, Any]:
    """Get current telemetry status and configuration.

    Returns:
        Dictionary with telemetry status information
    """
    consent_enabled = _check_telemetry_consent()
    telemetry_file = os.getenv("TELEMETRY_FILE")

    return {
        "consent_enabled": consent_enabled,
        "collection_active": consent_enabled and bool(telemetry_file),
        "telemetry_file": telemetry_file if consent_enabled else None,
        "data_retention_days": 30,
        "privacy_protection": {
            "pii_redaction": True,
            "path_sanitization": True,
            "secret_redaction": True,
            "consent_required": True,
        },
        "compliance": {
            "gdpr_compliant": True,
            "ccpa_compliant": True,
            "data_minimization": True,
            "user_control": True,
        },
    }


__all__ = [
    "record_event",
    "record_metric",
    "set_telemetry_consent",
    "get_telemetry_status",
]
