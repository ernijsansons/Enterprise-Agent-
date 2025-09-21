"""Telemetry helpers."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _emit(payload: Dict[str, Any]) -> None:
    """Emit telemetry payload to file with enhanced error handling."""
    logger.debug("Telemetry event: %s", payload)
    path = os.getenv("TELEMETRY_FILE")
    if not path:
        return

    try:
        line = json.dumps(payload, default=str)
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


__all__ = ["record_event", "record_metric"]
