"""Human-in-the-loop gating utilities."""
from __future__ import annotations

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)

PROMPT_FN = Callable[[str], str]


def hitl_gate(
    risk_level: str,
    description: str,
    *,
    prompt_fn: PROMPT_FN | None = None,
) -> bool:
    """Return True if the action is approved based on risk level and configuration."""

    risk_level = (risk_level or "low").lower()
    description = description or ""
    auto_flag = (os.getenv("HITL_AUTO_APPROVE", "").lower()).split(",")
    auto_levels = {flag.strip() for flag in auto_flag if flag.strip()}

    if risk_level == "low":
        logger.info("HITL auto-approved low risk: %s", description)
        return True

    if "all" in auto_levels or risk_level in auto_levels:
        logger.info("HITL auto-approved via flag (%s): %s", risk_level, description)
        return True

    prompt = prompt_fn or (lambda msg: input(msg))
    response = (
        prompt(f"Approve {description or 'operation'} (risk={risk_level})? [y/N]: ")
        .strip()
        .lower()
    )
    approved = response in {"y", "yes"}
    logger.info(
        "HITL decision for %s: %s", description, "approved" if approved else "denied"
    )
    return approved


__all__ = ["hitl_gate"]
