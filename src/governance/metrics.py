"""Governance metrics and decision helpers."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Optional

import requests

from src.utils.hitl import hitl_gate

logger = logging.getLogger(__name__)

MetricFetcher = Callable[[str, Dict[str, Any]], Dict[str, float]]
HITLCallback = Callable[[str, Dict[str, Any]], bool]


def _request_with_backoff(
    method: str, url: str, *, retries: int = 3, backoff: float = 1.5, **kwargs: Any
) -> Optional[requests.Response]:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.request(method, url, timeout=10, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Governance request failure (%s %s) attempt %s/%s: %s",
                method,
                url,
                attempt,
                retries,
                exc,
            )
            if attempt == retries:
                return None
            time.sleep(backoff**attempt)
    return None


def _fetch_sonarqube_metrics(
    config: Dict[str, Any], component: str
) -> Dict[str, float]:
    token = os.getenv("SONARQUBE_TOKEN")
    url = config.get("url")
    if not token or not url:
        return {}
    metrics = config.get("metric_keys", "bugs,complexity,maintainability_rating")
    params = {"component": component, "metricKeys": metrics}
    response = _request_with_backoff(
        "GET", f"{url}/api/measures/component", params=params, auth=(token, "")
    )
    if not response:
        return {}
    try:
        data = response.json()
    except ValueError:
        logger.warning("SonarQube response parse error")
        return {}
    measures = {
        m["metric"]: float(m.get("value", 0))
        for m in data.get("component", {}).get("measures", [])
    }
    bug_rate = measures.get("bugs", 0) / max(measures.get("complexity", 1), 1)
    maintainability = measures.get("maintainability_rating", 1)
    return {
        "bug_rate": bug_rate,
        "complexity": measures.get("complexity", 0),
        "maintainability": maintainability,
    }


def _default_metric_fetcher(
    domain: str, result: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, float]:
    if domain == "coding":
        component = (
            result.get("component")
            or config.get("sonarqube", {}).get("component")
            or "default"
        )
        sonar_metrics = _fetch_sonarqube_metrics(config.get("sonarqube", {}), component)
        if sonar_metrics:
            return sonar_metrics
        return {"bug_rate": 0.2, "complexity": 15, "maintainability": 90}
    if domain == "trading":
        return {"bug_rate": 0.1, "risk_score": 0.8}
    return {"bug_rate": 0.0}


class GovernanceChecker:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        metric_fetcher: Optional[MetricFetcher] = None,
        hitl_callback: HITLCallback = hitl_gate,
    ) -> None:
        self.config = config or {}
        self.thresholds = self.config.get(
            "rebuild_thresholds",
            {"bug_rate": 0.3, "complexity": 50, "maintainability": 70},
        )
        self.metric_fetcher = metric_fetcher or (
            lambda domain, result: _default_metric_fetcher(domain, result, self.config)
        )
        self.hitl_callback = hitl_callback
        self.latest_metrics: Dict[str, Any] = {}

    def check(self, result: Dict[str, Any]) -> bool:
        domain = result.get("domain", "")
        metrics = self.metric_fetcher(domain, result)
        self.latest_metrics = {"domain": domain, **metrics}

        if metrics.get("bug_rate", 0) > self.thresholds.get("bug_rate", 1):
            self._trigger_action("fix_targeted", result)
            return False
        if metrics.get("complexity", 0) > self.thresholds.get("complexity", 100):
            self._trigger_action("delete_refactor_parts", result)
            return False
        if metrics.get("maintainability", 100) < self.thresholds.get(
            "maintainability", 0
        ):
            self._trigger_action("rebuild_from_scratch", result)
            return False
        return True

    def _trigger_action(self, action: str, context: Dict[str, Any]) -> None:
        logger.info(
            "Governance triggered action %s for domain %s",
            action,
            context.get("domain"),
        )

    def hitl_check(
        self, risk_level: str, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        context = context or {}
        return self.hitl_callback(risk_level, context)


__all__ = ["GovernanceChecker"]
