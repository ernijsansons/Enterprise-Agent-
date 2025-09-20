import pytest

from src.governance import GovernanceChecker


def test_governance_passes_within_thresholds():
    def fetcher(domain, result):
        return {
            "bug_rate": 0.2,
            "complexity": 10,
            "maintainability": 90,
        }

    checker = GovernanceChecker(
        {
            "rebuild_thresholds": {
                "bug_rate": 0.3,
                "complexity": 20,
                "maintainability": 80,
            }
        },
        metric_fetcher=fetcher,
    )
    assert checker.check({"domain": "coding"}) is True
    assert checker.latest_metrics["bug_rate"] == pytest.approx(0.2)


def test_governance_triggers_action_when_bug_rate_high():
    def metrics(domain, result):
        return {"bug_rate": 0.5}

    actions = []

    class CaptureGovernance(GovernanceChecker):
        def _trigger_action(self, action, context):  # noqa: ANN001
            actions.append(action)

    checker = CaptureGovernance(
        {"rebuild_thresholds": {"bug_rate": 0.3}}, metric_fetcher=metrics
    )
    assert checker.check({"domain": "coding"}) is False
    assert actions == ["fix_targeted"]


def test_hitl_callback_invoked():
    called = {}

    def callback(risk, context):  # noqa: ANN001
        called["risk"] = risk
        called["context"] = context
        return False

    checker = GovernanceChecker(
        {}, metric_fetcher=lambda *_: {}, hitl_callback=callback
    )
    assert checker.hitl_check("high", {"domain": "coding"}) is False
    assert called["risk"] == "high"
