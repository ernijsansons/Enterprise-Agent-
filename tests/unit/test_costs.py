import pytest

from src.utils.costs import CostEstimator


def test_track_over_budget():
    estimator = CostEstimator({"budget_per_run": 0.1})
    estimator.track(10_000, "Coder", "gen", model="openai_gpt_5_codex")
    with pytest.raises(ValueError):
        estimator.track(10_000, "Reflector", "fix", model="openai_gpt_5_codex")


def test_summary_contains_events():
    estimator = CostEstimator({"budget_per_run": 1})
    estimator.track(10_000, "Planner", "plan")
    summary = estimator.summary()
    assert summary["total_tokens"] == 10_000
    assert summary["calls"] and summary["calls"][0]["role"] == "Planner"
