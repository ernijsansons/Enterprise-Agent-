from src.utils.hitl import hitl_gate


def test_hitl_low_risk_auto_approves(monkeypatch):
    monkeypatch.delenv("HITL_AUTO_APPROVE", raising=False)
    assert hitl_gate("low", "description") is True


def test_hitl_auto_flag(monkeypatch):
    monkeypatch.setenv("HITL_AUTO_APPROVE", "high")
    assert hitl_gate("high", "something") is True


def test_hitl_prompt(monkeypatch):
    monkeypatch.delenv("HITL_AUTO_APPROVE", raising=False)
    responses = iter(["y"])
    approved = hitl_gate("high", "critical", prompt_fn=lambda msg: next(responses))
    assert approved is True
    denied = hitl_gate("high", "critical", prompt_fn=lambda msg: "n")
    assert denied is False
