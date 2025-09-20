import json
from typing import Any, Dict, List

import pytest

from src.roles import Coder, Planner, Reflector, Reviewer, Validator


class _Memory:
    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Any]] = {}

    def store(self, scope: str, key: str, value: Any) -> None:
        self.data.setdefault(scope, {})[key] = value


class StubOrchestrator:
    def __init__(self) -> None:
        self.domain_packs = {"coding": {"validators": ["coverage >=0.97"]}}
        self.memory = _Memory()
        self.route_models: List[str] = []
        self.responses: List[str] = []
        self.cli_result = "cli-output"
        self.anthropic_client = None

    def route_to_model(self, text: str, domain: str, vuln_flag: bool = False) -> str:
        if self.route_models:
            return self.route_models.pop(0)
        return "stub-model"

    def _call_model(self, model: str, prompt: str, role: str, operation: str) -> str:
        if not self.responses:
            raise AssertionError("No stub response set for _call_model")
        return self.responses.pop(0)

    def _invoke_codex_cli(self, task_type: str, params: List[str], domain: str) -> str:
        return self.cli_result

    def _parse_json(self, text: str) -> Dict[str, Any]:
        return json.loads(text)

    def telemetry(self, event: Dict[str, Any]) -> None:  # pragma: no cover
        pass


@pytest.fixture()
def stub() -> StubOrchestrator:
    return StubOrchestrator()


def test_planner_decompose_records_epics(stub: StubOrchestrator) -> None:
    stub.responses = ["Step 1\nStep 2"]
    planner = Planner(stub)
    result = planner.decompose("Build thing", "coding")
    assert result["epics"] == ["Step 1", "Step 2"]
    assert stub.memory.data["session"]["plan"] == "Step 1\nStep 2"


def test_coder_uses_cli_for_codex(stub: StubOrchestrator) -> None:
    stub.route_models = ["openai_gpt_5_codex"]
    stub.cli_result = "cli-result"
    coder = Coder(stub)
    result = coder.generate("Plan", "coding")
    assert result["output"] == "cli-result"
    assert result["source"] == "codex_cli"


def test_validator_uses_domain_stub(stub: StubOrchestrator) -> None:
    validator = Validator(stub)
    result = validator.validate("output", "coding")
    assert result["parsed"]["passes"] is True
    assert result["parsed"]["coverage"] == pytest.approx(0.97)
    assert "reason" in result["parsed"]


def test_reflector_updates_output(stub: StubOrchestrator) -> None:
    stub.responses = ["improved output"]
    reflector = Reflector(stub)
    result = reflector.reflect({"passes": False}, "old", "coding", iterations=0)
    assert result["output"] == "improved output"
    assert result["iterations"] == 1


def test_reviewer_aggregates_scores(stub: StubOrchestrator) -> None:
    stub.route_models = ["primary-model"]
    stub.responses = ["0.9", "0.7"]
    stub.anthropic_client = object()
    reviewer = Reviewer(stub)
    result = reviewer.review("output", "coding")
    assert result["confidence"] == pytest.approx(0.8)
    assert result["scores"] == [0.9, 0.7]
