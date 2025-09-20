import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import after path setup
import src.agent_orchestrator as orchestrator_module  # noqa: E402
import src.roles.validators as validators  # noqa: E402
from src.agent_orchestrator import AgentOrchestrator  # noqa: E402

if "langgraph.graph" not in sys.modules:

    class _FakeStateGraph:
        def __init__(self, state_cls):
            self.nodes = {}
            self.edges = {}
            self.cond = {}

        def add_node(self, name, func):
            self.nodes[name] = func

        def add_edge(self, start, end):
            self.edges[start] = end

        def add_conditional_edges(self, name, func):
            self.cond[name] = func

        def compile(self):
            nodes = self.nodes
            edges = self.edges
            cond = self.cond

            class Runner:
                def invoke(self, state):
                    current = "planner"
                    while True:
                        state = nodes[current](state)
                        if current in cond:
                            current = cond[current](state)
                        else:
                            current = edges.get(current)
                        if current == "governance":
                            state = nodes[current](state)
                            current = edges.get(current)
                        if current is None or current == "END":
                            return state
                    return state

            return Runner()

    fake_module = types.ModuleType("langgraph.graph")
    fake_module.StateGraph = _FakeStateGraph
    fake_module.END = "END"
    sys.modules["langgraph"] = types.ModuleType("langgraph")
    sys.modules["langgraph"].graph = fake_module
    sys.modules["langgraph.graph"] = fake_module


@pytest.fixture()
def patched_agent(monkeypatch):
    monkeypatch.setattr(
        orchestrator_module,
        "load_secrets",
        lambda: {
            k: "dummy"
            for k in [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GOOGLE_API_KEY",
                "XAI_API_KEY",
                "PINECONE_API_KEY",
                "SNYK_TOKEN",
                "SONARQUBE_TOKEN",
                "SOURCEGRAPH_API_KEY",
                "BAIDU_API_KEY",
            ]
        },
    )
    monkeypatch.setattr(AgentOrchestrator, "_init_clients", lambda self: None)
    monkeypatch.setattr(
        AgentOrchestrator,
        "_invoke_codex_cli",
        lambda self, *args, **kwargs: "generated via codex",
    )

    def fake_call(self, model, prompt, role, op, max_tokens=8192):
        if role == "Validator":
            return '{"passes": true, "coverage": 0.99}'
        if role == "Reflector":
            return json.dumps(
                {
                    "analysis": "analysis",
                    "fixes": [{"description": "fix", "risks": "low"}],
                    "selected_fix": 0,
                    "revised_output": "fixed output",
                    "confidence": 0.85,
                }
            )
        if role == "Reviewer":
            return '{"score": 0.9, "rationale": "looks good"}'
        return "result"

    monkeypatch.setattr(AgentOrchestrator, "_call_model", fake_call)
    monkeypatch.setattr(
        AgentOrchestrator, "route_to_model", lambda self, *args, **kwargs: "stub-model"
    )

    monkeypatch.setattr(
        validators,
        "validate_coding",
        lambda payload: {
            "passes": True,
            "tests_passed": True,
            "coverage": 0.99,
            "coverage_threshold": payload.get("coverage_threshold", 0.97),
            "reason": "stubbed",
        },
    )
    monkeypatch.setattr(
        validators, "validate_social_media", lambda payload: {"passes": True}
    )
    monkeypatch.setattr(
        validators, "validate_trading", lambda payload: {"passes": True}
    )
    monkeypatch.setattr(
        validators, "validate_content", lambda payload: {"passes": True}
    )
    monkeypatch.setattr(
        validators, "validate_real_estate", lambda payload: {"passes": True}
    )

    return AgentOrchestrator()


def test_smoke_runs_through_pipeline(patched_agent):
    result = patched_agent.run_mode("coding", "Smoke test")
    assert result["confidence"] >= 0
    assert "cost_summary" in result
    assert result["plan"]


@pytest.mark.parametrize("domain", ["social_media", "trading"])
def test_smoke_other_domains(patched_agent, domain):
    result = patched_agent.run_mode(domain, "Smoke test")
    assert result["domain"] == domain
