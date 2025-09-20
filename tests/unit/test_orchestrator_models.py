from types import SimpleNamespace

from src.agent_orchestrator import AgentOrchestrator
from src.utils.costs import CostEstimator


class DummyAnthropicClient:
    def __init__(self) -> None:
        self.last_kwargs = None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):  # noqa: ANN001
        self.last_kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(text="response")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )


class DummyOrchestrator:
    def __init__(self) -> None:
        self.cost_estimator = CostEstimator({})
        self.anthropic_client = DummyAnthropicClient()
        self.openai_client = None
        self.gemini_client = None
        # Add mock for new cache attribute
        self._model_cache = SimpleNamespace(
            get_response=lambda **kwargs: None,  # Return None to simulate cache miss
            cache_response=lambda **kwargs: None
        )
        # Add Claude Code attributes
        self._use_claude_code = False
        self._claude_code_provider = None

    def _enhance_prompt(self, prompt, role):  # noqa: ANN001
        return prompt

    def _require_model(self, model, stage):  # noqa: ANN001
        return model


def test_call_model_formats_anthropic_messages():
    orchestrator = DummyOrchestrator()
    result = AgentOrchestrator._call_model(
        orchestrator,
        model="claude_sonnet_4",
        prompt="hello",
        role="Planner",
        operation="decompose",
    )
    kwargs = orchestrator.anthropic_client.last_kwargs
    assert kwargs is not None
    assert kwargs["messages"][0]["content"][0] == {"type": "text", "text": "hello"}
    assert result == "response"
