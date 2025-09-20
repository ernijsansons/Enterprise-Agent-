from src.agent_orchestrator import AgentOrchestrator


def test_config_override_via_env(tmp_path, monkeypatch):
    config_path = tmp_path / "custom_config.yaml"
    config_path.write_text(
        """enterprise_coding_agent:
  orchestration: {}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ENTERPRISE_AGENT_CONFIG", str(config_path))
    monkeypatch.delenv("ENTERPRISE_AGENT_DOTENV", raising=False)

    orchestrator = AgentOrchestrator()

    assert orchestrator.config.get("enterprise_coding_agent") is not None


def test_dotenv_override_precedence(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        """OPENAI_API_KEY=override-key
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ENTERPRISE_AGENT_DOTENV", str(dotenv_path))
    monkeypatch.delenv("ENTERPRISE_AGENT_CONFIG", raising=False)

    orchestrator = AgentOrchestrator()

    assert orchestrator.secrets.get("OPENAI_API_KEY") == "override-key"
