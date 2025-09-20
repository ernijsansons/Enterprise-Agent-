import json

from src.tools import (
    embed_code,
    integrations as integrations_module,
    invoke_codex_cli,
    scan_vulnerabilities,
    semantic_search,
)


def test_invoke_codex_cli_appends_domain(monkeypatch):
    captured = {}

    def fake_shell(cmd, allowed_commands=None):  # noqa: ANN001
        captured["cmd"] = cmd
        return "ok"

    monkeypatch.setattr("src.tools.integrations.sandboxed_shell", fake_shell)
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/codex" if name == "codex" else None
    )
    result = invoke_codex_cli("auto-edit", ["--prompt", "hi"], "coding")
    assert result == "ok"
    assert captured["cmd"][-1] == "--domain=coding"


def test_invoke_codex_cli_missing_cli(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    result = invoke_codex_cli("auto-edit")
    assert result == "codex command unavailable"


def test_semantic_search_with_key(monkeypatch):
    integrations_module._safe_secrets.cache_clear()

    def fake_load():
        return {"SOURCEGRAPH_API_KEY": "token"}

    class Resp:
        def json(self):  # noqa: D401
            return {
                "data": {
                    "search": {
                        "results": {
                            "results": [
                                {
                                    "repository": {"name": "repo"},
                                    "file": {"path": "file.py"},
                                    "lineMatches": [{"preview": "print('hi')"}],
                                }
                            ]
                        }
                    }
                }
            }

    monkeypatch.setattr("src.tools.integrations.load_secrets", fake_load)
    monkeypatch.setattr(
        "src.tools.integrations._request_with_backoff", lambda *args, **kwargs: Resp()
    )
    results = semantic_search("print('hi')")
    assert results == ["repo/file.py: print('hi')"]


def test_embed_code_with_key(monkeypatch):
    integrations_module._safe_secrets.cache_clear()

    def fake_load():
        return {"OPENAI_API_KEY": "tok"}

    class FakeEmbeddings:
        @staticmethod
        def create(model: str, input: str):  # noqa: ANN001
            return type(
                "Resp",
                (),
                {
                    "data": [type("Data", (), {"embedding": [0.1, 0.2]})()],
                    "model": model,
                },
            )()

    class FakeClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr("src.tools.integrations.load_secrets", fake_load)
    monkeypatch.setattr(
        "src.tools.integrations._create_openai_client", lambda key: FakeClient()
    )
    result = embed_code("print('hi')")
    assert result["vector"] == [0.1, 0.2]


def test_scan_vulnerabilities_with_key(monkeypatch):
    integrations_module._safe_secrets.cache_clear()
    monkeypatch.setenv("SNYK_TOKEN", "tok")

    def fake_run(
        args, capture_output=False, text=False, stdout=None, stderr=None, check=False
    ):  # noqa: ANN001
        class Resp:
            def __init__(self, returncode, stdout="", stderr=""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        if args[:2] == ["snyk", "auth"]:
            return Resp(0)
        if args[:2] == ["snyk", "test"]:
            payload = {
                "vulnerabilities": [{"severity": "high", "description": "issue"}]
            }
            return Resp(1, stdout=json.dumps(payload), stderr="")
        raise AssertionError("Unexpected command")

    monkeypatch.setattr("subprocess.run", fake_run)
    report = scan_vulnerabilities("payload")
    assert report["passes"] is False
    assert report["issues"][0]["severity"] == "high"


def test_semantic_search_without_key(monkeypatch):
    integrations_module._safe_secrets.cache_clear()
    monkeypatch.setattr("src.tools.integrations.load_secrets", lambda: {})
    assert "Stub result" in semantic_search("query")[0]


def test_embed_code_without_key(monkeypatch):
    integrations_module._safe_secrets.cache_clear()
    monkeypatch.setattr("src.tools.integrations.load_secrets", lambda: {})
    result = embed_code("foo")
    assert result["model"] == "codet5_plus"


def test_scan_vulnerabilities_without_key(monkeypatch):
    integrations_module._safe_secrets.cache_clear()
    monkeypatch.delenv("SNYK_TOKEN", raising=False)
    report = scan_vulnerabilities("payload")
    assert report["issues"] == []
    assert report["exploitability_score"] == 0.0
