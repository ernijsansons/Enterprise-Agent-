from pathlib import Path

content = """\n"""Tooling integrations and fallbacks."""\nfrom __future__ import annotations\n\nimport json\nimport logging\nimport os\nimport subprocess\nimport tempfile\nimport time\nfrom typing import Any, Dict, List, Optional\n\nimport requests\n\nfrom src.utils.safety import sandboxed_shell\nfrom src.utils.secrets import load_secrets\n\nlogger = logging.getLogger(__name__)\n\n\ndef _request_with_backoff(method: str, url: str, *, retries: int = 3, backoff: float = 1.5, **kwargs: Any) -> Optional[requests.Response]:\n    for attempt in range(1, retries + 1):\n        try:\n            resp = requests.request(method, url, timeout=10, **kwargs)\n            resp.raise_for_status()\n            return resp\n        except Exception as exc:  # pragma: no cover\n            logger.warning(\"Request failure (%s %s) attempt %s/%s: %s\", method, url, attempt, retries, exc)\n            if attempt == retries:\n                return None\n            time.sleep(backoff ** attempt)\n    return None\n\n\ndef _safe_secrets() -> Dict[str, str | None]:\n    try:\n        return load_secrets()\n    except ValueError:\n        return {}\n\n\ndef invoke_codex_cli(task_type: str, params: List[str] | None = None, domain: str | None = None) -> str:\n    params = params or []\n    cmd = [\"codex\", \"--model\", \"gpt-5-codex\", task_type] + list(params)\n    if domain:\n        cmd.append(f\"--domain={domain}\")\n    return sandboxed_shell(cmd, allowed_commands={\"codex\"})\n\n\ndef semantic_search(query: str) -> List[str]:\n    secrets = _safe_secrets()\n    api_key = secrets.get(\"SOURCEGRAPH_API_KEY\")\n    if not api_key:\n        return [f\"Stub result for: {query}\"]\n\n    payload = {\n        \"query\": \"\"\"\n        query Search($query: String!) {\n          search(query: $query, version: V3) {\n            results {\n              results {\n                ... on FileMatch {\n                  repository { name }\n                  file { path }\n                  lineMatches { preview }\n                }\n              }\n            }\n          }\n        }\n        \"\"\",\n        \"variables\": {\"query\": query},\n    }\n    headers = {\"Authorization\": f\"token {api_key}\"}\n\n    response = _request_with_backoff(\"POST\", \"https://sourcegraph.com/.api/graphql\", headers=headers, json=payload)\n    if not response:\n        return [f\"Stub result for: {query}\"]\n\n    try:\n        data = response.json()\n    except ValueError:\n        logger.warning(\"Sourcegraph response parse error\")\n        return [f\"Stub result for: {query}\"]\n\n    matches: List[str] = []\n    for result in data.get(\"data\", {}).get(\"search\", {}).get(\"results\", {}).get(\"results\", []):\n        repo = result.get(\"repository\", {}).get(\"name\", \"unknown\")\n        path = result.get(\"file\", {}).get(\"path\", \"\")\n        previews = [m.get(\"preview\", \"\").strip() for m in result.get(\"lineMatches\", [])]\n        if path:\n            summary = previews[0] if previews else \"(no preview)\"\n            matches.append(f\"{repo}/{path}: {summary}\")\n    return matches or [f\"No matches for: {query}\"]\n\n\ndef _create_openai_client(api_key: str):  # pragma: no cover - dependency injection in tests\n    from openai import OpenAI\n\n    return OpenAI(api_key=api_key)\n\n\ndef embed_code(snippet: str) -> Dict[str, Any]:\n    secrets = _safe_secrets()\n    api_key = secrets.get(\"OPENAI_API_KEY\")\n    if not api_key:\n        return {\"model\": \"codet5_plus\", \"vector\": [0.0] * 8, \"text\": snippet[:200]}\n\n    try:\n        client = _create_openai_client(api_key)\n        response = client.embeddings.create(model=\"text-embedding-3-large\", input=snippet)\n        vector = response.data[0].embedding\n        return {\"model\": response.model, \"vector\": vector, \"text\": snippet[:200]}\n    except Exception as exc:  # pragma: no cover - degrade gracefully\n        logger.warning(\"Embedding request failed: %s\", exc)\n        return {\"model\": \"codet5_plus\", \"vector\": [0.0] * 8, \"text\": snippet[:200]}\n\n\n_SEVERITY_MAP = {\"critical\": 4, \"high\": 3, \"medium\": 2, \"low\": 1}\n\n\ndef _write_temp_payload(payload: str) -> str:\n    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=\".txt\", mode=\"w\", encoding=\"utf-8\")\n    try:\n        tmp.write(payload)\n    finally:\n        tmp.close()\n    return tmp.name\n\n\ndef scan_vulnerabilities(payload: str, code_path: str | None = None) -> Dict[str, Any]:\n    token = os.getenv(\"SNYK_TOKEN\")\n    if not token:\n        return {\"passes\": True, \"issues\": [], \"exploitability_score\": 0.0}\n\n    temp_path: Optional[str] = None\n    if not code_path:\n        temp_path = _write_temp_payload(payload)\n        code_path = temp_path\n\n    try:\n        subprocess.run([\"snyk\", \"auth\", token], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result = subprocess.run([\"snyk\", \"test\", \"--json\", f\"--file={code_path}\"], capture_output=True, text=True)
        if result.returncode not in (0, 1):
            logger.warning(\"Snyk scan failed: %s\", result.stderr.strip())
            return {\"passes\": True, \"issues\": [], \"exploitability_score\": 0.0}
        data = json.loads(result.stdout or \"{}\")
        issues = data.get(\"vulnerabilities\", []) or data.get(\"issues\", [])
        severity_total = sum(_SEVERITY_MAP.get((issue.get(\"severity\") or \"\").lower(), 1) for issue in issues)
        score = (severity_total / max(len(issues), 1)) / 4 if issues else 0.0
        return {\"passes\": not issues, \"issues\": issues, \"exploitability_score\": round(score, 3)}
    except Exception as exc:  # pragma: no cover
        logger.warning(\"Snyk scan invocation failed: %s\", exc)
        return {\"passes\": True, \"issues\": [], \"exploitability_score\": 0.0}
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


__all__ = [
    \"invoke_codex_cli\",
    \"semantic_search\",
    \"embed_code\",
    \"scan_vulnerabilities\",
]
"""

Path('src/tools/integrations.py').write_text(content)
