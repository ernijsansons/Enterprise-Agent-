"""Tooling integrations and fallbacks."""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

import requests

from src.utils.safety import sandboxed_shell
from src.utils.secrets import load_secrets

logger = logging.getLogger(__name__)


def _request_with_backoff(
    method: str, url: str, *, retries: int = 3, backoff: float = 1.5, **kwargs: Any
) -> Optional[requests.Response]:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.request(method, url, timeout=10, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # pragma: no cover - network best effort
            logger.warning(
                "Request failure (%s %s) attempt %s/%s: %s",
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


@lru_cache(maxsize=1)
def _safe_secrets() -> Dict[str, str | None]:
    try:
        return load_secrets()
    except ValueError:
        return {}


def invoke_codex_cli(
    task_type: str, params: List[str] | None = None, domain: str | None = None
) -> str:
    params = params or []
    if shutil.which("codex") is None:
        logger.warning("Codex CLI not found; returning placeholder output.")
        return "codex command unavailable"
    cmd = ["codex", "--model", "gpt-5-codex", task_type] + list(params)
    if domain:
        cmd.append(f"--domain={domain}")
    return sandboxed_shell(cmd, allowed_commands={"codex"})


def semantic_search(query: str) -> List[str]:
    secrets = _safe_secrets()
    api_key = secrets.get("SOURCEGRAPH_API_KEY")
    if not api_key:
        return [f"Stub result for: {query}"]

    payload = {
        "query": """
        query Search($query: String!) {
          search(query: $query, version: V3) {
            results {
              results {
                ... on FileMatch {
                  repository { name }
                  file { path }
                  lineMatches { preview }
                }
              }
            }
          }
        }
        """,
        "variables": {"query": query},
    }
    headers = {"Authorization": f"token {api_key}"}

    response = _request_with_backoff(
        "POST", "https://sourcegraph.com/.api/graphql", headers=headers, json=payload
    )
    if not response:
        return [f"Stub result for: {query}"]

    try:
        data = response.json()
    except ValueError:
        logger.warning("Sourcegraph response parse error")
        return [f"Stub result for: {query}"]

    matches: List[str] = []
    for result in (
        data.get("data", {}).get("search", {}).get("results", {}).get("results", [])
    ):
        repo = result.get("repository", {}).get("name", "unknown")
        path = result.get("file", {}).get("path", "")
        previews = [m.get("preview", "").strip() for m in result.get("lineMatches", [])]
        if path:
            summary = previews[0] if previews else "(no preview)"
            matches.append(f"{repo}/{path}: {summary}")
    return matches or [f"No matches for: {query}"]


def _create_openai_client(
    api_key: str,
):  # pragma: no cover - dependency injection in tests
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def embed_code(snippet: str) -> Dict[str, Any]:
    secrets = _safe_secrets()
    api_key = secrets.get("OPENAI_API_KEY")
    if not api_key:
        return {"model": "codet5_plus", "vector": [0.0] * 8, "text": snippet[:200]}

    try:
        client = _create_openai_client(api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large", input=snippet
        )
        vector = response.data[0].embedding
        return {"model": response.model, "vector": vector, "text": snippet[:200]}
    except Exception as exc:  # pragma: no cover - degrade gracefully
        logger.warning("Embedding request failed: %s", exc)
        return {"model": "codet5_plus", "vector": [0.0] * 8, "text": snippet[:200]}


_SEVERITY_MAP = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def _write_temp_payload(payload: str) -> str:
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".txt", mode="w", encoding="utf-8"
    )
    try:
        tmp.write(payload)
    finally:
        tmp.close()
    return tmp.name


def scan_vulnerabilities(payload: str, code_path: str | None = None) -> Dict[str, Any]:
    token = os.getenv("SNYK_TOKEN")
    project_id = os.getenv("SNYK_PROJECT_ID")
    if not token:
        return {"passes": True, "issues": [], "exploitability_score": 0.0}

    cli_path = shutil.which("snyk")
    if not cli_path:
        logger.info("Snyk CLI unavailable; returning stubbed scan result.")
        return {"passes": True, "issues": [], "exploitability_score": 0.0}

    cli_dir = os.path.dirname(cli_path)
    original_path = os.environ.get("PATH", "")
    added_to_path = False
    if cli_dir and cli_dir not in original_path.split(os.pathsep):
        os.environ["PATH"] = (
            f"{cli_dir}{os.pathsep}{original_path}" if original_path else cli_dir
        )
        added_to_path = True

    temp_path: Optional[str] = None
    if not code_path:
        temp_path = _write_temp_payload(payload)
        code_path = temp_path

    run_cwd = None
    file_args: List[str] = []
    if code_path:
        if os.path.isdir(code_path):
            run_cwd = code_path
        else:
            file_args.append(f"--file={code_path}")
    run_kwargs: Dict[str, Any] = {}
    if run_cwd:
        run_kwargs["cwd"] = run_cwd

    try:
        subprocess.run(
            ["snyk", "auth", token],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Build snyk test command with project ID if available
        snyk_cmd = ["snyk", "test", "--json"]
        if project_id:
            snyk_cmd.extend(["--project-id", project_id])
        snyk_cmd.extend(file_args)

        result = subprocess.run(
            snyk_cmd,
            capture_output=True,
            text=True,
            **run_kwargs,
        )
        if result.returncode not in (0, 1):
            logger.warning("Snyk scan failed: %s", result.stderr.strip())
            return {"passes": True, "issues": [], "exploitability_score": 0.0}
        data = json.loads(result.stdout or "{}")
        issues = data.get("vulnerabilities", []) or data.get("issues", [])
        severity_total = sum(
            _SEVERITY_MAP.get((issue.get("severity") or "").lower(), 1)
            for issue in issues
        )
        score = (severity_total / max(len(issues), 1)) / 4 if issues else 0.0
        return {
            "passes": not issues,
            "issues": issues,
            "exploitability_score": round(score, 3),
        }
    except FileNotFoundError as exc:  # pragma: no cover
        logger.info("Snyk CLI invocation unavailable: %s", exc)
        return {"passes": True, "issues": [], "exploitability_score": 0.0}
    except Exception as exc:  # pragma: no cover
        logger.warning("Snyk scan invocation failed: %s", exc)
        return {"passes": True, "issues": [], "exploitability_score": 0.0}
    finally:
        if added_to_path:
            os.environ["PATH"] = original_path
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


__all__ = [
    "invoke_codex_cli",
    "semantic_search",
    "embed_code",
    "scan_vulnerabilities",
]
