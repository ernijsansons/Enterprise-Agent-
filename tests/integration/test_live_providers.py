import os

import pytest

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None

LIVE_TESTS_ENABLED = os.getenv("ENTERPRISE_AGENT_LIVE_TESTS") == "1"


@pytest.mark.skipif(
    not LIVE_TESTS_ENABLED,
    reason="live provider tests disabled; set ENTERPRISE_AGENT_LIVE_TESTS=1 to enable",
)
def test_openai_live_call():
    if openai is None:
        pytest.skip("openai library not installed")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    model = os.getenv("LIVE_OPENAI_MODEL", "gpt-4o-mini")
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word ok."}],
        max_tokens=5,
        temperature=0,
    )
    message = response.choices[0].message.content.strip().lower()
    assert "ok" in message


@pytest.mark.skipif(
    not LIVE_TESTS_ENABLED,
    reason="live provider tests disabled; set ENTERPRISE_AGENT_LIVE_TESTS=1 to enable",
)
def test_anthropic_live_call():
    if Anthropic is None:
        pytest.skip("anthropic library not installed")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=os.getenv("LIVE_ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
        max_tokens=32,
        messages=[{"role": "user", "content": [{"type": "text", "text": "Reply with the single word ok."}]}],
        temperature=0,
    )
    blocks = getattr(response, "content", []) or []
    text_output = "".join(getattr(block, "text", "") for block in blocks)
    assert "ok" in text_output.lower()
