import logging
from unittest.mock import patch

from src.utils.secrets import load_secrets


def test_load_secrets_missing(caplog):
    with patch("src.utils.secrets.load_dotenv"), patch.dict(
        "os.environ", {"OPENAI_API_KEY": "test"}, clear=True
    ):
        with caplog.at_level(logging.WARNING):
            secrets = load_secrets()
    assert secrets["OPENAI_API_KEY"] == "test"
    assert secrets["ANTHROPIC_API_KEY"] == "STUBBED_FALLBACK"
    assert "using stub value" in caplog.text
