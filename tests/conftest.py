import pytest

from scripts.pdf_writer import build_pdf


@pytest.fixture(autouse=True)
def no_live_api_keys(monkeypatch):
    """Tests must never reach a real LLM, even if a key is set in the developer's shell."""
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def make_pdf():
    return build_pdf
