import pytest

from scripts.pdf_writer import build_pdf


@pytest.fixture
def make_pdf():
    return build_pdf
