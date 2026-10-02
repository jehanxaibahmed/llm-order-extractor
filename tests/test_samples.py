"""Keep the sample set and its ground truth consistent with the code and with each other."""

import json
import re
from datetime import date
from pathlib import Path

import pytest

from order_extractor.adapters.parsing import parse_file
from order_extractor.domain.models import Order
from order_extractor.domain.validation import validate_order
from scripts.make_sample_pdfs import PDFS
from scripts.pdf_writer import build_pdf

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
EXPECTED = sorted((SAMPLES / "expected").glob("*.json"))
SOURCES = sorted(p for d in ("emails", "pdfs") for p in (SAMPLES / d).iterdir())


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_every_sample_has_expected_output():
    assert len(SOURCES) >= 10
    assert {p.stem for p in SOURCES} == {p.stem for p in EXPECTED}


@pytest.mark.parametrize("path", EXPECTED, ids=lambda p: p.stem)
def test_expected_issues_match_validation_rules(path):
    expected = load(path)
    order = Order.model_validate(expected["order"])
    issues = validate_order(order, date.fromisoformat(expected["today"]))
    assert [{"field": i.field, "severity": i.severity} for i in issues] == expected["issues"]


@pytest.mark.parametrize("path", EXPECTED, ids=lambda p: p.stem)
def test_expected_values_appear_in_the_document(path):
    """Ground truth may only contain what the document says (the prompt's 'never invent' rule)."""
    expected = load(path)
    text = " ".join(parse_file(SAMPLES / expected["source"]).casefold().split())
    order = expected["order"]
    for line in order["lines"]:
        assert line["product_description"].casefold() in text
        assert line["quantity_text"].casefold() in text, line
    for field in ("customer_name", "customer_reference"):
        if order[field]:
            assert order[field].casefold() in text, field


def test_sample_pdfs_match_generator():
    for name, pages in PDFS.items():
        assert (SAMPLES / "pdfs" / name).read_bytes() == build_pdf(pages), (
            f"{name} is stale: run python -m scripts.make_sample_pdfs"
        )


def test_samples_use_only_reserved_example_domains():
    """Data rule: synthetic data only. Real-looking email addresses would be a red flag."""
    for path in SOURCES + EXPECTED:
        text = parse_file(path) if path.suffix != ".json" else path.read_text()
        for domain in re.findall(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)", text):
            assert domain.endswith(".example"), f"{path.name}: {domain}"
