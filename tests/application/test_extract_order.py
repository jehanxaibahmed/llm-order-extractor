import json
from datetime import date

import pytest

from order_extractor.adapters.llm import SyntheticLLM
from order_extractor.adapters.parsing import PlainTextParser
from order_extractor.application.errors import DocumentParseError, LLMError
from order_extractor.application.extract_order import ExtractOrder, parse_order
from order_extractor.application.prompts import SYSTEM_PROMPT
from order_extractor.domain.models import Order

TODAY = date(2026, 10, 1)
EMAIL = (
    "From: Maya Patel <maya@greenleafcafe.example>\n"
    "Subject: Order PO-2231\n\n"
    "Hi, 2 boxes of red peppers and a tray of basil for Thursday please.\nMaya, Green Leaf Café"
)
GOOD = {
    "customer_name": "Green Leaf Café",
    "customer_reference": "PO-2231",
    "requested_delivery_date": "2026-10-08",
    "delivery_address": None,
    "lines": [
        {
            "product_description": "red peppers",
            "quantity": 2,
            "quantity_text": "2",
            "quantity_is_estimate": False,
            "unit": "box",
            "notes": None,
        },
        {
            "product_description": "basil",
            "quantity": 1,
            "quantity_text": "a",
            "quantity_is_estimate": False,
            "unit": "tray",
            "notes": None,
        },
    ],
    "notes": None,
}


class FakeClock:
    def __init__(self, *times):
        self._times = iter(times)

    def __call__(self):
        return next(self._times)


def use_case(*responses, **kwargs):
    llm = SyntheticLLM(*responses)
    kwargs.setdefault("clock", FakeClock(10.0, 10.25))
    return ExtractOrder(llm, **kwargs), llm


def fields(result):
    return [(i.field, i.severity) for i in result.issues]


async def test_valid_order():
    extract, llm = use_case(GOOD)
    result = await extract.from_text(EMAIL, TODAY)

    assert result.is_valid
    assert result.issues == []
    assert result.order == Order.model_validate(GOOD)
    assert result.model == "fake-llm"
    assert result.latency_ms == 250
    assert result.input_tokens and result.output_tokens

    call = llm.calls[0]
    assert call.system == SYSTEM_PROMPT
    assert "Today is Thursday, 2026-10-01." in call.user
    assert EMAIL in call.user
    assert call.schema == Order.model_json_schema()


async def test_validation_warnings_keep_order_valid():
    data = {**GOOD, "customer_name": None}
    data["lines"] = [
        {**GOOD["lines"][0], "quantity_is_estimate": True, "quantity_text": "a couple"}
    ]
    result = await use_case(data)[0].from_text(EMAIL, TODAY)
    assert result.is_valid
    assert fields(result) == [("lines[0].quantity", "warning"), ("customer_name", "warning")]


async def test_no_order_in_email_is_an_error_not_a_crash():
    data = {**GOOD, "lines": [], "customer_reference": None, "requested_delivery_date": None}
    result = await use_case(data)[0].from_text("Thanks for the lovely delivery last week!", TODAY)
    assert result.order is not None
    assert not result.is_valid
    assert fields(result) == [("lines", "error")]


async def test_invalid_json():
    result = await use_case("Sorry, I can't find an order here.")[0].from_text(EMAIL, TODAY)
    assert result.order is None
    assert not result.is_valid
    assert fields(result) == [("llm_output", "error")]
    assert result.model == "fake-llm"


async def test_json_wrapped_in_prose_is_recovered():
    raw = f"Here is the order:\n{json.dumps(GOOD)}\nLet me know if you need anything else."
    result = await use_case(raw)[0].from_text(EMAIL, TODAY)
    assert result.is_valid
    assert result.order.customer_reference == "PO-2231"


async def test_json_that_is_not_an_object():
    result = await use_case("[1, 2, 3]")[0].from_text(EMAIL, TODAY)
    assert result.order is None
    assert fields(result) == [("llm_output", "error")]


async def test_schema_mismatch_reports_each_field():
    data = {**GOOD, "requested_delivery_date": "Thursday"}
    data["lines"] = [{**GOOD["lines"][0], "quantity": "two"}, {"quantity": 1}]
    result = await use_case(data)[0].from_text(EMAIL, TODAY)
    assert result.order is None
    assert not result.is_valid
    assert fields(result) == [
        ("requested_delivery_date", "error"),
        ("lines[0].quantity", "error"),
        ("lines[1].product_description", "error"),
    ]
    assert all(i.message.startswith("Doesn't match the schema") for i in result.issues)


@pytest.mark.parametrize("text", ["", "   \n\t "])
async def test_empty_document_skips_the_llm(text):
    extract, llm = use_case(GOOD)
    result = await extract.from_text(text, TODAY)
    assert llm.calls == []
    assert result.order is None
    assert fields(result) == [("text", "error")]
    assert result.model == "fake-llm"
    assert result.latency_ms == 0


async def test_long_document_is_truncated_with_warning():
    extract, llm = use_case(GOOD, max_input_chars=100)
    result = await extract.from_text("x" * 500, TODAY)
    assert "x" * 100 in llm.calls[0].user
    assert "x" * 101 not in llm.calls[0].user
    assert fields(result) == [("text", "warning")]
    assert result.is_valid


async def test_llm_errors_propagate():
    extract, _ = use_case(LLMError("rate limited"))
    with pytest.raises(LLMError, match="rate limited"):
        await extract.from_text(EMAIL, TODAY)


async def test_from_document_parses_then_extracts():
    extract, llm = use_case(GOOD)
    result = await extract.from_document(EMAIL.encode(), PlainTextParser(), TODAY)
    assert result.is_valid
    assert "2 boxes of red peppers" in llm.calls[0].user


async def test_from_document_parse_errors_propagate():
    class BrokenParser:
        def parse(self, data: bytes) -> str:
            raise DocumentParseError("corrupt")

    extract, llm = use_case(GOOD)
    with pytest.raises(DocumentParseError):
        await extract.from_document(b"...", BrokenParser(), TODAY)
    assert llm.calls == []


def test_parse_order_on_its_own():
    order, issues = parse_order(json.dumps(GOOD))
    assert issues == []
    assert order.lines[1].unit == "tray"
