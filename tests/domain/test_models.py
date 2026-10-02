from datetime import date

import pytest
from pydantic import ValidationError

from order_extractor.domain.models import ExtractionResult, Order, OrderLine


def test_order_parses_llm_json():
    order = Order.model_validate_json(
        """{
            "customer_name": "Green Leaf Café",
            "customer_reference": "PO-1042",
            "requested_delivery_date": "2026-10-08",
            "delivery_address": null,
            "lines": [
                {"product_description": "red peppers", "quantity": 2, "quantity_text": "2",
                 "quantity_is_estimate": false, "unit": "box", "notes": null}
            ],
            "notes": null
        }"""
    )
    assert order.requested_delivery_date == date(2026, 10, 8)
    assert order.lines[0].quantity == 2
    assert order.lines[0].unit == "box"


def test_order_allows_empty_lines():
    # Business rules live in validation.py, so an empty order must parse.
    assert Order(lines=[]).lines == []


def test_line_allows_missing_and_non_positive_quantity():
    assert OrderLine(product_description="basil").quantity is None
    assert OrderLine(product_description="basil", quantity=0).quantity == 0


def test_line_requires_description():
    with pytest.raises(ValidationError):
        OrderLine(quantity=1)


def test_bad_date_rejected():
    with pytest.raises(ValidationError):
        Order(requested_delivery_date="next Thursday")


def test_extraction_result_allows_no_order():
    result = ExtractionResult(order=None, issues=[], is_valid=False, model="fake", latency_ms=0)
    assert result.order is None
