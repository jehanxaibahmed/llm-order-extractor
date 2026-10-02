from datetime import date, timedelta

from order_extractor.domain.models import Order, OrderLine
from order_extractor.domain.validation import is_valid, validate_order

TODAY = date(2026, 10, 1)


def line(description="red peppers", quantity=2, **kwargs):
    return OrderLine(product_description=description, quantity=quantity, **kwargs)


def order(lines=None, customer_name="Green Leaf Café", **kwargs):
    return Order(customer_name=customer_name, lines=[line()] if lines is None else lines, **kwargs)


def issues_for(o):
    return [(i.field, i.severity) for i in validate_order(o, TODAY)]


def test_clean_order_has_no_issues():
    o = order(requested_delivery_date=TODAY + timedelta(days=3))
    assert validate_order(o, TODAY) == []
    assert is_valid([])


def test_no_lines_is_error():
    issues = validate_order(order(lines=[]), TODAY)
    assert [(i.field, i.severity) for i in issues] == [("lines", "error")]
    assert not is_valid(issues)


def test_missing_quantity_is_error():
    assert issues_for(order(lines=[line(quantity=None)])) == [("lines[0].quantity", "error")]


def test_zero_and_negative_quantity_are_errors():
    assert issues_for(order(lines=[line(quantity=0)])) == [("lines[0].quantity", "error")]
    assert issues_for(order(lines=[line(quantity=-3)])) == [("lines[0].quantity", "error")]


def test_large_quantity_is_warning():
    assert issues_for(order(lines=[line(quantity=500)])) == []
    issues = validate_order(order(lines=[line(quantity=501)]), TODAY)
    assert [(i.field, i.severity) for i in issues] == [("lines[0].quantity", "warning")]
    assert is_valid(issues)


def test_estimated_quantity_is_warning():
    o = order(lines=[line(quantity=2, quantity_text="a couple of", quantity_is_estimate=True)])
    issues = validate_order(o, TODAY)
    assert [(i.field, i.severity) for i in issues] == [("lines[0].quantity", "warning")]
    assert "a couple of" in issues[0].message


def test_past_delivery_date_is_warning():
    o = order(requested_delivery_date=TODAY - timedelta(days=1))
    assert issues_for(o) == [("requested_delivery_date", "warning")]


def test_delivery_today_is_fine():
    assert issues_for(order(requested_delivery_date=TODAY)) == []


def test_far_future_delivery_date_is_warning():
    assert issues_for(order(requested_delivery_date=TODAY + timedelta(days=60))) == []
    o = order(requested_delivery_date=TODAY + timedelta(days=61))
    assert issues_for(o) == [("requested_delivery_date", "warning")]


def test_missing_customer_name_is_warning():
    assert issues_for(order(customer_name=None)) == [("customer_name", "warning")]
    assert issues_for(order(customer_name="  ")) == [("customer_name", "warning")]


def test_duplicate_lines_are_warning():
    o = order(lines=[line("Red Peppers"), line("basil"), line("  red   peppers ")])
    assert issues_for(o) == [("lines[2].product_description", "warning")]


def test_multiple_issues_reported_together():
    o = order(customer_name=None, lines=[line(quantity=0), line("basil", quantity=900)])
    assert issues_for(o) == [
        ("lines[0].quantity", "error"),
        ("lines[1].quantity", "warning"),
        ("customer_name", "warning"),
    ]
