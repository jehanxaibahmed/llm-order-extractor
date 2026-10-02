"""Pydantic models for orders and extraction results.

The order models are deliberately permissive: they describe what the LLM returned, not what a
good order looks like. Business rules (at least one line, quantity > 0, ...) live in
``validation.py`` so a bad order becomes a ``ValidationIssue`` instead of an exception.
"""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel


class OrderLine(BaseModel):
    product_description: str
    """As written by the customer; matching to a catalogue happens later."""
    quantity: float | None = None
    quantity_text: str | None = None
    """Original wording of the quantity, e.g. "a couple of"."""
    quantity_is_estimate: bool = False
    """True when the quantity was vague ("a couple", "a few") and ``quantity`` is a best guess."""
    unit: str | None = None
    notes: str | None = None


class Order(BaseModel):
    customer_name: str | None = None
    customer_reference: str | None = None
    """PO number or similar reference, if given."""
    requested_delivery_date: date | None = None
    delivery_address: str | None = None
    lines: list[OrderLine] = []
    notes: str | None = None


class ValidationIssue(BaseModel):
    field: str
    severity: Literal["error", "warning"]
    message: str


class ExtractionResult(BaseModel):
    order: Order | None
    issues: list[ValidationIssue]
    is_valid: bool
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int


def llm_schema() -> dict[str, Any]:
    """JSON schema for ``Order`` that OpenAI structured outputs accept in strict mode.

    Strict mode requires every property to be listed in ``required`` (optional fields are
    nullable instead) and ``additionalProperties: false`` on every object, and it rejects
    ``default``. Pydantic's schema has none of that, so we rewrite it.
    """
    schema = Order.model_json_schema()
    _make_strict(schema)
    return schema


def _make_strict(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("default", None)
        if node.get("type") == "object" and "properties" in node:
            node["required"] = list(node["properties"])
            node["additionalProperties"] = False
        for value in node.values():
            _make_strict(value)
    elif isinstance(node, list):
        for item in node:
            _make_strict(item)
