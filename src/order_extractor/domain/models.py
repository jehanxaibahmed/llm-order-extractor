"""Pydantic models for orders and extraction results.

The order models are deliberately permissive: they describe what the LLM returned, not what a
good order looks like. Business rules (at least one line, quantity > 0, ...) live in
``validation.py`` so a bad order becomes a ``ValidationIssue`` instead of an exception.

Field docstrings become JSON-schema descriptions, which show up in the API docs and are sent
to the LLM as part of the schema.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Model(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)


class OrderLine(_Model):
    product_description: str
    """The product exactly as the customer wrote it; catalogue matching happens later."""
    quantity: float | None = None
    """Numeric quantity, or null if none was given."""
    quantity_text: str | None = None
    """Original wording of the quantity, e.g. "2" or "a couple of"."""
    quantity_is_estimate: bool = False
    """True when the wording was vague ("a couple", "a few") and quantity is a best guess."""
    unit: str | None = None
    """Packaging or measure as written: "box", "case", "tray", "kg", ..."""
    notes: str | None = None
    """Anything else the customer said about this line, e.g. "ripe please"."""


class Order(_Model):
    customer_name: str | None = None
    """The ordering business, or person if no business is named."""
    customer_reference: str | None = None
    """PO number or similar reference, if given."""
    requested_delivery_date: date | None = None
    """Delivery date as an ISO date, with relative dates ("Thursday") resolved."""
    delivery_address: str | None = None
    """Delivery address, if the document gives one for this order."""
    lines: list[OrderLine] = []
    """Order lines; empty when the document contains no order."""
    notes: str | None = None
    """Order-level instructions, e.g. delivery access or invoicing notes."""


class ValidationIssue(_Model):
    field: str
    """Path of the field the issue is about, e.g. "lines[0].quantity"."""
    severity: Literal["error", "warning"]
    """"error" makes the result invalid; "warning" needs a human look."""
    message: str
    """Human-readable explanation."""


class ExtractionResult(_Model):
    order: Order | None
    """The extracted order, or null if the model output could not be read."""
    issues: list[ValidationIssue]
    """Problems found while reading or validating the order."""
    is_valid: bool
    """True when there are no errors (warnings are allowed)."""
    model: str
    """The model that produced the result."""
    input_tokens: int | None = None
    """Prompt tokens used, if the provider reports them."""
    output_tokens: int | None = None
    """Completion tokens used, if the provider reports them."""
    latency_ms: int
    """Time taken for the whole extraction, in milliseconds."""
