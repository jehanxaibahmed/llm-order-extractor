"""Business rules that turn an extracted order into validation issues.

Rules return issues instead of raising, so the caller always gets a result it can show.
"""

from datetime import date

from order_extractor.schemas import Order, ValidationIssue

QUANTITY_WARNING_THRESHOLD = 500
MAX_DAYS_AHEAD = 60


def validate_order(order: Order, today: date) -> list[ValidationIssue]:
    """Check ``order`` against the business rules. ``today`` is the date given to the prompt."""
    issues: list[ValidationIssue] = []

    if not order.lines:
        issues.append(_error("lines", "No order lines found."))

    seen: dict[str, int] = {}
    for i, line in enumerate(order.lines):
        field = f"lines[{i}]"
        if line.quantity is None:
            issues.append(_error(f"{field}.quantity", "Quantity is missing."))
        elif line.quantity <= 0:
            issues.append(
                _error(f"{field}.quantity", f"Quantity must be above 0, got {line.quantity:g}.")
            )
        elif line.quantity > QUANTITY_WARNING_THRESHOLD:
            issues.append(
                _warning(
                    f"{field}.quantity",
                    f"Quantity {line.quantity:g} is above {QUANTITY_WARNING_THRESHOLD}.",
                )
            )

        if line.quantity_is_estimate:
            wording = f' ("{line.quantity_text}")' if line.quantity_text else ""
            issues.append(
                _warning(
                    f"{field}.quantity", f"Quantity is an estimate from vague wording{wording}."
                )
            )

        key = _normalise(line.product_description)
        if key in seen:
            issues.append(
                _warning(
                    f"{field}.product_description",
                    f"Duplicate of line {seen[key]}: {line.product_description!r}.",
                )
            )
        else:
            seen[key] = i

    if order.requested_delivery_date is not None:
        days_ahead = (order.requested_delivery_date - today).days
        if days_ahead < 0:
            issues.append(_warning("requested_delivery_date", "Delivery date is in the past."))
        elif days_ahead > MAX_DAYS_AHEAD:
            issues.append(
                _warning(
                    "requested_delivery_date",
                    f"Delivery date is {days_ahead} days away (more than {MAX_DAYS_AHEAD}).",
                )
            )

    if not (order.customer_name or "").strip():
        issues.append(_warning("customer_name", "Customer name is missing."))

    return issues


def is_valid(issues: list[ValidationIssue]) -> bool:
    return not any(issue.severity == "error" for issue in issues)


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def _error(field: str, message: str) -> ValidationIssue:
    return ValidationIssue(field=field, severity="error", message=message)


def _warning(field: str, message: str) -> ValidationIssue:
    return ValidationIssue(field=field, severity="warning", message=message)
