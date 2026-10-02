"""ExtractOrder use case: document text -> LLM -> JSON -> Order -> validation issues.

Problems with the *content* (empty document, bad JSON, output that doesn't fit the schema,
business-rule failures) come back as issues in the result. Infrastructure failures (the LLM
call itself, an unreadable file) raise ``LLMError`` / ``DocumentParseError`` for the caller
to map, e.g. to an HTTP status.
"""

import json
import time
from collections.abc import Callable
from datetime import date
from typing import Any

from pydantic import ValidationError

from order_extractor.application.ports import DocumentParser, LLMClient
from order_extractor.application.prompts import SYSTEM_PROMPT, build_user_prompt
from order_extractor.domain.models import ExtractionResult, Order, ValidationIssue
from order_extractor.domain.validation import is_valid, validate_order

DEFAULT_MAX_INPUT_CHARS = 50_000
OUTPUT_FIELD = "llm_output"


class ExtractOrder:
    def __init__(
        self,
        llm: LLMClient,
        *,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._llm = llm
        self._max_input_chars = max_input_chars
        self._clock = clock

    async def from_document(
        self, data: bytes, parser: DocumentParser, today: date
    ) -> ExtractionResult:
        return await self.from_text(parser.parse(data), today)

    async def from_text(self, text: str, today: date) -> ExtractionResult:
        started = self._clock()
        text = text.strip()
        if not text:
            return ExtractionResult(
                order=None,
                issues=[_error("text", "The document is empty.")],
                is_valid=False,
                model=self._llm.model,
                latency_ms=0,
            )

        issues: list[ValidationIssue] = []
        if len(text) > self._max_input_chars:
            text = text[: self._max_input_chars]
            issues.append(
                _warning("text", f"Document truncated to {self._max_input_chars:,} characters.")
            )

        response = await self._llm.extract(
            SYSTEM_PROMPT, build_user_prompt(text, today), Order.model_json_schema()
        )
        order, parse_issues = parse_order(response.text)
        issues += parse_issues
        if order is not None:
            issues += validate_order(order, today)

        return ExtractionResult(
            order=order,
            issues=issues,
            is_valid=order is not None and is_valid(issues),
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=round((self._clock() - started) * 1000),
        )


def parse_order(text: str) -> tuple[Order | None, list[ValidationIssue]]:
    """Turn raw model output into an ``Order``, or issues explaining why it couldn't."""
    data = _load_json_object(text)
    if data is None:
        return None, [_error(OUTPUT_FIELD, "The model did not return valid JSON.")]
    if not isinstance(data, dict):
        return None, [_error(OUTPUT_FIELD, "The model returned JSON that is not an object.")]
    try:
        return Order.model_validate(data), []
    except ValidationError as exc:
        return None, [
            _error(_field_path(err["loc"]), f"Doesn't match the schema: {err['msg']}.")
            for err in exc.errors()
        ]


def _load_json_object(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Models without structured outputs sometimes wrap the JSON in prose.
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _field_path(loc: tuple[int | str, ...]) -> str:
    """("lines", 0, "quantity") -> "lines[0].quantity", matching validation.py's field names."""
    path = ""
    for part in loc:
        if isinstance(part, int):
            path += f"[{part}]"
        elif path:
            path += f".{part}"
        else:
            path = str(part)
    return path or OUTPUT_FIELD


def _error(field: str, message: str) -> ValidationIssue:
    return ValidationIssue(field=field, severity="error", message=message)


def _warning(field: str, message: str) -> ValidationIssue:
    return ValidationIssue(field=field, severity="warning", message=message)
