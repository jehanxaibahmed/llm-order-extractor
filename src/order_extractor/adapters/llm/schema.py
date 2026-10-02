"""Rewrite a JSON schema so OpenAI structured outputs accept it in strict mode."""

import copy
from typing import Any


def to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a strict-mode-compatible copy of ``schema``.

    Strict mode requires every property to be listed in ``required`` (optional fields are
    nullable instead) and ``additionalProperties: false`` on every object, and it rejects
    ``default``. Pydantic's ``model_json_schema()`` has none of that.
    """
    strict = copy.deepcopy(schema)
    _make_strict(strict)
    return strict


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
