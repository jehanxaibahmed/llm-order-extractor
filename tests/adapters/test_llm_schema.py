from order_extractor.adapters.llm.schema import to_strict_schema
from order_extractor.domain.models import Order


def _objects(node):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from _objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _objects(item)


def _keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _keys(item)


def test_order_schema_is_strict_compatible():
    schema = to_strict_schema(Order.model_json_schema())
    objects = list(_objects(schema))
    assert len(objects) == 2  # Order and OrderLine
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    assert "default" not in set(_keys(schema))


def test_input_schema_is_not_mutated():
    original = Order.model_json_schema()
    to_strict_schema(original)
    assert original == Order.model_json_schema()
