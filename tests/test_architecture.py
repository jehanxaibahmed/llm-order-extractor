"""Enforce the ports-and-adapters dependency rule by reading each module's imports."""

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

import order_extractor

SRC = Path(order_extractor.__file__).parent
LAYERS = ["domain", "application", "adapters", "entrypoints"]

# What each layer may import from inside the package.
ALLOWED = {
    "domain": {"domain"},
    "application": {"domain", "application"},
    "adapters": {"domain", "application", "adapters", "config"},
    "entrypoints": {"domain", "application", "adapters", "entrypoints", "config"},
    "config": {"config"},
}
IO_LIBRARIES = {"openai", "fastapi", "uvicorn", "pypdf", "httpx", "dotenv"}


def _modules():
    for path in sorted(SRC.rglob("*.py")):
        parts = path.relative_to(SRC).with_suffix("").parts
        if parts[0] in ALLOWED:
            yield parts[0], path


def _imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


@pytest.mark.parametrize(("layer", "path"), list(_modules()), ids=lambda v: str(v))
def test_layer_dependencies(layer, path):
    for name in _imports(path):
        parts = name.split(".")
        if parts[0] == "order_extractor" and len(parts) > 1:
            assert parts[1] in ALLOWED[layer], f"{layer} must not import {name} ({path.name})"
        if layer in {"domain", "application"}:
            assert parts[0] not in IO_LIBRARIES, f"{layer} must not import {name} ({path.name})"


def test_every_module_imports():
    for info in pkgutil.walk_packages([str(SRC)], prefix="order_extractor."):
        importlib.import_module(info.name)


def test_every_source_file_is_in_a_layer():
    top_level = {p.stem for p in SRC.glob("*.py")} | {p.name for p in SRC.iterdir() if p.is_dir()}
    assert top_level - {"__init__", "__pycache__"} == set(LAYERS) | {"config"}
