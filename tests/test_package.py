import importlib

import order_extractor

MODULES = ["schemas", "parsing", "prompts", "llm", "extractor", "validation", "api", "cli"]


def test_version():
    assert order_extractor.__version__ == "0.0.1"


def test_modules_import():
    for name in MODULES:
        importlib.import_module(f"order_extractor.{name}")
