"""Composition root: build the use case from settings. Shared by the API and the CLI."""

from dotenv import load_dotenv

from order_extractor.adapters.llm import create_llm_client
from order_extractor.application.extract_order import ExtractOrder
from order_extractor.config import Settings


def load_settings() -> Settings:
    """Read settings from the environment, after loading ``.env`` if there is one."""
    load_dotenv()
    return Settings.from_env()


def build_extract_order(settings: Settings) -> ExtractOrder:
    return ExtractOrder(create_llm_client(settings))
