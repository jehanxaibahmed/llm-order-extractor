"""Ports: the interfaces the application needs from the outside world.

Adapters implement these; the use case only ever sees the protocol, so an LLM provider or
file format can be swapped (or faked in tests) without touching application code.
"""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMClient(Protocol):
    async def extract(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        """Send the prompts and return raw text that should match ``schema`` (a JSON schema)."""
        ...


class DocumentParser(Protocol):
    def parse(self, data: bytes) -> str:
        """Turn a raw document (email, .eml, PDF, ...) into plain text."""
        ...
