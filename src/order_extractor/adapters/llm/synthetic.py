"""SyntheticLLM: returns canned responses so tests and showcases run without an API key."""

import json
from dataclasses import dataclass
from typing import Any

from order_extractor.application.ports import LLMResponse

Canned = str | dict[str, Any] | Exception


@dataclass
class FakeCall:
    system: str
    user: str
    schema: dict[str, Any]


class SyntheticLLM:
    """Implements the ``LLMClient`` port.

    Each call returns the next canned response; the last one repeats once the list runs out.
    A dict is serialised to JSON, a string is returned as-is (useful for testing bad JSON),
    and an exception is raised.
    """

    def __init__(self, *responses: Canned, model: str = "fake-llm") -> None:
        if not responses:
            raise ValueError("SyntheticLLM needs at least one canned response.")
        self._responses = list(responses)
        self.model = model
        self.calls: list[FakeCall] = []

    async def extract(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        self.calls.append(FakeCall(system, user, schema))
        index = min(len(self.calls), len(self._responses)) - 1
        response = self._responses[index]
        if isinstance(response, Exception):
            raise response
        text = response if isinstance(response, str) else json.dumps(response)
        return LLMResponse(
            text=text, model=self.model, input_tokens=len(user) // 4, output_tokens=len(text) // 4
        )
