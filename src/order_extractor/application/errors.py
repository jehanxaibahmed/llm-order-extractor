"""Errors the application layer and its adapters raise."""


class DocumentParseError(Exception):
    """A document could not be turned into text."""


class UnsupportedFileTypeError(DocumentParseError):
    """No parser is registered for the file's extension."""


class LLMError(Exception):
    """The LLM call failed (network, auth, rate limit after retries, refusal, empty output)."""
