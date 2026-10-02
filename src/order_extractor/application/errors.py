"""Errors the application layer and its adapters raise."""


class DocumentParseError(Exception):
    """A document could not be turned into text."""


class UnsupportedFileTypeError(DocumentParseError):
    """No parser is registered for the file's extension."""
