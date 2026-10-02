"""Document parser adapters and the registry that picks one by file extension."""

from pathlib import Path, PurePath

from order_extractor.adapters.parsing.eml import EmlParser
from order_extractor.adapters.parsing.pdf import PdfParser
from order_extractor.adapters.parsing.text import PlainTextParser
from order_extractor.application.errors import UnsupportedFileTypeError
from order_extractor.application.ports import DocumentParser

PARSERS: dict[str, DocumentParser] = {
    ".txt": PlainTextParser(),
    ".eml": EmlParser(),
    ".pdf": PdfParser(),
}


def supported_extensions() -> list[str]:
    return sorted(PARSERS)


def get_parser(filename: str) -> DocumentParser:
    extension = PurePath(filename).suffix.lower()
    try:
        return PARSERS[extension]
    except KeyError:
        raise UnsupportedFileTypeError(
            f"Unsupported file type {extension or '(none)'!r}; "
            f"expected one of {', '.join(supported_extensions())}."
        ) from None


def parse_bytes(data: bytes, filename: str) -> str:
    return get_parser(filename).parse(data)


def parse_file(path: str | Path) -> str:
    path = Path(path)
    return parse_bytes(path.read_bytes(), path.name)


__all__ = [
    "EmlParser",
    "PdfParser",
    "PlainTextParser",
    "get_parser",
    "parse_bytes",
    "parse_file",
    "supported_extensions",
]
