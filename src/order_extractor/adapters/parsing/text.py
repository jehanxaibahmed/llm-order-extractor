"""Plain-text parser and the text clean-up shared by all parsers."""

import re

# Tried in order. latin-1 maps every byte, so decoding never fails outright.
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def decode(data: bytes) -> str:
    for encoding in ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AssertionError("unreachable: latin-1 decodes any bytes")


def clean_text(text: str) -> str:
    """Normalise line endings, strip trailing spaces and collapse runs of blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = [line.rstrip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


class PlainTextParser:
    def parse(self, data: bytes) -> str:
        return clean_text(decode(data))
