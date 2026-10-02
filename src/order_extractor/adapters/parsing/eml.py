"""Parser for .eml files: headers that matter for an order plus the readable body."""

import re
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from html.parser import HTMLParser

from order_extractor.adapters.parsing.text import clean_text
from order_extractor.application.errors import DocumentParseError

# Subject often carries the PO number; From and Date help the LLM with customer and dates.
HEADERS = ("From", "Date", "Subject")
FORWARDED_SEPARATOR = "---------- Forwarded message ----------"


class EmlParser:
    def parse(self, data: bytes) -> str:
        message = BytesParser(policy=policy.default).parsebytes(data)
        if not isinstance(message, EmailMessage) or not (message.keys() or message.get_payload()):
            raise DocumentParseError("Not a valid email message.")
        return clean_text(_render(message))


def _render(message: EmailMessage) -> str:
    headers = [f"{name}: {message[name]}" for name in HEADERS if message[name]]
    parts = ["\n".join(headers), _body(message)]
    # Emails forwarded "as attachment" carry the original as a message/rfc822 part.
    for part in message.walk():
        if part.get_content_type() == "message/rfc822":
            for inner in part.iter_parts():
                parts.append(f"{FORWARDED_SEPARATOR}\n{_render(inner)}")
    return "\n\n".join(p for p in parts if p)


def _body(message: EmailMessage) -> str:
    part = message.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    content = part.get_content()
    if part.get_content_subtype() == "html":
        return html_to_text(content)
    return content


def html_to_text(html: str) -> str:
    extractor = _HTMLText()
    extractor.feed(html)
    extractor.close()
    lines = (line.strip() for line in "".join(extractor.chunks).splitlines())
    return "\n".join(line for line in lines if line)


class _HTMLText(HTMLParser):
    BLOCK_TAGS = {"p", "div", "br", "tr", "table", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"}
    SKIP_TAGS = {"script", "style", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "li":
            self.chunks.append("\n- ")
        elif tag in {"td", "th"}:
            self.chunks.append(" ")
        elif tag in self.BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self.BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            # Whitespace in HTML source is not meaningful; line breaks come from tags.
            self.chunks.append(re.sub(r"\s+", " ", data))
