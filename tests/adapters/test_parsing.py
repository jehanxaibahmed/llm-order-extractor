from email.message import EmailMessage

import pytest

from order_extractor.adapters.parsing import (
    EmlParser,
    PdfParser,
    PlainTextParser,
    get_parser,
    parse_bytes,
    parse_file,
    supported_extensions,
)
from order_extractor.adapters.parsing.eml import FORWARDED_SEPARATOR, html_to_text
from order_extractor.application.errors import DocumentParseError, UnsupportedFileTypeError

BODY = "Hi,\n\nPlease send 2 boxes of red peppers and a tray of basil.\n\nThanks,\nMaya"


def make_email(body=BODY, html=None, subject="Order PO-2231"):
    msg = EmailMessage()
    msg["From"] = "Maya Patel <maya@greenleafcafe.example>"
    msg["To"] = "orders@freshfarm.example"
    msg["Date"] = "Thu, 01 Oct 2026 09:15:00 +0100"
    msg["Subject"] = subject
    if body is not None:
        msg.set_content(body)
    if html is not None:
        if body is None:
            msg.set_content(html, subtype="html")
        else:
            msg.add_alternative(html, subtype="html")
    return msg


# --- plain text -----------------------------------------------------------------------------


def test_plain_text_utf8():
    assert PlainTextParser().parse("Green Leaf Café\n2 trays basil".encode()) == (
        "Green Leaf Café\n2 trays basil"
    )


def test_plain_text_strips_bom_and_windows_line_endings():
    data = "﻿Line one  \r\nLine two\r\n".encode()
    assert PlainTextParser().parse(data) == "Line one\nLine two"


def test_plain_text_falls_back_to_cp1252():
    assert PlainTextParser().parse("Café – 3 cases".encode("cp1252")) == "Café – 3 cases"


def test_plain_text_collapses_blank_lines():
    assert PlainTextParser().parse(b"a\n\n\n\n\nb") == "a\n\nb"


# --- .eml -----------------------------------------------------------------------------------


def test_eml_keeps_order_headers_and_body():
    text = EmlParser().parse(make_email().as_bytes())
    assert "Subject: Order PO-2231" in text
    assert "From: Maya Patel <maya@greenleafcafe.example>" in text
    assert "Date: Thu, 01 Oct 2026 09:15:00 +0100" in text
    assert "2 boxes of red peppers" in text
    assert "To:" not in text


def test_eml_prefers_plain_over_html():
    msg = make_email(html="<p>HTML version <b>only</b></p>")
    text = EmlParser().parse(msg.as_bytes())
    assert "2 boxes of red peppers" in text
    assert "HTML version" not in text


def test_eml_html_only_body_is_converted():
    html = (
        "<html><head><style>p {color: red}</style></head><body>"
        "<p>Order for Fresh Farm Ltd:</p><ul><li>3 cases cherry tomatoes</li>"
        "<li>1 kg rocket</li></ul><p>Fish &amp; chips night &ndash; thanks!</p></body></html>"
    )
    text = EmlParser().parse(make_email(body=None, html=html).as_bytes())
    assert "color: red" not in text
    assert "- 3 cases cherry tomatoes\n- 1 kg rocket" in text
    assert "Fish & chips night – thanks!" in text


def test_eml_decodes_non_ascii_body():
    msg = make_email(body="Green Leaf Café wants 4 crème fraîche tubs.")
    assert "Green Leaf Café wants 4 crème fraîche tubs." in EmlParser().parse(msg.as_bytes())


def test_eml_ignores_attachments():
    msg = make_email()
    msg.add_attachment(b"%PDF-1.4 binary", maintype="application", subtype="pdf", filename="x.pdf")
    text = EmlParser().parse(msg.as_bytes())
    assert "2 boxes of red peppers" in text
    assert "PDF-1.4" not in text


def test_eml_includes_message_forwarded_as_attachment():
    original = make_email(body="Please send 6 trays of microgreens.", subject="Weekly order")
    outer = make_email(body="FYI, see below.", subject="Fwd: Weekly order")
    outer.add_attachment(original)
    text = EmlParser().parse(outer.as_bytes())
    assert text.index("FYI, see below.") < text.index(FORWARDED_SEPARATOR)
    assert "Subject: Weekly order" in text
    assert "6 trays of microgreens" in text


def test_eml_rejects_empty_input():
    with pytest.raises(DocumentParseError):
        EmlParser().parse(b"")


def test_html_to_text_handles_tables():
    html = "<table><tr><td>Basil</td><td>2</td></tr><tr><td>Mint</td><td>1</td></tr></table>"
    assert html_to_text(html) == "Basil 2\nMint 1"


def test_html_to_text_line_breaks_and_whitespace():
    html = "<div>  Hello\n   there</div><div>3 trays<br>basil</div>"
    assert html_to_text(html) == "Hello there\n3 trays\nbasil"


# --- PDF ------------------------------------------------------------------------------------


def test_pdf_extracts_text_from_all_pages(make_pdf):
    data = make_pdf(
        [
            ["PURCHASE ORDER PO-7781", "Fresh Farm Ltd"],
            ["10 x Red peppers (box)", "Delivery: 2026-10-08"],
        ]
    )
    text = PdfParser().parse(data)
    assert "PURCHASE ORDER PO-7781" in text
    assert "Fresh Farm Ltd" in text
    assert text.index("Fresh Farm Ltd") < text.index("10 x Red peppers (box)")
    assert "Delivery: 2026-10-08" in text


def test_pdf_non_ascii(make_pdf):
    assert "Green Leaf Café" in PdfParser().parse(make_pdf([["Green Leaf Café"]]))


def test_pdf_without_text_is_rejected(make_pdf):
    with pytest.raises(DocumentParseError, match="OCR"):
        PdfParser().parse(make_pdf([[]]))


def test_pdf_garbage_is_rejected():
    with pytest.raises(DocumentParseError):
        PdfParser().parse(b"this is not a pdf")


# --- registry -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "parser_type"),
    [
        ("order.txt", PlainTextParser),
        ("ORDER.EML", EmlParser),
        ("po.pdf", PdfParser),
        ("dir.with.dots/po.pdf", PdfParser),
    ],
)
def test_get_parser_by_extension(filename, parser_type):
    assert isinstance(get_parser(filename), parser_type)


@pytest.mark.parametrize("filename", ["order.docx", "order", ""])
def test_get_parser_rejects_unknown_types(filename):
    with pytest.raises(UnsupportedFileTypeError, match=".eml, .pdf, .txt"):
        get_parser(filename)


def test_supported_extensions():
    assert supported_extensions() == [".eml", ".pdf", ".txt"]


def test_parse_bytes_and_file(tmp_path, make_pdf):
    assert parse_bytes(b"3 trays basil", "a.txt") == "3 trays basil"
    path = tmp_path / "po.pdf"
    path.write_bytes(make_pdf([["2 cases leeks"]]))
    assert parse_file(path) == "2 cases leeks"
