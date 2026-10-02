"""Parser for text-based PDFs. Scanned PDFs need OCR, which is out of scope for v0.1."""

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from order_extractor.adapters.parsing.text import clean_text
from order_extractor.application.errors import DocumentParseError


class PdfParser:
    def parse(self, data: bytes) -> str:
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages]
        except (PdfReadError, ValueError, OSError) as exc:
            raise DocumentParseError(f"Could not read PDF: {exc}") from exc

        text = clean_text("\n\n".join(pages))
        if not text:
            raise DocumentParseError(
                "PDF has no extractable text. It may be scanned; OCR is not supported yet."
            )
        return text
