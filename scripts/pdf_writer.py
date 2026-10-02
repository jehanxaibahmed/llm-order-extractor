"""Minimal text-PDF writer (no dependencies), used by the tests and make_sample_pdfs.py.

Each page is a list of lines. A line is a string, or ``(text, size, bold)`` for headings.
Text is set in Helvetica with WinAnsi encoding, so Latin-1 characters like "é" work.
"""

Line = str | tuple[str, int, bool]

TOP, LEFT, PAGE_WIDTH, PAGE_HEIGHT = 760, 60, 612, 792


def build_pdf(pages: list[list[Line]]) -> bytes:
    page_ids = [5 + 2 * i for i in range(len(pages))]
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    font = "<< /Type /Font /Subtype /Type1 /BaseFont /{} /Encoding /WinAnsiEncoding >>"
    objects = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>",
        3: font.format("Helvetica"),
        4: font.format("Helvetica-Bold"),
    }
    for pid, lines in zip(page_ids, pages, strict=True):
        stream = _page_stream(lines)
        objects[pid] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {pid + 1} 0 R >>"
        )
        objects[pid + 1] = (
            f"<< /Length {len(stream.encode('cp1252'))} >>\nstream\n{stream}\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n{objects[num]}\nendobj\n".encode("cp1252")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for num in sorted(objects):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _page_stream(lines: list[Line]) -> str:
    ops, y = [], TOP
    for line in lines:
        text, size, bold = (line, 11, False) if isinstance(line, str) else line
        if text:
            ops.append(
                f"BT /{'F2' if bold else 'F1'} {size} Tf {LEFT} {y} Td ({_escape(text)}) Tj ET"
            )
        y -= round(size * 1.5)
    return "\n".join(ops)


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
