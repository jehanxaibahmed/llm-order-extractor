# Sample orders

Synthetic orders for tests and evaluation. **Everything here is invented**: businesses,
people, addresses, phone numbers and PO numbers. Email addresses use the reserved
`.example` domain. The supplier receiving the orders is the made-up "Hillside Produce".

| # | File | What it tests |
|---|---|---|
| 01 | `emails/01_simple.txt` | single line, "tomorrow" |
| 02 | `emails/02_bulleted_list.txt` | bulleted lines, mixed units, a line with no unit |
| 03 | `emails/03_paragraph.txt` | lines written as a sentence, "a tray", no delivery date |
| 04 | `emails/04_relative_date.eml` | `.eml`, bare weekday ("Tuesday please") |
| 05 | `emails/05_po_in_subject.eml` | PO number in the subject, delivery address, quantity over 500 → warning |
| 06 | `emails/06_forwarded_signature.eml` | inline forward, long signature and disclaimer |
| 07 | `emails/07_chit_chat.txt` | order buried in small talk, "the 20th" |
| 08 | `emails/08_ambiguous_quantity.txt` | "a couple of", "a few" → estimates + warnings |
| 09 | `emails/09_no_order.txt` | no order at all → `lines: []` + error |
| 10 | `pdfs/10_purchase_order.pdf` | PDF purchase order, UK date format |
| 11 | `pdfs/11_two_page_po.pdf` | lines split across two pages, delivery date in the past → warning |

## Expected output

`expected/<name>.json` holds the ground truth for each sample:

```json
{
  "source": "emails/01_simple.txt",
  "today": "2026-10-01",
  "order": { "...": "an Order, as the extractor should return it" },
  "issues": [{ "field": "lines[0].quantity", "severity": "warning" }]
}
```

`today` is the reference date passed to the extractor (a Thursday), so relative dates have
one right answer. `tests/test_samples.py` checks that every expected order passes the real
validation rules with exactly the listed issues. It also checks that every product, quantity
and reference in the ground truth actually appears in the document.

The PDFs are generated: `python -m scripts.make_sample_pdfs`.
