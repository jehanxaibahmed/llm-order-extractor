"""Generate the synthetic sample purchase-order PDFs in samples/pdfs/.

    python scripts/make_sample_pdfs.py

All businesses, people, addresses and PO numbers are invented.
"""

from pathlib import Path

from scripts.pdf_writer import build_pdf

OUT = Path(__file__).resolve().parent.parent / "samples" / "pdfs"

H1 = ("PURCHASE ORDER", 18, True)

PDFS = {
    "10_purchase_order.pdf": [
        [
            H1,
            "",
            ("Riverside Bistro", 12, True),
            "3 Wharf Road, Ashbury AB1 4QT",
            "accounts@riversidebistro.example",
            "",
            "To: Hillside Produce, Unit 7 Valley Trading Estate",
            "",
            "PO number: RB-2026-118",
            "Order date: 01/10/2026",
            "Delivery date: 05/10/2026",
            "",
            ("Qty   Unit    Description", 11, True),
            "4     box     Chestnut mushrooms",
            "2     case    Baby spinach",
            "6     kg      Maris Piper potatoes",
            "1     tray    Flat-leaf parsley",
            "",
            "Deliver to: kitchen door at rear, 3 Wharf Road, Ashbury AB1 4QT",
            "Authorised by: Priya Shah",
        ]
    ],
    "11_two_page_po.pdf": [
        [
            H1,
            "",
            ("Oak & Ember", 12, True),
            "88 High Street, Porthaven PH2 7LD",
            "",
            "Supplier: Hillside Produce",
            "PO number: OE-5530",
            "Requested delivery: 28/09/2026",
            "",
            ("Qty   Unit    Description", 11, True),
            "5     box     Jersey Royal potatoes",
            "3     case    Little Gem lettuce",
            "2     kg      Fresh ginger",
            "",
            "Continued on page 2",
        ],
        [
            ("PURCHASE ORDER OE-5530 (page 2 of 2)", 12, True),
            "",
            ("Qty   Unit    Description", 11, True),
            "4     bunch   Thai basil",
            "1     box     Limes",
            "",
            "Notes: please call Leo on arrival, side gate is locked before 10am.",
        ],
    ],
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pages in PDFS.items():
        (OUT / name).write_bytes(build_pdf(pages))
        print(f"wrote samples/pdfs/{name}")


if __name__ == "__main__":
    main()
