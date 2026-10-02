"""Command line: extract an order from a file and print the result.

    order-extractor samples/emails/01_simple.txt --today 2026-10-01

The JSON result goes to stdout (so it can be piped), the human summary to stderr.
Exit codes: 0 valid order, 1 order has errors, 2 could not run (bad input, config or LLM).
"""

import argparse
import asyncio
import dataclasses
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import TextIO

from order_extractor import __version__
from order_extractor.adapters.parsing import get_parser, supported_extensions
from order_extractor.application.errors import DocumentParseError, LLMError
from order_extractor.application.extract_order import ExtractOrder
from order_extractor.application.ports import LLMClient
from order_extractor.config import PROVIDERS, ConfigError
from order_extractor.domain.models import ExtractionResult
from order_extractor.entrypoints.wiring import build_extract_order, load_settings

EXIT_VALID, EXIT_INVALID, EXIT_FAILED = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="order-extractor",
        description="Extract a structured order from an email or PDF using an LLM.",
    )
    parser.add_argument(
        "path", type=Path, help=f"file to read ({', '.join(supported_extensions())})"
    )
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        default=None,
        help="reference date for relative dates, YYYY-MM-DD (default: today)",
    )
    parser.add_argument("--provider", choices=PROVIDERS, help="override LLM_PROVIDER")
    parser.add_argument("--model", help="override LLM_MODEL")
    parser.add_argument("-q", "--quiet", action="store_true", help="print only the JSON result")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None, *, llm: LLMClient | None = None) -> int:
    """Run the CLI. ``llm`` lets tests inject a fake client instead of reading settings."""
    args = build_parser().parse_args(argv)
    try:
        parser = get_parser(args.path.name)
        data = args.path.read_bytes()
        extract = ExtractOrder(llm) if llm else _extract_order_from_settings(args)
        result = asyncio.run(extract.from_document(data, parser, args.today or date.today()))
    except (OSError, DocumentParseError, ConfigError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED

    print(result.model_dump_json(indent=2))
    if not args.quiet:
        print_summary(result, sys.stderr)
    return EXIT_VALID if result.is_valid else EXIT_INVALID


def _extract_order_from_settings(args: argparse.Namespace) -> ExtractOrder:
    settings = load_settings()
    overrides = {"llm_provider": args.provider, "llm_model": args.model}
    settings = dataclasses.replace(settings, **{k: v for k, v in overrides.items() if v})
    return build_extract_order(settings)


def print_summary(result: ExtractionResult, out: TextIO) -> None:
    order = result.order
    status = "Valid order" if result.is_valid else "Invalid order"
    if order is None:
        print(f"\n{status}: no order could be read.", file=out)
    else:
        details = [
            order.customer_name or "unknown customer",
            f"ref {order.customer_reference}" if order.customer_reference else None,
            f"delivery {order.requested_delivery_date}" if order.requested_delivery_date else None,
            f"{len(order.lines)} line{'s' if len(order.lines) != 1 else ''}",
        ]
        print(f"\n{status}: {', '.join(d for d in details if d)}", file=out)
        for line in order.lines:
            quantity = "?" if line.quantity is None else f"{line.quantity:g}"
            estimate = " (estimate)" if line.quantity_is_estimate else ""
            unit = f" {line.unit}" if line.unit else ""
            print(f"  - {quantity}{unit}{estimate} {line.product_description}", file=out)

    for issue in result.issues:
        print(f"  {issue.severity:<7} {issue.field}: {issue.message}", file=out)

    tokens = (
        f", {result.input_tokens} in / {result.output_tokens} out tokens"
        if result.input_tokens is not None
        else ""
    )
    print(f"  model {result.model}{tokens}, {result.latency_ms} ms", file=out)


if __name__ == "__main__":
    sys.exit(main())
