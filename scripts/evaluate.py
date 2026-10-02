"""Run every sample through the extractor and score it against samples/expected/.

    python -m scripts.evaluate                      # real LLM, settings from env / .env
    python -m scripts.evaluate --model gpt-4.1-mini
    python -m scripts.evaluate --dry-run            # FakeLLM replays the ground truth, no key

Prints a Markdown table and writes it to eval_results.md.

Scoring is per check, and lenient where the text can be written several ways: names and
references ignore case and punctuation, units ignore plurals ("boxes" = "box"), and lines are
matched to the expected ones by description, so their order doesn't matter.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from order_extractor.adapters.llm import FakeLLM
from order_extractor.adapters.parsing import parse_file
from order_extractor.application.errors import DocumentParseError, LLMError
from order_extractor.application.extract_order import ExtractOrder
from order_extractor.config import PROVIDERS, ConfigError
from order_extractor.domain.models import ExtractionResult, Order, OrderLine
from order_extractor.entrypoints.wiring import build_extract_order, load_settings

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

# USD per 1M tokens (input, output). Estimates only: check the provider's pricing page.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}
DESCRIPTION_MATCH = 0.6


# --- scoring --------------------------------------------------------------------------------


@dataclass
class Score:
    """Checks for one sample. Each check is (name, passed)."""

    checks: list[tuple[str, bool]] = field(default_factory=list)

    def add(self, name: str, passed: bool) -> None:
        self.checks.append((name, passed))

    @property
    def passed(self) -> int:
        return sum(ok for _, ok in self.checks)

    @property
    def accuracy(self) -> float:
        return self.passed / len(self.checks) if self.checks else 0.0

    def failed(self) -> list[str]:
        return [name for name, ok in self.checks if not ok]


def score(expected: dict[str, Any], result: ExtractionResult) -> Score:
    s = Score()
    want = Order.model_validate(expected["order"])
    got = result.order

    s.add("parsed", got is not None)
    if got is None:
        got = Order()

    s.add("customer_name", _same_name(want.customer_name, got.customer_name))
    s.add("customer_reference", _same_ref(want.customer_reference, got.customer_reference))
    s.add("delivery_date", want.requested_delivery_date == got.requested_delivery_date)
    s.add("line_count", len(want.lines) == len(got.lines))

    matches = _match_indices(want.lines, got.lines)
    for i, want_line in enumerate(want.lines):
        got_line = got.lines[matches[i]] if i in matches else None
        s.add(f"lines[{i}].description", got_line is not None)
        if got_line is None:
            s.add(f"lines[{i}].quantity", False)
            s.add(f"lines[{i}].unit", False)
            continue
        if want_line.quantity_is_estimate:
            # Any reasonable reading is fine, as long as it is flagged as an estimate.
            s.add(f"lines[{i}].quantity", got_line.quantity_is_estimate)
        else:
            s.add(f"lines[{i}].quantity", got_line.quantity == want_line.quantity)
        s.add(f"lines[{i}].unit", _unit(want_line.unit) == _unit(got_line.unit))

    # Renumber line issues to the expected line they matched, so line order doesn't matter.
    renumber = {g: w for w, g in matches.items()}
    want_issues = {(i["field"], i["severity"]) for i in expected["issues"]}
    got_issues = {(_renumber(i.field, renumber), i.severity) for i in result.issues}
    s.add("issues", want_issues == got_issues)
    s.add("is_valid", result.is_valid == all(sev != "error" for _, sev in want_issues))
    return s


def _match_indices(want: list[OrderLine], got: list[OrderLine]) -> dict[int, int]:
    """Map each expected line index to the most similar unused extracted line index."""
    pairs = sorted(
        (
            (_similarity(w.product_description, g.product_description), wi, gi)
            for wi, w in enumerate(want)
            for gi, g in enumerate(got)
        ),
        reverse=True,
    )
    matched: dict[int, int] = {}
    used: set[int] = set()
    for ratio, wi, gi in pairs:
        if ratio >= DESCRIPTION_MATCH and wi not in matched and gi not in used:
            matched[wi] = gi
            used.add(gi)
    return matched


def _renumber(field_path: str, renumber: dict[int, int]) -> str:
    def repl(match: re.Match[str]) -> str:
        index = renumber.get(int(match.group(1)))
        return f"lines[{'?' if index is None else index}]"

    return re.sub(r"^lines\[(\d+)\]", repl, field_path)


def _normal(text: str | None) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").casefold()).split())


def _similarity(a: str, b: str) -> float:
    a, b = _normal(a), _normal(b)
    if a and b and (a in b or b in a):
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _same_name(want: str | None, got: str | None) -> bool:
    want, got = _normal(want), _normal(got)
    if not want or not got:
        return want == got
    return want in got or got in want


def _same_ref(want: str | None, got: str | None) -> bool:
    def key(ref: str | None) -> str:
        return re.sub(r"^po", "", re.sub(r"[^a-z0-9]", "", (ref or "").casefold()))

    return key(want) == key(got)


def _unit(unit: str | None) -> str:
    unit = _normal(unit)
    for plural, singular in (("boxes", "box"), ("bunches", "bunch"), ("punnets", "punnet")):
        unit = unit.replace(plural, singular)
    return unit.removesuffix("s") if len(unit) > 3 else unit


# --- running --------------------------------------------------------------------------------


@dataclass
class Row:
    name: str
    score: Score | None = None
    result: ExtractionResult | None = None
    error: str | None = None


def load_samples(samples_dir: Path = SAMPLES) -> list[dict[str, Any]]:
    samples = []
    for path in sorted((samples_dir / "expected").glob("*.json")):
        expected = json.loads(path.read_text())
        expected["name"] = path.stem
        expected["path"] = samples_dir / expected["source"]
        samples.append(expected)
    return samples


async def run_sample(sample: dict[str, Any], extract: ExtractOrder) -> Row:
    row = Row(sample["name"])
    try:
        text = parse_file(sample["path"])
        row.result = await extract.from_text(text, date.fromisoformat(sample["today"]))
    except (DocumentParseError, LLMError) as exc:
        row.error = str(exc)
        return row
    row.score = score(sample, row.result)
    return row


async def run_all(samples: list[dict[str, Any]], make_extract, concurrency: int = 4) -> list[Row]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(sample: dict[str, Any]) -> Row:
        async with semaphore:
            return await run_sample(sample, make_extract(sample))

    return list(await asyncio.gather(*(one(s) for s in samples)))


# --- reporting ------------------------------------------------------------------------------


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    # Longest prefix wins, so "gpt-4o-mini-2024-07-18" is priced as gpt-4o-mini, not gpt-4o.
    model = model.rsplit("/", 1)[-1]
    for name in sorted(PRICES, key=len, reverse=True):
        if model.startswith(name):
            price_in, price_out = PRICES[name]
            return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
    return None


def render(rows: list[Row], *, model: str, provider: str, when: datetime) -> str:
    out = [
        "# Evaluation results",
        "",
        f"- **Model:** `{model}` via {provider}",
        f"- **Run:** {when:%Y-%m-%d %H:%M}",
        f"- **Samples:** {len(rows)} (synthetic, see `samples/README.md`)",
        "",
        "| Sample | Checks | Accuracy | Failed checks | Issues | Tokens in/out | Latency | Cost |",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]
    totals = {"passed": 0, "checks": 0, "in": 0, "out": 0, "ms": 0, "cost": 0.0}
    cost_known = True
    for row in rows:
        if row.error or row.score is None or row.result is None:
            out.append(f"| {row.name} | – | – | error: {row.error} | – | – | – | – |")
            continue
        s, r = row.score, row.result
        tokens_in, tokens_out = r.input_tokens or 0, r.output_tokens or 0
        cost = cost_usd(r.model, tokens_in, tokens_out)
        cost_known &= cost is not None
        issues = ", ".join(f"{i.severity[0].upper()}:{i.field}" for i in r.issues) or "–"
        out.append(
            f"| {row.name} | {s.passed}/{len(s.checks)} | {s.accuracy:.0%} | "
            f"{', '.join(s.failed()) or '–'} | {issues} | {tokens_in}/{tokens_out} | "
            f"{r.latency_ms} ms | {_money(cost)} |"
        )
        totals["passed"] += s.passed
        totals["checks"] += len(s.checks)
        totals["in"] += tokens_in
        totals["out"] += tokens_out
        totals["ms"] += r.latency_ms
        totals["cost"] += cost or 0.0

    scored = [r for r in rows if r.score]
    accuracy = totals["passed"] / totals["checks"] if totals["checks"] else 0.0
    mean_ms = totals["ms"] // len(scored) if scored else 0
    out.append(
        f"| **Total** | **{totals['passed']}/{totals['checks']}** | **{accuracy:.0%}** | | | "
        f"{totals['in']}/{totals['out']} | {mean_ms} ms avg | "
        f"{_money(totals['cost'] if cost_known else None)} |"
    )

    out += ["", "## Accuracy by check", "", "| Check | Passed |", "|---|---:|"]
    for name, (passed, total) in by_check(scored).items():
        out.append(f"| {name} | {passed}/{total} ({passed / total:.0%}) |")

    out += [
        "",
        "Issues column: `E:` error, `W:` warning. Costs are estimates from list prices.",
        "",
    ]
    return "\n".join(out)


def by_check(rows: list[Row]) -> dict[str, tuple[int, int]]:
    """Group per-line checks (lines[0].quantity, lines[1].quantity, ...) into line.quantity."""
    groups: dict[str, list[bool]] = {}
    for row in rows:
        assert row.score is not None
        for name, ok in row.score.checks:
            groups.setdefault(re.sub(r"^lines\[\d+\]", "line", name), []).append(ok)
    return {name: (sum(oks), len(oks)) for name, oks in groups.items()}


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


# --- CLI ------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", choices=PROVIDERS, help="override LLM_PROVIDER")
    parser.add_argument("--model", help="override LLM_MODEL")
    parser.add_argument(
        "--dry-run", action="store_true", help="replay the ground truth with FakeLLM (no key)"
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "eval_results.md")
    args = parser.parse_args(argv)

    samples = load_samples()
    if args.dry_run:
        model, provider = "fake-llm", "FakeLLM (dry run)"

        def make_extract(sample):
            return ExtractOrder(FakeLLM(sample["order"]))
    else:
        try:
            settings = load_settings()
            overrides = {"llm_provider": args.provider, "llm_model": args.model}
            settings = dataclasses.replace(settings, **{k: v for k, v in overrides.items() if v})
            extract = build_extract_order(settings)
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        model, provider = settings.llm_model, settings.llm_provider

        def make_extract(sample):
            return extract

    rows = asyncio.run(run_all(samples, make_extract, args.concurrency))
    report = render(rows, model=model, provider=provider, when=datetime.now())
    print(report)
    args.output.write_text(report)
    print(f"wrote {args.output}", file=sys.stderr)
    return 1 if any(r.error for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
