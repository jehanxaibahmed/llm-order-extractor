import copy
from datetime import datetime

import pytest

from order_extractor.adapters.llm import FakeLLM
from order_extractor.application.errors import LLMError
from order_extractor.application.extract_order import ExtractOrder
from scripts import evaluate
from scripts.evaluate import Row, by_check, cost_usd, load_samples, render, run_all


@pytest.fixture(scope="module")
def samples():
    return {s["name"]: s for s in load_samples()}


async def run(sample, llm_output):
    return await evaluate.run_sample(sample, ExtractOrder(FakeLLM(llm_output)))


def failed(row):
    return row.score.failed()


async def test_ground_truth_scores_perfectly(samples):
    rows = await run_all(list(samples.values()), lambda s: ExtractOrder(FakeLLM(s["order"])))
    assert all(r.score.accuracy == 1.0 for r in rows), [(r.name, failed(r)) for r in rows]


async def test_lenient_on_case_punctuation_plurals_and_line_order(samples):
    sample = samples["05_po_in_subject"]
    got = copy.deepcopy(sample["order"])
    got["customer_name"] = "harbour lights hotel (purchasing)"
    got["customer_reference"] = "PO 44871"
    got["lines"].reverse()
    got["lines"][0]["product_description"] = "Unsalted Butter"
    got["lines"][1]["unit"] = None  # eggs had no unit in the ground truth either
    got["lines"][2]["unit"] = "Cases"
    row = await run(sample, got)
    assert failed(row) == []


async def test_wrong_values_fail_their_checks(samples):
    sample = samples["02_bulleted_list"]
    got = copy.deepcopy(sample["order"])
    got["requested_delivery_date"] = "2026-10-12"
    got["lines"][0]["quantity"] = 40
    got["lines"][1]["unit"] = "bag"
    del got["lines"][3]
    row = await run(sample, got)
    assert failed(row) == [
        "delivery_date",
        "line_count",
        "lines[0].quantity",
        "lines[1].unit",
        "lines[3].description",
        "lines[3].quantity",
        "lines[3].unit",
    ]


async def test_invented_line_breaks_line_count(samples):
    sample = samples["01_simple"]
    got = copy.deepcopy(sample["order"])
    got["lines"].append({"product_description": "basil", "quantity": 1})
    assert failed(await run(sample, got)) == ["line_count"]


async def test_estimates_need_the_flag_not_an_exact_number(samples):
    sample = samples["08_ambiguous_quantity"]
    got = copy.deepcopy(sample["order"])
    got["lines"][1]["quantity"] = 4  # "a few" read as 4 instead of 3 is fine
    assert failed(await run(sample, got)) == []

    got["lines"][1]["quantity_is_estimate"] = False
    assert failed(await run(sample, got)) == ["lines[1].quantity", "issues"]


async def test_unreadable_output_fails_everything_but_validity(samples):
    sample = samples["01_simple"]
    row = await run(sample, "not json")
    assert "parsed" in failed(row)
    assert "customer_name" in failed(row)
    assert "issues" in failed(row)


async def test_missing_order_on_no_order_email(samples):
    sample = samples["09_no_order"]
    got = copy.deepcopy(sample["order"])
    got["lines"] = [{"product_description": "mushrooms", "quantity": 1}]  # invented
    assert failed(await run(sample, got)) == ["line_count", "issues", "is_valid"]


async def test_llm_error_becomes_error_row(samples):
    row = await run(samples["01_simple"], LLMError("RateLimitError"))
    assert row.error == "RateLimitError"
    assert row.score is None


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4o-mini-2024-07-18", (1000 * 0.15 + 500 * 0.60) / 1e6),
        ("openai/gpt-4o-mini", (1000 * 0.15 + 500 * 0.60) / 1e6),
        ("gpt-4o-2024-08-06", (1000 * 2.50 + 500 * 10.00) / 1e6),
        ("some-other-model", None),
    ],
)
def test_cost_uses_longest_matching_price(model, expected):
    cost = cost_usd(model, 1000, 500)
    if expected is None:
        assert cost is None
    else:
        assert cost == pytest.approx(expected)


async def test_render_and_by_check(samples):
    good = await run(samples["01_simple"], samples["01_simple"]["order"])
    rows = [good, Row("02_bulleted_list", error="AuthenticationError: bad key")]
    report = render(rows, model="gpt-4o-mini", provider="openai", when=datetime(2026, 10, 2, 9))
    assert "`gpt-4o-mini` via openai" in report
    assert "| 01_simple | 10/10 | 100% |" in report
    assert "| 02_bulleted_list | – | – | error: AuthenticationError: bad key |" in report
    assert "| **Total** | **10/10** | **100%** |" in report
    assert "| line.quantity | 1/1 (100%) |" in report
    assert by_check([good])["customer_name"] == (1, 1)


def test_dry_run_cli_writes_report(tmp_path, capsys):
    output = tmp_path / "eval.md"
    assert evaluate.main(["--dry-run", "--output", str(output)]) == 0
    assert "**164/164** | **100%**" in output.read_text()
    assert "# Evaluation results" in capsys.readouterr().out


def test_cli_without_key_exits_2(monkeypatch, capsys):
    from order_extractor.config import Settings

    monkeypatch.setattr(evaluate, "load_settings", lambda: Settings.from_env({}))
    assert evaluate.main(["--output", "unused.md"]) == 2
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().err
