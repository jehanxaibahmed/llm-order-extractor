import json

import pytest

from order_extractor.adapters.llm import FakeLLM
from order_extractor.application.errors import LLMError
from order_extractor.application.extract_order import ExtractOrder
from order_extractor.config import Settings
from order_extractor.entrypoints import cli

ORDER = {
    "customer_name": "Green Leaf Café",
    "customer_reference": "PO-2231",
    "requested_delivery_date": "2026-10-08",
    "delivery_address": None,
    "lines": [
        {
            "product_description": "red peppers",
            "quantity": 2,
            "quantity_text": "2",
            "quantity_is_estimate": False,
            "unit": "box",
            "notes": None,
        },
        {
            "product_description": "basil",
            "quantity": 2,
            "quantity_text": "a couple of",
            "quantity_is_estimate": True,
            "unit": "tray",
            "notes": None,
        },
    ],
    "notes": None,
}


@pytest.fixture
def email_file(tmp_path):
    path = tmp_path / "order.txt"
    path.write_text("2 boxes of red peppers and a couple of trays of basil, Thursday please.")
    return path


def test_valid_order_prints_json_and_summary(email_file, capsys):
    llm = FakeLLM(ORDER)
    code = cli.main([str(email_file), "--today", "2026-10-01"], llm=llm)
    out, err = capsys.readouterr()

    assert code == cli.EXIT_VALID
    result = json.loads(out)
    assert result["order"]["customer_name"] == "Green Leaf Café"
    assert "Valid order: Green Leaf Café, ref PO-2231, delivery 2026-10-08, 2 lines" in err
    assert "- 2 box red peppers" in err
    assert "- 2 tray (estimate) basil" in err
    assert "warning lines[1].quantity" in err
    assert "model fake-llm" in err
    assert "Today is Thursday, 2026-10-01." in llm.calls[0].user


def test_quiet_prints_only_json(email_file, capsys):
    cli.main([str(email_file), "--quiet"], llm=FakeLLM(ORDER))
    out, err = capsys.readouterr()
    json.loads(out)
    assert err == ""


def test_invalid_order_exits_1(email_file, capsys):
    code = cli.main([str(email_file)], llm=FakeLLM({**ORDER, "lines": []}))
    _, err = capsys.readouterr()
    assert code == cli.EXIT_INVALID
    assert "Invalid order: Green Leaf Café" in err
    assert "error   lines: No order lines found." in err


def test_unreadable_model_output(email_file, capsys):
    code = cli.main([str(email_file)], llm=FakeLLM("not json"))
    out, err = capsys.readouterr()
    assert code == cli.EXIT_INVALID
    assert json.loads(out)["order"] is None
    assert "no order could be read" in err


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (lambda p: p / "missing.txt", "No such file"),
        (lambda p: (p / "order.docx").write_bytes(b"x") and p / "order.docx", "Unsupported"),
    ],
)
def test_input_errors_exit_2(tmp_path, capsys, setup, message):
    code = cli.main([str(setup(tmp_path))], llm=FakeLLM(ORDER))
    out, err = capsys.readouterr()
    assert code == cli.EXIT_FAILED
    assert out == ""
    assert message in err


def test_llm_error_exits_2(email_file, capsys):
    code = cli.main([str(email_file)], llm=FakeLLM(LLMError("AuthenticationError: bad key")))
    assert code == cli.EXIT_FAILED
    assert "error: AuthenticationError: bad key" in capsys.readouterr().err


def test_missing_api_key_exits_2(email_file, capsys, monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda: Settings.from_env({}))
    code = cli.main([str(email_file)])
    assert code == cli.EXIT_FAILED
    assert "API key is not set for provider" in capsys.readouterr().err


def test_provider_and_model_overrides(email_file, monkeypatch):
    seen = {}

    def fake_build(settings):
        seen["settings"] = settings
        return ExtractOrder(FakeLLM(ORDER))

    monkeypatch.setattr(cli, "load_settings", lambda: Settings())
    monkeypatch.setattr(cli, "build_extract_order", fake_build)
    cli.main([str(email_file), "--provider", "openrouter", "--model", "openai/gpt-4o-mini", "-q"])
    assert seen["settings"].llm_provider == "openrouter"
    assert seen["settings"].llm_model == "openai/gpt-4o-mini"


def test_bad_today_is_a_usage_error(email_file):
    with pytest.raises(SystemExit) as info:
        cli.main([str(email_file), "--today", "Thursday"], llm=FakeLLM(ORDER))
    assert info.value.code == 2
