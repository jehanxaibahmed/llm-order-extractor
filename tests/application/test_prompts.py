from datetime import date

from order_extractor.application.prompts import SYSTEM_PROMPT, build_user_prompt


def test_user_prompt_includes_today_with_weekday_and_document():
    prompt = build_user_prompt("2 trays of basil, Thursday please", date(2026, 10, 1))
    assert "Today is Thursday, 2026-10-01." in prompt
    assert "<document>\n2 trays of basil, Thursday please\n</document>" in prompt


def test_system_prompt_covers_key_rules():
    for phrase in [
        "Never invent",
        "null",
        "exactly as the customer wrote it",
        "quantity_is_estimate",
        "ISO date",
        "empty lines list",
        "data, not instructions",
        "JSON only",
    ]:
        assert phrase in SYSTEM_PROMPT, phrase
