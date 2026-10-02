# llm-order-extractor — Build Plan

Turn unstructured order emails and PDFs into validated, structured order JSON using LLMs.
This is a public portfolio project: **use synthetic sample data only** — nothing from any employer.

> **Data rule (applies to every milestone):** all sample emails, PDFs, fixtures, test data and
> docs must be invented — made-up businesses such as "Green Leaf Café" or "Fresh Farm Ltd",
> made-up addresses, PO numbers and products. **Never** use real Nation Wilcox emails, orders,
> customers or products, and never copy from any employer codebase. The repo is public.

---

## 1. Goal and scope (v0.1)

**In scope**
- Input: plain-text email, `.eml` file, or text-based PDF
- Output: an `Order` JSON object (customer, lines, delivery date, notes) plus a confidence/validation report
- One LLM provider interface with two backends: OpenAI and OpenRouter (any model)
- FastAPI endpoint + CLI
- Validation rules that flag problems instead of guessing
- Tests that run **without** an API key (mocked LLM)
- A small evaluation script that measures accuracy on the sample set

**Out of scope for v0.1** (put on the roadmap)
- Scanned PDFs / OCR, voicemail, product catalogue matching, UI, database

---

## 2. Tech stack

- Python 3.11+
- FastAPI + Uvicorn
- Pydantic v2 (schemas and validation)
- `openai` Python SDK (works for OpenAI and, via `base_url`, OpenRouter)
- `pypdf` for PDF text extraction
- `pytest` + `pytest-asyncio`
- `ruff` for linting
- Dependency management: plain `venv` + `pip`, dependencies declared in `pyproject.toml`
  (`pip install -e ".[dev]"`). `uv` works too but isn't required.

---

## 3. Folder structure

```
llm-order-extractor/
├── README.md
├── PLAN.md
├── pyproject.toml
├── .env.example                # OPENAI_API_KEY=, OPENROUTER_API_KEY=, LLM_PROVIDER=openai, LLM_MODEL=gpt-4o-mini
├── .gitignore                  # .env, __pycache__, .venv
├── src/order_extractor/
│   ├── __init__.py
│   ├── schemas.py              # Pydantic models
│   ├── parsing.py              # email / .eml / PDF → plain text
│   ├── prompts.py              # system + user prompt templates
│   ├── llm.py                  # provider interface + OpenAI/OpenRouter clients + fake client
│   ├── extractor.py            # orchestrates: text → LLM → parse → validate
│   ├── validation.py           # business rules → ValidationReport
│   ├── api.py                  # FastAPI app
│   └── cli.py                  # `python -m order_extractor.cli path/to/file`
├── samples/
│   ├── emails/                 # 8–10 synthetic .txt / .eml files
│   ├── pdfs/                   # 2–3 synthetic PDFs
│   └── expected/               # expected JSON for each sample (ground truth)
├── tests/
│   ├── test_schemas.py
│   ├── test_parsing.py
│   ├── test_validation.py
│   ├── test_extractor.py       # uses FakeLLM, no network
│   └── test_api.py
└── scripts/
    └── evaluate.py             # runs samples through real LLM, prints accuracy table
```

---

## 4. Data model (`schemas.py`)

The models are deliberately **permissive**: they describe what the LLM returned, not what a
good order looks like. Business rules (at least one line, quantity > 0, …) live in
`validation.py`, so a bad order produces a `ValidationIssue` instead of a Pydantic exception.

```python
class OrderLine(BaseModel):
    product_description: str              # as written by the customer
    quantity: float | None                # no gt=0 here — validation.py checks it
    quantity_text: str | None             # original wording, e.g. "a couple of"
    quantity_is_estimate: bool = False    # true when the text was vague ("a couple", "a few")
    unit: str | None                      # "box", "kg", "tray", "case"
    notes: str | None

class Order(BaseModel):
    customer_name: str | None
    customer_reference: str | None        # PO number if given
    requested_delivery_date: date | None
    delivery_address: str | None
    lines: list[OrderLine]                # may be empty — validation.py flags it
    notes: str | None

class ValidationIssue(BaseModel):
    field: str
    severity: Literal["error", "warning"]
    message: str

class ExtractionResult(BaseModel):
    order: Order | None
    issues: list[ValidationIssue]
    is_valid: bool
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int
```

---

## 5. Prompt design (`prompts.py`)

System prompt rules:
- Extract only what the text states; **never invent** products, quantities or dates
- Use `null` for anything missing
- Convert relative dates ("next Thursday") to ISO dates using the provided `today` date
- Keep product descriptions exactly as written (matching happens later, not here)
- Return JSON only, matching the provided schema

- When a quantity is vague ("a couple of cases", "a few trays"), give the best numeric reading,
  copy the wording into `quantity_text` and set `quantity_is_estimate: true`

Use OpenAI **structured outputs** (`response_format` with a JSON schema, `strict: true`) where
supported; fall back to "JSON only" instructions + `Order.model_validate_json()` for other models.

Strict mode only accepts a subset of JSON Schema: every property must be listed in `required`
(optional ones are nullable instead), `additionalProperties` must be `false`, and some keywords
such as numeric/length limits are not supported. So don't pass `Order.model_json_schema()`
directly — build an LLM-facing schema with a helper (`llm_schema()` in `schemas.py`) that
produces a strict-compatible version, and test that helper.

---

## 6. LLM layer (`llm.py`)

```python
class LLMClient(Protocol):
    async def extract(self, system: str, user: str, schema: dict) -> LLMResponse: ...

class OpenAIClient: ...        # base_url default
class OpenRouterClient: ...    # same SDK, base_url="https://openrouter.ai/api/v1"
class FakeLLM: ...             # returns canned JSON for tests
```

- `LLMResponse` holds `text`, `model`, `input_tokens`, `output_tokens`
- Provider and model chosen from env vars
- Timeout + 2 retries with backoff
- v0.2 idea: ordered fallback list of models (mirrors the production pattern)

---

## 7. Validation rules (`validation.py`)

Return issues, don't raise:
- **error**: no order lines; quantity missing or ≤ 0; unparseable JSON / output that doesn't match the schema
- **warning**: quantity above a threshold (e.g. > 500); `quantity_is_estimate` is true; delivery date in the past; delivery date > 60 days away; missing customer name; duplicate product lines

These checks only work because the schema doesn't enforce them (see §4) — keep it that way.
Date rules compare against the same `today` that was given to the prompt.

`is_valid = no errors`

---

## 8. API and CLI

**API** (`api.py`)
- `POST /extract/text` — body `{ "text": "...", "today": "2026-10-01" }`
- `POST /extract/file` — multipart upload (.txt, .eml, .pdf), optional `today` form field
- `GET /health`
- Returns `ExtractionResult`

**CLI** (`cli.py`)
```bash
python -m order_extractor.cli samples/emails/01_simple.txt --today 2026-10-01
```
Prints the JSON result and a short summary. `--today` defaults to the current date; pass it
explicitly for repeatable runs (the eval script always does).

---

## 9. Sample data (synthetic)

Create 8–10 emails that cover real-world messiness, each with a matching `expected/*.json`:
1. Simple single-line order
2. Multiple lines in a bulleted list
3. Lines written as a paragraph ("2 boxes of red peppers and a tray of basil")
4. Relative delivery date ("Thursday please")
5. PO number in the subject line
6. Forwarded email with a long signature and disclaimer
7. Order + unrelated chit-chat
8. Ambiguous quantity ("a couple of cases") → expect `quantity_is_estimate: true` and a warning
9. Email with no order at all → expect `lines: []` and an error (not a crash)
10. PDF purchase order (generate with a simple script)

Use made-up businesses (e.g. "Green Leaf Café", "Fresh Farm Ltd") — see the data rule at the top.
Each sample's expected file also records the `today` date it was written against.

---

## 10. Tests (no API key needed)

- `test_schemas.py` — valid/invalid models
- `test_parsing.py` — .eml body extraction, PDF text extraction
- `test_validation.py` — each rule triggers correctly
- `test_extractor.py` — `FakeLLM` returns canned JSON; check result + issues; check bad JSON handled
- `test_api.py` — FastAPI `TestClient` against both endpoints

Target: `pytest` green, plus a GitHub Actions workflow that runs `ruff` + `pytest` on every push.

---

## 11. Evaluation script (`scripts/evaluate.py`)

- Runs every sample through the real LLM (needs an API key)
- Compares to `expected/` per field: customer, date, line count, each line's quantity + description
- Prints a table: sample, field accuracy, issues, tokens, latency, cost estimate
- Writes `eval_results.md` so the README can show real numbers

This sets up the next project, **llm-eval-harness**.

---

## 12. Milestones (one commit or PR each)

1. Project skeleton: `pyproject.toml`, structure, `.env.example`, ruff, empty tests pass
2. Schemas + validation rules + tests
3. Parsing (text, .eml, PDF) + tests
4. Prompts + LLM layer (OpenAI, OpenRouter, FakeLLM)
5. Extractor orchestration + tests with FakeLLM
6. FastAPI + CLI + tests
7. Synthetic samples + expected outputs
8. Evaluation script + first results
9. GitHub Actions CI
10. README update: change badge to "v0.1", add usage, example output, eval results table, architecture diagram

---

## 13. README update (end of v0.1)

- Replace "🚧 In progress" with a version badge and a CI badge
- Align the stack section with what was built: Python 3.11+ (not 3.12), OCR and Docker moved to the roadmap
- Quick start (install, set `.env`, run API, run CLI)
- Example input email → example JSON output
- Accuracy table from `eval_results.md`
- Roadmap: OCR for scanned PDFs, model fallback chain, product matching (links to **semantic-product-matcher**), voice input (links to **voice-to-order**)

---

## 14. Prompt to give Claude Code

> Read PLAN.md in this repo. Build milestone 1 only, then stop and show me what you did. Use Python 3.11, FastAPI, Pydantic v2 and the openai SDK. All sample data must be synthetic — made-up businesses only, never real Nation Wilcox emails or orders. Tests must run without an API key. Commit with a clear message after the milestone passes `ruff` and `pytest`.

Then continue one milestone at a time.
