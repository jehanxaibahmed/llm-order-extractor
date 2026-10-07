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
- Tests that run **without** an API key (simulatored LLM)
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

## 3. Architecture and folder structure

The code follows **ports and adapters** (hexagonal / clean architecture). Dependencies point
inwards only:

```
entrypoints  ──►  adapters  ──►  application  ──►  domain
 (API, CLI)       (OpenAI,       (use case,        (models,
                   parsers)       prompts, ports)   rules)
```

| Layer | Contains | May import |
|---|---|---|
| `domain/` | Pydantic models, validation rules | nothing else in the package; no I/O libraries |
| `application/` | `ExtractOrder` use case, prompts, **ports** (`LLMClient`, `DocumentParser` protocols) | `domain` |
| `adapters/` | concrete implementations of the ports: OpenAI/OpenRouter/Fake LLM clients, text/.eml/PDF parsers | `domain`, `application`, `config` |
| `entrypoints/` | FastAPI app, CLI — the **composition root** that builds adapters and injects them | everything |

Patterns used:
- **Ports and adapters** — the use case depends on protocols, never on `openai` or `pypdf`
- **Strategy** — LLM providers and document parsers are interchangeable implementations of a port
- **Factory** — `create_llm_client(settings)` picks OpenAI / OpenRouter / Fake from config
- **Registry** — `get_parser(filename)` picks a parser by file extension
- **Dependency injection** — `ExtractOrder(llm=...)` receives its client; the API wires it with
  FastAPI `Depends`, so tests swap in `SyntheticLLM` without patching

`tests/test_architecture.py` reads every module's imports and fails if a layer breaks these rules.

```
llm-order-extractor/
├── README.md
├── PLAN.md
├── pyproject.toml
├── .env.example                    # OPENAI_API_KEY=, OPENROUTER_API_KEY=, LLM_PROVIDER=openai, LLM_MODEL=gpt-4o-mini
├── .gitignore                      # .env, __pycache__, .venv
├── src/order_extractor/
│   ├── __init__.py
│   ├── config.py                   # Settings from env / .env
│   ├── domain/
│   │   ├── models.py               # Order, OrderLine, ValidationIssue, ExtractionResult
│   │   └── validation.py           # business rules → list[ValidationIssue]
│   ├── application/
│   │   ├── ports.py                # LLMClient, DocumentParser protocols; LLMResponse
│   │   ├── errors.py               # DocumentParseError, UnsupportedFileTypeError, ...
│   │   ├── prompts.py              # system + user prompt templates
│   │   └── extract_order.py        # ExtractOrder use case: text → LLM → parse → validate
│   ├── adapters/
│   │   ├── llm/
│   │   │   ├── __init__.py         # create_llm_client(settings) factory
│   │   │   ├── schema.py           # to_strict_schema() for OpenAI strict mode
│   │   │   ├── openai_client.py    # OpenAIClient, OpenRouterClient (same SDK, different base_url)
│   │   │   └── synthetic.py             # SyntheticLLM for tests
│   │   └── parsing/
│   │       ├── __init__.py         # get_parser(filename) registry
│   │       ├── text.py             # .txt, plus decoding / clean-up shared by all parsers
│   │       ├── eml.py              # .eml: From/Date/Subject + body (HTML → text, forwarded parts)
│   │       └── pdf.py              # text-based PDF via pypdf
│   └── entrypoints/
│       ├── api.py                  # FastAPI app
│       └── cli.py                  # `python -m order_extractor.entrypoints.cli path/to/file`
├── samples/
│   ├── emails/                     # 8–10 synthetic .txt / .eml files
│   ├── pdfs/                       # 2–3 synthetic PDFs
│   └── expected/                   # expected JSON for each sample (ground truth)
├── tests/                          # mirrors src/: domain/, application/, adapters/, entrypoints/
│   └── test_architecture.py        # enforces the layer dependency rules
└── scripts/
    └── evaluate.py                 # runs samples through real LLM, prints accuracy table
```

---

## 4. Data model (`domain/models.py`)

The models are deliberately **permissive**: they describe what the LLM returned, not what a
good order looks like. Business rules (at least one line, quantity > 0, …) live in
`domain/validation.py`, so a bad order produces a `ValidationIssue` instead of a Pydantic exception.

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

## 5. Prompt design (`application/prompts.py`)

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
directly — the OpenAI adapter converts it with `to_strict_schema()` (`adapters/llm/schema.py`).
That quirk belongs to the OpenAI adapter, so the domain model stays provider-neutral.

---

## 6. LLM layer (`application/ports.py` + `adapters/llm/`)

```python
# application/ports.py
class LLMClient(Protocol):
    async def extract(self, system: str, user: str, schema: dict) -> LLMResponse: ...

# adapters/llm/
class OpenAIClient: ...        # base_url default
class OpenRouterClient: ...    # same SDK, base_url="https://openrouter.ai/api/v1"
class SyntheticLLM: ...             # returns canned JSON for tests
```

- `LLMResponse` holds `text`, `model`, `input_tokens`, `output_tokens`
- Provider and model chosen from env vars via `config.py` and the `create_llm_client()` factory
- Timeout + 2 retries with backoff — delegated to the `openai` SDK (`timeout`, `max_retries`), which
  retries connection errors, 408/409/429 and 5xx with exponential backoff
- SDK errors, refusals and empty replies are raised as `LLMError` (`application/errors.py`)
- Structured outputs default on for OpenAI, off for OpenRouter (`LLM_STRUCTURED_OUTPUT` overrides)
- v0.2 idea: ordered fallback list of models (mirrors the production pattern)

---

## 7. Validation rules (`domain/validation.py`)

Return issues, don't raise:
- **error**: no order lines; quantity missing or ≤ 0; unparseable JSON / output that doesn't match the schema
- **warning**: quantity above a threshold (e.g. > 500); `quantity_is_estimate` is true; delivery date in the past; delivery date > 60 days away; missing customer name; duplicate product lines

These checks only work because the schema doesn't enforce them (see §4) — keep it that way.

`ExtractOrder` (`application/extract_order.py`) adds the issues that come before these rules:
empty document (error, LLM not called), document over 50,000 characters (truncated, warning),
output that isn't JSON or doesn't fit the schema (error, `order: null`, one issue per bad field).
LLM and file-parsing failures are raised, not returned, so the API can map them to HTTP errors.
Date rules compare against the same `today` that was given to the prompt.

`is_valid = no errors`

---

## 8. API and CLI

**API** (`entrypoints/api.py`) — `uvicorn order_extractor.entrypoints.api:app --reload`
- `POST /extract/text` — body `{ "text": "...", "today": "2026-10-01" }` (`today` optional)
- `POST /extract/file` — multipart upload (.txt, .eml, .pdf, max 5 MB), optional `today` form field
- `GET /health`
- **Swagger UI** at `/docs` (`/` redirects there) and ReDoc at `/redoc`: endpoints grouped by tag,
  example request and response, every error status documented with an example, field
  descriptions from the model docstrings, "Try it out" on by default
- Returns `ExtractionResult` with **200** whenever the document was processed — content problems
  are in `issues`. Errors: **415** unsupported file type, **400** unreadable file, **413** too large,
  **422** bad request body, **502** LLM call failed, **503** LLM not configured (missing key)
- The use case is a FastAPI dependency (`get_extract_order`); tests override it with `SyntheticLLM`

**CLI** (`entrypoints/cli.py`) — installed as `order-extractor`
```bash
order-extractor samples/emails/01_simple.txt --today 2026-10-01
order-extractor po.pdf --provider openrouter --model openai/gpt-4o-mini --quiet
```
JSON result on stdout (pipeable), human summary on stderr. `--today` defaults to the current
date; pass it explicitly for repeatable runs (the eval script always does). Exit codes: **0**
valid order, **1** order has errors, **2** could not run (bad file, config or LLM failure).

Both entrypoints build the use case through `entrypoints/wiring.py`, which loads `.env`.

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

Built: 9 emails (`.txt` and `.eml`) and 2 PDFs, with today = Thursday 2026-10-01. See
`samples/README.md` for the list and the expected-file format. PDFs are generated by
`python -m scripts.make_sample_pdfs`. `tests/test_samples.py` checks that the ground truth
matches the validation rules and only contains values present in each document.

---

## 10. Tests (no API key needed)

`tests/` mirrors `src/`:
- `domain/test_models.py` — valid/invalid models
- `domain/test_validation.py` — each rule triggers correctly
- `adapters/test_llm_schema.py` — strict-mode schema conversion
- `adapters/test_parsing.py` — .eml body extraction, PDF text extraction
- `application/test_extract_order.py` — `SyntheticLLM` returns canned JSON; check result + issues; check bad JSON handled
- `entrypoints/test_api.py` — FastAPI `TestClient` against both endpoints, `SyntheticLLM` injected via `Depends` override
- `test_architecture.py` — layer dependency rules

Target: `pytest` green, plus a GitHub Actions workflow that runs `ruff` + `pytest` on every push.

Built: `.github/workflows/ci.yml` runs on pushes to `main` and on every pull request. A lint
job runs `ruff check` + `ruff format --check`, and a test job runs `pytest`, the evaluation
`--dry-run` and a CLI smoke test on Python 3.11–3.14. No secrets are configured, so CI can never
call a real LLM. Dependabot checks actions and pip dependencies monthly.

---

## 11. Evaluation script (`scripts/evaluate.py`)

- Runs every sample through the real LLM (needs an API key)
- Compares to `expected/` per field: customer, date, line count, each line's quantity + description
- Prints a table: sample, field accuracy, issues, tokens, latency, cost estimate
- Writes `eval_results.md` so the README can show real numbers

Built: `python -m scripts.evaluate [--provider] [--model] [--dry-run] [--output]`.
- Checks per sample: parsed, customer name, reference, delivery date, line count, and per line
  description / quantity / unit; plus the exact issue set and `is_valid`
- Lenient where wording varies: case, punctuation and plurals are ignored, lines are matched
  by description (order doesn't matter), and estimated quantities only need the estimate flag
- `--dry-run` replays the ground truth through `SyntheticLLM` (no key, must score 100%)
- Exits 1 if any sample errored, 2 if the LLM isn't configured
- First real results are pending: no live API calls until explicitly approved

This sets up the next project, **llm-eval-harness**.

---

## 12. Milestones (one commit or PR each)

1. Project skeleton: `pyproject.toml`, structure, `.env.example`, ruff, empty tests pass
2. Layered package structure + domain models + validation rules + tests
3. Parser adapters (text, .eml, PDF) + registry + tests
4. Config, prompts, LLM adapters (OpenAI, OpenRouter, SyntheticLLM) + factory
5. `ExtractOrder` use case + tests with SyntheticLLM
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

Built in v0.1.0, except the accuracy table: it is a marked placeholder until the first live
evaluation run is approved. The README example output is the sample's ground truth passed
through the real validation step, and is labelled as such.

---

## 14. Prompt to give Claude Code

> Read PLAN.md in this repo. Build milestone 1 only, then stop and show me what you did. Use Python 3.11, FastAPI, Pydantic v2 and the openai SDK. All sample data must be synthetic — made-up businesses only, never real Nation Wilcox emails or orders. Tests must run without an API key. Commit with a clear message after the milestone passes `ruff` and `pytest`.

Then continue one milestone at a time.
