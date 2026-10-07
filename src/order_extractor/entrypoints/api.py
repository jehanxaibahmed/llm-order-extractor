"""FastAPI app exposing the extractor over HTTP.

Run with ``uvicorn order_extractor.entrypoints.api:app --reload``; Swagger UI is at ``/docs``
(``/`` redirects there) and ReDoc at ``/redoc``.
"""

from datetime import date
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from order_extractor import __version__
from order_extractor.adapters.parsing import get_parser
from order_extractor.application.errors import (
    DocumentParseError,
    LLMError,
    UnsupportedFileTypeError,
)
from order_extractor.application.extract_order import ExtractOrder
from order_extractor.application.ports import DocumentParser
from order_extractor.config import ConfigError
from order_extractor.domain.models import ExtractionResult
from order_extractor.entrypoints.wiring import build_extract_order, load_settings

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_TEXT_CHARS = 200_000

API_DESCRIPTION = """
Send an order email or PDF; get back a structured **order**, a list of **issues** found by the
validation rules, and whether the order **is valid**.

* Errors (e.g. no order lines, missing quantity) make `is_valid` false.
* Warnings (e.g. vague quantity, delivery date in the past) need a human look but keep the
  order valid.
* The model never invents values: anything the document doesn't state is `null`.

All examples use synthetic data.
"""


TODAY_DESCRIPTION = (
    'Reference date for relative dates like "Thursday" or "tomorrow", as YYYY-MM-DD. '
    "Defaults to the server's date; pass it for repeatable results."
)
EXAMPLE_EMAIL = (
    "Hi,\n\nCould we get 3 boxes of vine tomatoes and a couple of trays of basil "
    "delivered tomorrow? PO-2231.\n\nThanks,\nMaya\nGreen Leaf Café"
)
EXAMPLE_RESULT = {
    "order": {
        "customer_name": "Green Leaf Café",
        "customer_reference": "PO-2231",
        "requested_delivery_date": "2026-10-02",
        "delivery_address": None,
        "lines": [
            {
                "product_description": "vine tomatoes",
                "quantity": 3,
                "quantity_text": "3",
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
    },
    "issues": [
        {
            "field": "lines[1].quantity",
            "severity": "warning",
            "message": 'Quantity is an estimate from vague wording ("a couple of").',
        }
    ],
    "is_valid": True,
    "model": "gpt-4o-mini-2024-07-18",
    "input_tokens": 612,
    "output_tokens": 118,
    "latency_ms": 1840,
}


class ExtractTextRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"text": EXAMPLE_EMAIL, "today": "2026-10-01"}]}
    )

    text: str = Field(max_length=MAX_TEXT_CHARS, description="The email or document text.")
    today: date | None = Field(default=None, description=TODAY_DESCRIPTION)


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    version: str = Field(examples=[__version__])


class ErrorResponse(BaseModel):
    detail: str = Field(description="What went wrong.")


def _error(description: str, example: str) -> dict:
    return {
        "model": ErrorResponse,
        "description": description,
        "content": {"application/json": {"example": {"detail": example}}},
    }


RESULT_RESPONSE = {
    200: {
        "description": "The document was processed. Problems with its content (no order "
        "lines, unreadable model output, business-rule warnings) are listed in `issues`, "
        "and `is_valid` is false if any of them is an error.",
        "content": {"application/json": {"example": EXAMPLE_RESULT}},
    }
}
LLM_RESPONSES = {
    502: _error("The LLM call failed after retries.", "RateLimitError: Rate limit reached"),
    503: _error("The LLM is not configured.", "OPENAI_API_KEY is not set (LLM_PROVIDER=openai)."),
}
FILE_RESPONSES = {
    400: _error(
        "The file could not be read.",
        "PDF has no extractable text. It may be scanned; OCR is not supported yet.",
    ),
    413: _error("The file is larger than 5 MB.", "File is larger than 5 MB."),
    415: _error(
        "Unsupported file type.", "Unsupported file type '.docx'; expected one of .eml, .pdf, .txt."
    ),
}


@lru_cache
def get_extract_order() -> ExtractOrder:
    """Dependency: one use case (and one HTTP client) per process. Tests override this."""
    return build_extract_order(load_settings())


ExtractOrderDep = Annotated[ExtractOrder, Depends(get_extract_order)]
UploadDep = Annotated[UploadFile, File(description="A .txt, .eml or text-based .pdf file.")]


def get_upload_parser(file: UploadDep) -> DocumentParser:
    """Dependency: reject unsupported file types (415) before the LLM client is even built."""
    return get_parser(file.filename or "")


api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(
    api_key: str | None = Security(APIKeyHeader(name="X-API-Key", auto_error=False)),
) -> str:
    secret = load_settings().api_key_secret
    if secret and api_key != secret:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key or ""


ApiKeyDep = Annotated[str, Depends(verify_api_key)]


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Order Extractor",
        version=__version__,
        summary="Turn unstructured order emails and PDFs into validated order JSON.",
        description=API_DESCRIPTION,
        contact={"name": "Jahanzaib Ahmad", "url": "https://github.com/jehanxaibahmed"},
        openapi_tags=[
            {"name": "extraction", "description": "Extract an order from text or a file."},
            {"name": "health", "description": "Service status."},
        ],
        swagger_ui_parameters={
            "tryItOutEnabled": True,
            "displayRequestDuration": True,
            "defaultModelsExpandDepth": 0,
        },
    )

    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health", tags=["health"], summary="Health check")
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.post(
        "/extract/text",
        tags=["extraction"],
        summary="Extract an order from text",
        responses={**RESULT_RESPONSE, **LLM_RESPONSES},
        dependencies=[Depends(verify_api_key)],
    )
    async def extract_text(body: ExtractTextRequest, extract: ExtractOrderDep) -> ExtractionResult:
        """Paste the body of an order email. The text is sent to the configured LLM."""
        return await extract.from_text(body.text, body.today or date.today())

    @app.post(
        "/extract/file",
        tags=["extraction"],
        summary="Extract an order from a file",
        responses={**RESULT_RESPONSE, **FILE_RESPONSES, **LLM_RESPONSES},
        dependencies=[Depends(verify_api_key)],
    )
    async def extract_file(
        file: UploadDep,
        parser: Annotated[DocumentParser, Depends(get_upload_parser)],
        extract: ExtractOrderDep,
        today: Annotated[date | None, Form(description=TODAY_DESCRIPTION)] = None,
    ) -> ExtractionResult:
        """Upload a `.txt` email, an `.eml` file or a text-based PDF (max 5 MB).

        For `.eml` files the From, Date and Subject headers are kept, since the subject often
        holds the PO number. Scanned PDFs are rejected with 400 (OCR is not supported yet).
        """
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        # PDF parsing is CPU-bound; keep it off the event loop.
        text = await run_in_threadpool(parser.parse, data)
        return await extract.from_text(text, today or date.today())

    _add_error_handlers(app)
    return app


def _add_error_handlers(app: FastAPI) -> None:
    statuses: list[tuple[type[Exception], int]] = [
        (UnsupportedFileTypeError, 415),
        (DocumentParseError, 400),
        (LLMError, 502),
        (ConfigError, 503),
    ]
    for exc_type, status in statuses:

        async def handler(request: Request, exc: Exception, status: int = status) -> JSONResponse:
            return JSONResponse(status_code=status, content={"detail": str(exc)})

        app.add_exception_handler(exc_type, handler)


app = create_app()
