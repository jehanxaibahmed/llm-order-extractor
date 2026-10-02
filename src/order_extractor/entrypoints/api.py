"""FastAPI app exposing the extractor over HTTP.

Run with ``uvicorn order_extractor.entrypoints.api:app --reload``.
"""

from datetime import date
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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


class ExtractTextRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_CHARS)
    today: date | None = None
    """Reference date for relative dates like "Thursday". Defaults to the server's date."""


class HealthResponse(BaseModel):
    status: str
    version: str


@lru_cache
def get_extract_order() -> ExtractOrder:
    """Dependency: one use case (and one HTTP client) per process. Tests override this."""
    return build_extract_order(load_settings())


ExtractOrderDep = Annotated[ExtractOrder, Depends(get_extract_order)]
UploadDep = Annotated[UploadFile, File(description="A .txt, .eml or text-based .pdf file")]


def get_upload_parser(file: UploadDep) -> DocumentParser:
    """Dependency: reject unsupported file types (415) before the LLM client is even built."""
    return get_parser(file.filename or "")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Order Extractor",
        version=__version__,
        description="Turn unstructured order emails and PDFs into validated order JSON.",
    )

    @app.get("/health")
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.post("/extract/text")
    async def extract_text(body: ExtractTextRequest, extract: ExtractOrderDep) -> ExtractionResult:
        return await extract.from_text(body.text, body.today or date.today())

    @app.post("/extract/file")
    async def extract_file(
        file: UploadDep,
        parser: Annotated[DocumentParser, Depends(get_upload_parser)],
        extract: ExtractOrderDep,
        today: Annotated[date | None, Form()] = None,
    ) -> ExtractionResult:
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
