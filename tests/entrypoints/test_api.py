from datetime import date

import pytest
from fastapi.testclient import TestClient

from order_extractor import __version__
from order_extractor.adapters.llm import FakeLLM
from order_extractor.application.errors import LLMError
from order_extractor.application.extract_order import ExtractOrder
from order_extractor.config import ConfigError
from order_extractor.entrypoints import api

ORDER = {
    "customer_name": "Fresh Farm Ltd",
    "customer_reference": "PO-7781",
    "requested_delivery_date": "2026-10-08",
    "delivery_address": None,
    "lines": [
        {
            "product_description": "red peppers",
            "quantity": 10,
            "quantity_text": "10",
            "quantity_is_estimate": False,
            "unit": "box",
            "notes": None,
        }
    ],
    "notes": None,
}
EMAIL = "Please send 10 boxes of red peppers on Thursday 8th. PO-7781. Fresh Farm Ltd"


@pytest.fixture
def fake_llm():
    return FakeLLM(ORDER)


@pytest.fixture
def client(fake_llm):
    app = api.create_app()
    app.dependency_overrides[api.get_extract_order] = lambda: ExtractOrder(fake_llm)
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_extract_text(client, fake_llm):
    response = client.post("/extract/text", json={"text": EMAIL, "today": "2026-10-01"})
    assert response.status_code == 200
    body = response.json()
    assert body["is_valid"] is True
    assert body["order"]["customer_reference"] == "PO-7781"
    assert body["order"]["lines"][0]["quantity"] == 10
    assert body["model"] == "fake-llm"
    assert "Today is Thursday, 2026-10-01." in fake_llm.calls[0].user


def test_extract_text_defaults_today_to_server_date(client, fake_llm):
    client.post("/extract/text", json={"text": EMAIL})
    assert date.today().isoformat() in fake_llm.calls[0].user


def test_extract_text_returns_issues_with_200(client):
    response = client.post("/extract/text", json={"text": "   ", "today": "2026-10-01"})
    assert response.status_code == 200
    body = response.json()
    assert body["is_valid"] is False
    assert body["issues"][0]["field"] == "text"


@pytest.mark.parametrize(
    "payload",
    [{}, {"text": 123}, {"text": "x", "today": "next Thursday"}, {"text": "x" * 200_001}],
)
def test_extract_text_rejects_bad_requests(client, payload):
    assert client.post("/extract/text", json=payload).status_code == 422


def test_extract_file_txt(client, fake_llm):
    response = client.post(
        "/extract/file",
        files={"file": ("order.txt", EMAIL.encode(), "text/plain")},
        data={"today": "2026-10-01"},
    )
    assert response.status_code == 200
    assert response.json()["is_valid"] is True
    assert "10 boxes of red peppers" in fake_llm.calls[0].user


def test_extract_file_pdf(client, fake_llm, make_pdf):
    pdf = make_pdf([["PURCHASE ORDER PO-7781", "10 x red peppers (box)"]])
    response = client.post("/extract/file", files={"file": ("po.pdf", pdf, "application/pdf")})
    assert response.status_code == 200
    assert "PURCHASE ORDER PO-7781" in fake_llm.calls[0].user


def test_extract_file_unsupported_type(client, fake_llm):
    response = client.post("/extract/file", files={"file": ("order.docx", b"x", "application/x")})
    assert response.status_code == 415
    assert ".eml, .pdf, .txt" in response.json()["detail"]
    assert fake_llm.calls == []


def test_extract_file_unreadable_pdf(client):
    response = client.post(
        "/extract/file", files={"file": ("po.pdf", b"not a pdf", "application/pdf")}
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_extract_file_too_large(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)
    response = client.post("/extract/file", files={"file": ("order.txt", b"x" * 11, "text/plain")})
    assert response.status_code == 413


def test_extract_file_requires_a_file(client):
    assert client.post("/extract/file", data={"today": "2026-10-01"}).status_code == 422


def test_llm_failure_is_502():
    app = api.create_app()
    app.dependency_overrides[api.get_extract_order] = lambda: ExtractOrder(
        FakeLLM(LLMError("RateLimitError: slow down"))
    )
    response = TestClient(app).post("/extract/text", json={"text": EMAIL})
    assert response.status_code == 502
    assert response.json() == {"detail": "RateLimitError: slow down"}


def test_missing_configuration_is_503(monkeypatch):
    def not_configured():
        raise ConfigError("OPENAI_API_KEY is not set (LLM_PROVIDER=openai).")

    app = api.create_app()
    app.dependency_overrides[api.get_extract_order] = not_configured
    response = TestClient(app).post("/extract/text", json={"text": EMAIL})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_openapi_schema_lists_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/extract/text", "/extract/file"} <= set(paths)


def test_unsupported_file_is_415_even_when_llm_not_configured():
    def not_configured():
        raise ConfigError("OPENAI_API_KEY is not set (LLM_PROVIDER=openai).")

    app = api.create_app()
    app.dependency_overrides[api.get_extract_order] = not_configured
    response = TestClient(app).post(
        "/extract/file", files={"file": ("order.docx", b"x", "application/x")}
    )
    assert response.status_code == 415
