# 🧾 LLM Order Extractor

![Status](https://img.shields.io/badge/status-in%20progress-orange?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white) ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white) ![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

> Turn unstructured customer emails and PDF orders into validated, structured order JSON using large language models.

## 🎯 Why this project

Businesses still receive many orders as free-text emails and attachments. This project extracts the customer, order lines, quantities and delivery date, validates them against a schema, and returns clean JSON ready for any ERP or ordering system.

## 🧱 Planned stack

- Python 3.12 and FastAPI for the REST API
- OpenAI and OpenRouter models with structured JSON output
- Pydantic schemas for validation
- PDF text extraction and OCR for scanned documents
- Docker for local and cloud deployment

## 🗺️ Roadmap

- [ ] Email and PDF ingestion endpoint
- [ ] Schema-first extraction with Pydantic
- [ ] Validation and confidence scores per field
- [ ] Model fallback and retries
- [ ] Sample dataset of synthetic orders
- [ ] Docker image and deployment guide

## 📌 Status

🚧 This project is in early development. Code is coming soon. It uses synthetic sample data only.

---

Built by [Jahanzaib Ahmad](https://github.com/jehanxaibahmed) · Full Stack Engineer · AI & LLM Systems
