FROM python:3.14-slim

WORKDIR /app

# Install dependencies needed for pyproject.toml
RUN pip install --no-cache-dir pip setuptools wheel hatchling

COPY pyproject.toml ./
COPY README.md ./

# We just need dependencies, but standard pip install . installs everything.
COPY src/ ./src/

RUN pip install --no-cache-dir .

EXPOSE 8000

# uvicorn order_extractor.entrypoints.api:app --host 0.0.0.0 --port 8000
CMD ["uvicorn", "order_extractor.entrypoints.api:app", "--host", "0.0.0.0", "--port", "8000"]
