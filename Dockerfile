FROM python:3.13-slim AS base

WORKDIR /app

# System deps for Pillow/lightgbm/faiss wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -e .

COPY . .

EXPOSE 8000

# One service, one endpoint set (§10: no microservices). Batch pipeline
# commands run via `pramaan <command>`; this default launches the API.
CMD ["python", "-m", "pramaan.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
