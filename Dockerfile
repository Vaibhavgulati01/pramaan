# Multi-stage so the runtime image does not carry build tooling.
FROM python:3.13-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install ".[provenance]"


FROM python:3.13-slim

WORKDIR /app

# libgomp is required by LightGBM at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

# Determinism (docs/EVALUATION_PROTOCOL.md): single-threaded BLAS/OpenMP
# so results are reproducible regardless of the host's core count.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PRAMAAN_SCALE=dev \
    PYTHONUNBUFFERED=1

# Run as a non-root user: the service accepts uploaded images from
# untrusted callers, so a decoder bug should not land as root.
RUN useradd --create-home --uid 10001 pramaan && chown -R pramaan /app
USER pramaan

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

# One service, one endpoint set (Sec.10: no microservices). Batch
# commands run via `docker run ... pramaan <command>`.
CMD ["python", "-m", "pramaan.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
