# Multi-stage build for the GMAI inference service.
#
# The training stack (gymnasium, matplotlib, the CUDA build of torch) is an
# order of magnitude larger than what serving a move actually needs, so the
# runtime image installs the CPU wheel of torch and the API dependencies only.
# See Dockerfile.train for the training image.

# --------------------------------------------------------------- build stage
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# CPU-only torch: the CUDA wheel is ~2.5 GB and useless for single-position
# inference, which is dominated by Python overhead rather than matmul.
RUN pip install --no-cache-dir \
      torch==2.*  --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
      "python-chess>=1.10" \
      "numpy>=1.24" \
      "fastapi>=0.110" \
      "uvicorn[standard]>=0.27" \
      "prometheus-client>=0.20" \
      "pydantic>=2.0"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# ------------------------------------------------------------- runtime stage
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="GMAI" \
      org.opencontainers.image.description="Search-free deep RL chess agent for forced-mate endgames" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GMAI_DEVICE=cpu

# Non-root: the service reads a checkpoint and answers HTTP, nothing more.
RUN useradd --create-home --uid 10001 gmai
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Bake in a checkpoint at build time with:
#   docker build --build-arg CHECKPOINT=runs/<run>/final.pt .
# or mount one at runtime and set GMAI_CHECKPOINT.
COPY models/ /app/models/
ENV GMAI_CHECKPOINT=/app/models/final.pt

USER gmai
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "gmai.api:app", "--host", "0.0.0.0", "--port", "8000"]
