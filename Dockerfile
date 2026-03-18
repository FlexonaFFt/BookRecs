FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    UV_CACHE_DIR=/tmp/uv-cache

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock /build/
RUN uv export --frozen --no-dev --no-emit-project --format requirements.txt -o /build/requirements.txt
RUN pip install --no-cache-dir -r /build/requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY source /app/source
COPY data /app/data
COPY .env.example /app/.env.example

CMD ["python", "-m", "source.interfaces.pipeline_entrypoint"]
