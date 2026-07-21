# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ENV POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PATH="/opt/poetry/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python -

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts

RUN poetry install --only main --no-root

FROM python:3.11-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY scripts ./scripts
COPY README.md pyproject.toml ./

CMD ["uvicorn", "reco.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
