FROM python:3.12-slim

# Install curl so we can do a healtcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# GENAI COMMENT:
#   uv resolves and installs the whole dependency set (scikit-learn, OTel, FastAPI)
#   faster than pip, and reads pyproject.toml directly so there is no separate
#   requirements.txt to keep in sync.
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .
RUN uv pip install --system --no-cache -r pyproject.toml --all-extras

COPY . .

ARG BUILD_SHA=dev
ENV BUILD_SHA=$BUILD_SHA

EXPOSE 8080

ENV PYTHONOPTIMIZE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
