# Build the project and its demo dependencies in an isolated environment.
FROM python:3.11-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[demo]"


# Keep the runtime image free of compilers, test tools, and neural frameworks.
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HOME=/home/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run Streamlit as an unprivileged account with writable app and home directories.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
    && mkdir -p /home/app/.streamlit \
    && chown -R 10001:10001 /app /home/app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=10001:10001 demo/ ./demo/

USER 10001:10001

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "demo/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
