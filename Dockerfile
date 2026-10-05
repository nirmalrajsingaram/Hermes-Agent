# ============================================================
# Hermes Agent — Multi-stage Docker build
# ============================================================

# ---- Stage 1: dependency builder ----
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ---- Stage 2: runtime image ----
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Hermes Agent"
LABEL org.opencontainers.image.description="Configurable multi-model AI agent"
LABEL org.opencontainers.image.source="https://github.com/your-org/hermes-agent"

# Non-root user for security
RUN useradd --create-home --shell /bin/bash hermes
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY --chown=hermes:hermes . .

# Create logs directory with correct permissions
RUN mkdir -p logs && chown hermes:hermes logs

USER hermes

# Environment defaults (override via docker run -e or docker-compose)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HERMES_CONFIG=/app/config/config.yaml \
    LOG_LEVEL=INFO

# Expose no ports by default (adjust if you add an API server)
# EXPOSE 8080

ENTRYPOINT ["python", "-m", "src.main"]
CMD []
