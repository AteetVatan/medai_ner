# =============================================================================
# Stage 1: Build Dependencies
# =============================================================================
FROM python:3.12-slim as builder

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency files
COPY pyproject.toml ./        
COPY requirements-prod.txt ./

# Install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-prod.txt

# =============================================================================
# Stage 2: Production Runtime
# =============================================================================
FROM python:3.12-slim as production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/app" \
    PATH="/opt/venv/bin:$PATH"

# Install runtime libs (Debian Trixie compatible)
RUN apt-get update && apt-get install -y \
    libxml2 \
    libxslt1.1 \
    libffi8 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Non-root user
RUN groupadd -r medai && useradd -r -g medai -d /app -s /bin/bash medai

# Copy venv
COPY --from=builder /opt/venv /opt/venv

# Set workdir and copy app
WORKDIR /app
COPY --chown=medai:medai . .

# Create app directories
RUN mkdir -p /app/logs /app/cache /app/models && chown -R medai:medai /app

USER medai
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=10)" || exit 1

CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
