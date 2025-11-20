# ============================================================
#               STAGE 1 — Build Dependencies
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies required for building Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements for efficient caching
COPY requirements.txt .

# Create a virtual environment inside /opt/venv
RUN python -m venv /opt/venv

# Activate venv & install dependencies
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ============================================================
#               STAGE 2 — Final Runtime Image
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Copy the Python virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Ensure venv is used
ENV PATH="/opt/venv/bin:$PATH"

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project code
COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY model_metadata.json model_metadata.json

# Expose FastAPI port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Non-root safety
RUN useradd -m appuser
USER appuser

# Start FastAPI via Uvicorn (best practice)
CMD ["uvicorn", "src.api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
