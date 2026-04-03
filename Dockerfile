# Minimal single-stage Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y pip setuptools wheel

# Copy app code (respects .dockerignore)
COPY src/ src/
COPY models/ models/
# COPY model_metadata.json .

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose both ports
EXPOSE 8000 8501

# Default command (overridden by docker-compose)
# CMD ["uvicorn", "src.api.serve:app", "--host", "0.0.0.0", "--port", "8000"]