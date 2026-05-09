# Single-stage Python slim image for the FastAPI fraud detection service.
# Build:   docker build -t fraud-detection-api .
# Run:     docker run --rm -p 8000:8000 fraud-detection-api

FROM python:3.13-slim

# Faster, leaner Python at runtime:
#   - PYTHONDONTWRITEBYTECODE  no .pyc files to disk (image stays smaller)
#   - PYTHONUNBUFFERED         logs flush immediately so docker logs show output
#   - PIP_NO_CACHE_DIR         do not cache pip downloads in the layer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 is needed by xgboost's compiled OpenMP binaries at runtime.
# slim images don't include it; without this the import fails at startup.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so Docker can cache the heavy `pip install` layer
# across rebuilds when only application code changes.
COPY requirements-prod.txt ./
RUN pip install -r requirements-prod.txt

# Application code + model bundle.
COPY src/ ./src/
COPY app/ ./app/
COPY models/fraud_model.joblib ./models/fraud_model.joblib

# Run as an unprivileged user — security best practice. The image's contents
# are world-readable by default, and the service doesn't write to the filesystem.
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 8000

# Container healthcheck — uses Python (no curl in slim image) to hit /health.
# Kubernetes / docker swarm / Render use this to know when the service is ready.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).read() else 1)"

# 0.0.0.0 (not 127.0.0.1) so the host can reach the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
