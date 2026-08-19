# ──────────────────────────────────────────────────────────
# Dockerfile — ML Model Serving (FastAPI + scikit-learn)
# ──────────────────────────────────────────────────────────
#
# Packages the trained model + FastAPI app into a portable
# container that can run anywhere Docker runs.
#
# Build:
#   docker build -t ml-model-api .
#
# Run:
#   docker run -p 8321:8321 ml-model-api
#
# Test:
#   curl http://localhost:8321/health
#   curl -X POST http://localhost:8321/predict \
#     -H "Content-Type: application/json" \
#     -d '{"features": [[0.5, -1.2, 0.3, 0.8]]}'
# ──────────────────────────────────────────────────────────

# --- Stage 1: lightweight Python base ---
FROM python:3.11-slim AS base

# prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- Stage 2: install dependencies ---
# copy only requirements first to leverage Docker layer caching
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# --- Stage 3: copy application code + model artifact ---
COPY app.py .
COPY model.joblib .

# non-root user for security (production best practice)
RUN adduser --disabled-password --no-create-home appuser
USER appuser

# expose the port the app listens on
EXPOSE 8321

# health check — container orchestrators (K8s, ECS) use this
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8321/health')" || exit 1

# start the server
#   --host 0.0.0.0  : listen on all interfaces (required inside a container)
#   --port 8321     : match the EXPOSE above
#   --workers 2     : 2 uvicorn workers (adjust for your CPU count)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8321", "--workers", "2"]
