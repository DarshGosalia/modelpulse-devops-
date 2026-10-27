# ==============================================================================
# ModelPulse — Containerized PyCaret Flask Application (Unit 1.3 & Unit 10)
# Base Image: Official Python 3.10 slim
# ==============================================================================

FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

WORKDIR /app

# Install system dependencies (libgomp1 required for LightGBM/scikit-learn, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies (Layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

# Ensure logs directory exists
RUN mkdir -p /app/logs

# Train & package fresh model pipeline during container build (Unit 10.1 / Automated Retraining)
RUN python train.py

# Expose web service port
EXPOSE 5000

# Container Healthcheck (DevOps Best Practice)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Start Flask application using production-ready Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]