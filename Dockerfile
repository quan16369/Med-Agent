FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY medassist/ ./medassist/
COPY *.py ./
COPY .env.example .env

# Create non-root user and logs directory
RUN useradd -m -u 1000 medassist && \
    mkdir -p /app/logs && \
    chown -R medassist:medassist /app

USER medassist

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/liveness || exit 1

# Expose port
EXPOSE 8000

# Run API server
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
