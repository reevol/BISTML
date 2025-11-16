FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY microservices/ml-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY src/models /app/src/models
COPY src/features /app/src/features
COPY src/utils /app/src/utils
COPY microservices/ml-service/main.py /app/

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
