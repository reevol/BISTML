FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY microservices/news-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model for Turkish
RUN python -m spacy download tr_core_news_lg || true

# Copy service code
COPY src/data/collectors/news_collector.py /app/src/data/collectors/
COPY src/models/nlp /app/src/models/nlp
COPY src/utils /app/src/utils
COPY microservices/news-service/main.py /app/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/src/data/collectors /app/src/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
