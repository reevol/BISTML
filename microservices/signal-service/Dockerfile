FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY microservices/signal-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/signals /app/src/signals
COPY src/features /app/src/features
COPY src/utils /app/src/utils
COPY microservices/signal-service/main.py /app/

RUN mkdir -p /app/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
