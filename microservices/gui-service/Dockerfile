FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY microservices/gui-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ui/dashboard.py /app/dashboard.py
COPY microservices/gui-service/.streamlit /app/.streamlit

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
