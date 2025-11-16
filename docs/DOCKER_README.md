# BIST AI Trading System - Docker Microservices

## 🏗️ Architecture

The system is built as a microservices architecture using Docker containers:

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway (Port 8000)                   │
│                    Routes requests to services                   │
└────────────┬────────────┬────────────┬───────────┬──────────────┘
             │            │            │           │
    ┌────────▼───┐ ┌─────▼─────┐ ┌───▼────┐ ┌───▼─────┐
    │Data Service│ │News Service│ │ML Service│Signal Svc│
    │  (8001)    │ │  (8002)    │ │ (8003) │ │ (8004)  │
    └────────────┘ └────────────┘ └────────┘ └─────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Portfolio Service  │
                    │     (8005)         │
                    └────────────────────┘
                              │
              ┌───────────────▼───────────────┐
              │   GUI Service (Streamlit)     │
              │        Port 8501              │
              └───────────────────────────────┘

Infrastructure:
┌──────────┐  ┌──────┐  ┌──────────┐
│PostgreSQL│  │Redis │  │ RabbitMQ │
│  (5432)  │  │(6379)│  │  (5672)  │
└──────────┘  └──────┘  └──────────┘
```

## 📦 Services

### 1. **Data Service** (Port 8001)
- Collects OHLCV data from BIST
- Fetches fundamental data
- Gathers macro indicators
- Collects whale/brokerage data
- **API**: `/api/v1/data/*`

### 2. **News Service** (Port 8002)
- Collects Turkish financial news
- Performs sentiment analysis (BERTurk)
- Entity extraction
- LLM-based impact analysis
- **API**: `/api/v1/news/*`

### 3. **ML Service** (Port 8003)
- Trains ML models (LSTM, XGBoost, LightGBM, RF)
- Serves predictions
- Model management
- Feature engineering
- **API**: `/api/v1/ml/*`

### 4. **Signal Service** (Port 8004)
- Generates trading signals
- Aggregates multi-model predictions
- Signal prioritization
- Confidence scoring
- **API**: `/api/v1/signals/*`

### 5. **Portfolio Service** (Port 8005)
- Manages portfolios
- Tracks positions and P&L
- Portfolio optimization
- Alerts (Email, Telegram, SMS)
- **API**: `/api/v1/portfolio/*`

### 6. **API Gateway** (Port 8000)
- Central entry point
- Routes to services
- Load balancing
- Health checks

### 7. **GUI Service** (Port 8501)
- Streamlit dashboard
- Real-time signal display
- Portfolio visualization
- Backtesting results

### 8. **Scheduler**
- Automated signal generation (30-min/1-hr)
- Data collection scheduling
- BIST trading hours detection

## 🚀 Quick Start

### Prerequisites
- Docker >= 20.10
- Docker Compose >= 2.0
- 8GB RAM minimum
- 20GB disk space

### 1. Clone and Configure

```bash
git clone https://github.com/yourusername/BISTML.git
cd BISTML

# Copy environment file
cp .env.example .env

# Edit .env with your API keys
nano .env
```

### 2. Build and Start All Services

```bash
# Build all services
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
curl http://localhost:8000/health
```

### 3. Access Services

- **API Gateway**: http://localhost:8000
- **Data Service**: http://localhost:8001
- **News Service**: http://localhost:8002
- **ML Service**: http://localhost:8003
- **Signal Service**: http://localhost:8004
- **Portfolio Service**: http://localhost:8005
- **Dashboard (GUI)**: http://localhost:8501
- **RabbitMQ Management**: http://localhost:15672 (user: bistml, password: from .env)

## 🔧 Individual Service Management

```bash
# Start specific service
docker-compose up -d data-service

# Stop specific service
docker-compose stop news-service

# Restart service
docker-compose restart ml-service

# View service logs
docker-compose logs -f signal-service

# Rebuild single service
docker-compose build --no-cache gui-service

# Scale service (if needed)
docker-compose up -d --scale ml-service=3
```

## 📊 Database Access

```bash
# Access PostgreSQL
docker-compose exec postgres psql -U bistml_user -d bistml

# Access Redis CLI
docker-compose exec redis redis-cli

# Backup database
docker-compose exec postgres pg_dump -U bistml_user bistml > backup.sql

# Restore database
docker-compose exec -T postgres psql -U bistml_user bistml < backup.sql
```

## 🧪 Testing Services

### Test Data Service
```bash
curl -X POST http://localhost:8000/api/data/v1/ohlcv \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["THYAO", "GARAN"], "timeframe": "1d", "period": "1mo"}'
```

### Test News Service
```bash
curl http://localhost:8000/api/news/v1/stock/THYAO
```

### Test ML Service
```bash
curl -X POST http://localhost:8000/api/ml/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "xgboost", "features": [[1.0, 2.0, 3.0]]}'
```

### Test Signal Generation
```bash
curl -X POST http://localhost:8000/api/signals/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["THYAO", "GARAN"]}'
```

## 🐛 Debugging

```bash
# Check service status
docker-compose ps

# Check resource usage
docker stats

# Access service shell
docker-compose exec data-service /bin/bash

# View service logs with timestamps
docker-compose logs -f --timestamps signal-service

# Inspect network
docker network inspect bistml_bistml-network
```

## 📈 Monitoring

```bash
# Check all service health
curl http://localhost:8000/health

# Check individual service
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

## 🔄 Updates and Maintenance

```bash
# Pull latest code
git pull origin main

# Rebuild and restart services
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Clean up old images
docker system prune -a

# Remove volumes (⚠️ deletes data!)
docker-compose down -v
```

## 🌐 Production Deployment

### Using Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml bistml

# Check services
docker service ls

# Scale services
docker service scale bistml_ml-service=3
```

### Using Kubernetes
```bash
# Convert to Kubernetes manifests
kompose convert -f docker-compose.yml

# Deploy
kubectl apply -f .
```

## 📝 Environment Variables

Key environment variables in `.env`:

```bash
# Database
DB_PASSWORD=your_secure_password

# APIs
FRED_API_KEY=your_fred_key
EVDS_API_KEY=your_evds_key
OPENAI_API_KEY=your_openai_key

# Message Queue
RABBITMQ_USER=bistml
RABBITMQ_PASSWORD=your_rabbitmq_password

# Notifications
SMTP_HOST=smtp.gmail.com
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 🛡️ Security

- All services communicate on internal Docker network
- Only API Gateway and GUI are exposed externally
- Use strong passwords in `.env`
- Don't commit `.env` to version control
- Consider using Docker secrets for production

## 🆘 Troubleshooting

**Service won't start**:
```bash
docker-compose logs service-name
```

**Database connection errors**:
- Check if PostgreSQL is healthy: `docker-compose ps postgres`
- Verify DATABASE_URL in service environment

**Out of memory**:
```bash
# Increase Docker memory limit in Docker Desktop settings
# Or add resource limits in docker-compose.yml
```

**Port already in use**:
```bash
# Change port mapping in docker-compose.yml
ports:
  - "8002:8000"  # Change 8002 to available port
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [RabbitMQ Documentation](https://www.rabbitmq.com/documentation.html)

---

**Need help?** Open an issue on GitHub or check the logs with `docker-compose logs -f`
