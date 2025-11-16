# BIST AI Trading System - Installation Guide

## 🚀 Quick Installation

### Option 1: Automated Installation (Recommended)

```bash
# 1. Install Docker
sudo ./install-docker.sh

# 2. Log out and log back in (for docker group to take effect)
exit
# Then log back in

# 3. Setup and run the system
cd /home/user/BISTML
./setup-and-run.sh

# 4. Test services
./test-services.sh
```

### Option 2: Manual Installation

#### Step 1: Install Docker

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

**Log out and log back in** for group changes to take effect.

#### Step 2: Configure Environment

```bash
cd /home/user/BISTML

# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

**Required API Keys:**
- `FRED_API_KEY` - Get from https://fred.stlouisfed.org/docs/api/api_key.html (free)
- `EVDS_API_KEY` - Get from https://evds2.tcmb.gov.tr/ (free, Turkish Central Bank)
- `OPENAI_API_KEY` - Get from https://platform.openai.com/api-keys (paid, for LLM features)

**Optional (for alerts):**
- `SMTP_*` - Email configuration
- `TELEGRAM_BOT_TOKEN` - Telegram bot token
- `TELEGRAM_CHAT_ID` - Your Telegram chat ID

#### Step 3: Build and Start Services

```bash
# Build all Docker images (takes 5-10 minutes first time)
docker compose build

# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

#### Step 4: Verify Installation

```bash
# Check health
curl http://localhost:8000/health

# Or use the test script
./test-services.sh
```

## 🌐 Access Services

Once running, access:

- **Dashboard**: http://localhost:8501
- **API Gateway**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **RabbitMQ Management**: http://localhost:15672 (user: bistml, pass: from .env)

## 📊 Service Ports

| Service | Port | Purpose |
|---------|------|---------|
| API Gateway | 8000 | Main API entry point |
| Data Service | 8001 | Market data collection |
| News Service | 8002 | News & sentiment |
| ML Service | 8003 | Model predictions |
| Signal Service | 8004 | Trading signals |
| Portfolio Service | 8005 | Portfolio management |
| Dashboard | 8501 | Web UI |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache |
| RabbitMQ | 5672 | Message queue |
| RabbitMQ UI | 15672 | Queue management |

## 🔧 Common Operations

### Start/Stop Services

```bash
# Start all
docker compose up -d

# Stop all
docker compose down

# Restart specific service
docker compose restart signal-service

# Stop and remove volumes (⚠️ deletes all data!)
docker compose down -v
```

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f news-service

# Last 100 lines
docker compose logs --tail=100 ml-service

# With timestamps
docker compose logs -f --timestamps
```

### Manage Data

```bash
# Backup database
docker compose exec postgres pg_dump -U bistml_user bistml > backup.sql

# Restore database
docker compose exec -T postgres psql -U bistml_user bistml < backup.sql

# Access PostgreSQL
docker compose exec postgres psql -U bistml_user -d bistml

# Access Redis
docker compose exec redis redis-cli
```

### Scale Services

```bash
# Run multiple ML service instances
docker compose up -d --scale ml-service=3

# Run multiple signal service instances
docker compose up -d --scale signal-service=2
```

## 🧪 Testing the System

### 1. Test Data Collection

```bash
curl -X POST http://localhost:8000/api/data/v1/ohlcv \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["THYAO", "GARAN"],
    "timeframe": "1d",
    "period": "1mo"
  }'
```

### 2. Test News Collection

```bash
curl "http://localhost:8000/api/news/v1/stock/THYAO?days_back=3"
```

### 3. Test Signal Generation

```bash
curl -X POST http://localhost:8000/api/signals/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["THYAO", "GARAN", "AKBNK"],
    "strategy": "BALANCED"
  }'
```

### 4. Test ML Predictions

```bash
# First, check available models
curl http://localhost:8000/api/ml/v1/models

# Make a prediction (requires trained model)
curl -X POST http://localhost:8000/api/ml/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "xgboost",
    "features": [[1.0, 2.0, 3.0]]
  }'
```

## 🐛 Troubleshooting

### Services won't start

```bash
# Check Docker is running
sudo systemctl status docker

# Check logs
docker compose logs

# Rebuild from scratch
docker compose down -v
docker compose build --no-cache
docker compose up -d
```

### Port already in use

```bash
# Find what's using the port
sudo lsof -i :8000

# Or change port in docker-compose.yml
ports:
  - "8080:8000"  # Change 8000 to 8080
```

### Out of memory

```bash
# Check Docker resources
docker stats

# Increase Docker memory in Docker Desktop settings
# Or add resource limits in docker-compose.yml
```

### Database connection errors

```bash
# Check PostgreSQL is running
docker compose ps postgres

# Check logs
docker compose logs postgres

# Verify credentials in .env match docker-compose.yml
```

### Services can't reach each other

```bash
# Check network
docker network inspect bistml_bistml-network

# Restart networking
docker compose down
docker compose up -d
```

## 📋 System Requirements

### Minimum
- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 20GB
- **OS**: Ubuntu 20.04+ or similar Linux

### Recommended
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Disk**: 50GB+ SSD
- **GPU**: NVIDIA GPU (for ML training)
- **OS**: Ubuntu 22.04 LTS

## 🔐 Security Checklist

- [ ] Changed default passwords in `.env`
- [ ] `.env` file is in `.gitignore`
- [ ] Database password is strong
- [ ] RabbitMQ credentials are changed
- [ ] API keys are kept secret
- [ ] Services are behind firewall (if production)
- [ ] SSL/TLS enabled for external access

## 🚀 Production Deployment

For production, consider:

1. **Use environment-specific .env files**
   ```bash
   cp .env.example .env.production
   ```

2. **Enable SSL/TLS with nginx reverse proxy**

3. **Set up monitoring** (Prometheus, Grafana)

4. **Configure backups** (automated database backups)

5. **Use Docker Swarm or Kubernetes** for orchestration

6. **Set resource limits** in docker-compose.yml

7. **Enable log rotation**

## 📞 Support

If you encounter issues:

1. Check logs: `docker compose logs -f`
2. Run tests: `./test-services.sh`
3. Check health: `curl http://localhost:8000/health`
4. Review DOCKER_README.md
5. Check GitHub issues

## 🎯 Next Steps

After installation:

1. ✅ Configure API keys in `.env`
2. ✅ Start services with `./setup-and-run.sh`
3. ✅ Access dashboard at http://localhost:8501
4. ✅ Test data collection
5. ✅ Train ML models
6. ✅ Generate first signals
7. ✅ Set up portfolio

**Enjoy your AI trading system!** 🚀
