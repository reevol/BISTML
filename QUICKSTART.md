# BIST AI Trading System - Quick Start

## 1️⃣ Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum

## 2️⃣ Install (First Time Only)

```bash
# Clone
git clone https://github.com/reevol/BISTML.git
cd BISTML

# Install Docker (if needed)
sudo scripts/install-docker.sh
```

## 3️⃣ Configure

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

**Required Keys**:
- `FRED_API_KEY` - Free from https://fred.stlouisfed.org
- `EVDS_API_KEY` - Free from https://evds2.tcmb.gov.tr

**Optional Keys**:
- `OPENAI_API_KEY` - For LLM news analysis
- Email/Telegram settings - For alerts

## 4️⃣ Run

```bash
docker-compose up -d
```

That's it! The system is now running.

## 5️⃣ Access

- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **RabbitMQ UI**: http://localhost:15672 (user: bistml, pass: from .env)

## 6️⃣ Test

```bash
# Check health
curl http://localhost:8000/health

# Or use test script
scripts/test-services.sh
```

## 7️⃣ Use

### Generate Signals

```bash
curl -X POST http://localhost:8000/api/signals/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["THYAO", "GARAN"]}'
```

### View Dashboard

Open http://localhost:8501 and explore:
- Live signals
- Portfolio tracking
- Backtesting results
- Performance charts

## 🛠️ Management

```bash
# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Stop everything
docker-compose down

# Rebuild
docker-compose build --no-cache
docker-compose up -d
```

## 🔍 Troubleshooting

**Services won't start?**
```bash
docker-compose logs
```

**Port already in use?**
Edit `docker-compose.yml` and change port mappings.

**Out of memory?**
Increase Docker memory in Docker Desktop settings.

## 📚 Next Steps

- Read [STRUCTURE.md](STRUCTURE.md) - Understand the architecture
- Check [docs/](docs/) - Detailed documentation
- Explore [examples/](examples/) - Code examples
- Run tests: `pytest tests/`

## ⚡ One-Line Commands

```bash
# Everything in one command
cp .env.example .env && nano .env && docker-compose up -d

# Check status
docker-compose ps && curl http://localhost:8000/health

# Watch logs
docker-compose logs -f signal-service
```

---

**Time to first signal**: < 5 minutes

**Questions?** Check [README.md](README.md) or [docs/INSTALLATION.md](docs/INSTALLATION.md)
