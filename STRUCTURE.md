# BIST AI Trading System

## 📁 Project Structure

```
BISTML/
├── README.md                    # Start here
├── docker-compose.yml           # Run: docker-compose up -d
├── .env.example                 # Configure: cp .env.example .env
├── .gitignore
├── requirements.txt
├── setup.py
│
├── microservices/               # Docker microservices
│   ├── api-gateway/            # API routing (port 8000)
│   ├── data-service/           # Market data (port 8001)
│   ├── news-service/           # News & sentiment (port 8002)
│   ├── ml-service/             # ML models (port 8003)
│   ├── signal-service/         # Trading signals (port 8004)
│   ├── portfolio-service/      # Portfolio management (port 8005)
│   ├── gui-service/            # Dashboard (port 8501)
│   ├── scheduler/              # Automation
│   └── database/               # PostgreSQL setup
│
├── src/                        # Python source code
│   ├── data/                   # Data collection & processing
│   ├── features/               # Feature engineering
│   ├── models/                 # ML models
│   ├── signals/                # Signal generation
│   ├── portfolio/              # Portfolio management
│   ├── backtesting/            # Backtesting engine
│   ├── utils/                  # Utilities
│   └── ui/                     # Dashboard & CLI
│
├── tests/                      # Unit tests
├── examples/                   # Usage examples
├── notebooks/                  # Jupyter notebooks
├── configs/                    # Configuration files
├── docs/                       # Documentation
│   ├── INSTALLATION.md         # Detailed setup guide
│   ├── DOCKER_README.md        # Docker guide
│   ├── MICROSERVICES_ARCHITECTURE.md  # Architecture
│   └── guides/                 # Component guides
│
└── scripts/                    # Helper scripts
    ├── install-docker.sh       # Install Docker
    ├── setup-and-run.sh        # Quick start
    └── test-services.sh        # Test all services
```

## 🚀 Quick Start

1. **Configure**:
   ```bash
   cp .env.example .env
   nano .env  # Add API keys
   ```

2. **Run**:
   ```bash
   docker-compose up -d
   ```

3. **Access**:
   - Dashboard: http://localhost:8501
   - API: http://localhost:8000/docs

## 📚 Key Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview & quick start |
| `docker-compose.yml` | Service orchestration |
| `.env.example` | Environment configuration template |
| `requirements.txt` | Python dependencies |
| `setup.py` | Package installation |

## 📖 Documentation

- **[README.md](../README.md)** - Start here
- **[Installation](docs/INSTALLATION.md)** - Detailed setup
- **[Docker Guide](docs/DOCKER_README.md)** - Docker deployment
- **[Architecture](docs/MICROSERVICES_ARCHITECTURE.md)** - System design
- **[Guides](docs/guides/)** - Component documentation

## 🔧 Development

```bash
# Install for development
pip install -e .

# Run tests
pytest tests/

# Run single service
python -m uvicorn microservices.data-service.main:app

# View logs
docker-compose logs -f [service-name]
```

## 📊 Microservices

Each microservice is independent and can be scaled separately:

| Service | Port | Purpose |
|---------|------|---------|
| api-gateway | 8000 | API routing |
| data-service | 8001 | Data collection |
| news-service | 8002 | News & NLP |
| ml-service | 8003 | ML predictions |
| signal-service | 8004 | Signal generation |
| portfolio-service | 8005 | Portfolio mgmt |
| gui-service | 8501 | Dashboard |

## 🎯 Main Components

### Data Pipeline
```
Data Collection → Processing → Feature Engineering → Storage
```

### ML Pipeline
```
Features → Model Training → Predictions → Signal Generation
```

### Trading Pipeline
```
Signals → Portfolio → Execution → Monitoring
```

## 📝 Configuration

All configuration in `.env`:
- Database credentials
- API keys (FRED, EVDS, OpenAI)
- Alert settings (Email, Telegram)
- Environment settings

## 🔄 Workflow

1. **Data Collection** (automated every 30 min)
2. **Feature Calculation** (technical, fundamental, whale)
3. **ML Predictions** (LSTM, XGBoost, LightGBM)
4. **Signal Generation** (multi-model ensemble)
5. **Portfolio Management** (position tracking, alerts)
6. **Backtesting** (historical validation)

## 🎓 Learning Path

1. Read **README.md** (overview)
2. Try **Quick Start** (get it running)
3. Explore **Dashboard** (see it in action)
4. Read **Architecture** (understand design)
5. Check **Examples** (see code usage)
6. Dive into **src/** (understand internals)

---

**Total**: 168 files, 62,000+ lines of production code
