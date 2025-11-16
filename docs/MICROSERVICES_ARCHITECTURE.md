# BIST AI Trading System - Microservices Architecture

## 🎯 Overview

The BIST AI Trading System has been implemented as a **Docker-based microservices architecture**, allowing independent scaling, deployment, and maintenance of each component.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL ACCESS                                │
├─────────────────────────────────────────────────────────────────────────┤
│  Port 8501: Streamlit Dashboard  │  Port 8000: API Gateway              │
└─────────────┬─────────────────────┴──────────────────┬──────────────────┘
              │                                         │
              │            ┌────────────────────────────▼───────┐
              │            │      API Gateway Service           │
              │            │  - Request routing                 │
              │            │  - Load balancing                  │
              │            │  - Health monitoring               │
              │            └──┬───┬───┬───┬───┬────────────────┘
              │               │   │   │   │   │
              │    ┌──────────┘   │   │   │   └──────────┐
              │    │              │   │   │              │
┌─────────────▼────▼──┐  ┌────────▼───▼──┐  ┌──────────▼──────▼──────┐
│   GUI Service       │  │ Data Service   │  │  News Service           │
│   (Streamlit)       │  │ - BIST OHLCV   │  │  - News collection      │
│  - Live signals     │  │ - Fundamentals │  │  - Sentiment analysis   │
│  - Portfolio view   │  │ - Macro data   │  │  - Entity extraction    │
│  - Backtests        │  │ - Whale data   │  │  - LLM integration      │
│  - Performance      │  │ - Validation   │  │  - Turkish NLP          │
└─────────────────────┘  └────────────────┘  └─────────────────────────┘
                                  │                      │
                         ┌────────▼──────────────────────▼────────┐
                         │         ML Service                      │
                         │  - LSTM/GRU training                    │
                         │  - XGBoost/LightGBM                     │
                         │  - Random Forest/ANN                    │
                         │  - Model serving                        │
                         │  - Predictions                          │
                         └──────────┬──────────────────────────────┘
                                    │
                         ┌──────────▼──────────────────────────────┐
                         │      Signal Service                     │
                         │  - Multi-model aggregation              │
                         │  - Confidence scoring                   │
                         │  - Signal prioritization                │
                         │  - Risk assessment                      │
                         └──────────┬──────────────────────────────┘
                                    │
                         ┌──────────▼──────────────────────────────┐
                         │    Portfolio Service                    │
                         │  - Position management                  │
                         │  - P&L tracking                         │
                         │  - Optimization (Kelly, Risk Parity)    │
                         │  - Alerts (Email, Telegram, SMS)        │
                         └─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE SERVICES                           │
├─────────────────────────────────────────────────────────────────────────┤
│  PostgreSQL (5432)  │  Redis (6379)  │  RabbitMQ (5672, 15672)         │
│  - Time series data │  - Caching     │  - Message queue                 │
│  - Signals          │  - Sessions    │  - Async processing              │
│  - Portfolios       │  - Temp data   │  - Event streaming               │
│  - News articles    │                │                                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          SCHEDULER SERVICE                               │
│  - 30-minute signal generation                                          │
│  - Hourly signal generation                                             │
│  - BIST trading hours detection                                         │
│  - Automated data collection                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📦 Microservices Breakdown

### 1. **Data Service** 🗄️
**Port**: 8001
**Purpose**: Collects and processes all market data

**Responsibilities**:
- BIST OHLCV data collection (15m, 30m, 1h, daily)
- Fundamental data (financial statements, ratios)
- Macro indicators (Turkish CPI, interest rates, global indices)
- Whale/brokerage distribution data
- Data cleaning and validation

**Tech Stack**:
- FastAPI for REST API
- yfinance for BIST data
- FRED API for macro data
- PostgreSQL for storage
- Redis for caching

**Endpoints**:
- `POST /api/v1/data/ohlcv` - Collect OHLCV data
- `POST /api/v1/data/fundamental` - Collect fundamentals
- `POST /api/v1/data/macro` - Collect macro indicators
- `GET /api/v1/data/symbols` - List available symbols

---

### 2. **News Service** 📰
**Port**: 8002
**Purpose**: Turkish financial news collection and sentiment analysis

**Responsibilities**:
- Collect news from Turkish sources
- Scrape KAP (Public Disclosure Platform)
- Turkish sentiment analysis (BERTurk)
- Entity extraction (stocks, companies, people)
- LLM-based impact analysis
- News aggregation by stock

**Tech Stack**:
- FastAPI
- BeautifulSoup/Scrapy for scraping
- Transformers (BERTurk)
- OpenAI API for LLM
- spaCy for NER

**Endpoints**:
- `POST /api/v1/news/collect` - Collect news
- `GET /api/v1/news/stock/{code}` - Get stock-specific news
- `POST /api/v1/news/sentiment` - Analyze sentiment
- `POST /api/v1/news/impact` - LLM impact analysis

---

### 3. **ML Service** 🤖
**Port**: 8003
**Purpose**: Machine learning model training and serving

**Responsibilities**:
- Train LSTM/GRU models
- Train XGBoost/LightGBM models
- Train Random Forest/ANN classifiers
- Serve predictions via REST API
- Model versioning and management
- Feature engineering

**Tech Stack**:
- FastAPI
- TensorFlow/Keras (LSTM, ANN)
- PyTorch (GRU)
- XGBoost, LightGBM
- scikit-learn

**Endpoints**:
- `POST /api/v1/predict` - Make predictions
- `POST /api/v1/train` - Train model
- `GET /api/v1/models` - List models
- `POST /api/v1/models/load/{name}` - Load model

---

### 4. **Signal Service** 📊
**Port**: 8004
**Purpose**: Trading signal generation and prioritization

**Responsibilities**:
- Aggregate multi-model predictions
- Calculate confidence scores
- Prioritize signals (WAI, sentiment, agreement)
- Risk-adjusted scoring
- Generate buy/sell/hold signals

**Tech Stack**:
- FastAPI
- Calls Data, ML, and News services
- Custom signal aggregation logic

**Endpoints**:
- `POST /api/v1/signals/generate` - Generate signals
- `POST /api/v1/signals/prioritize` - Prioritize signals
- `GET /api/v1/signals/latest` - Get latest signals

---

### 5. **Portfolio Service** 💼
**Port**: 8005
**Purpose**: Portfolio management and alerts

**Responsibilities**:
- Track positions and transactions
- Calculate P&L (realized/unrealized)
- Portfolio optimization (Kelly Criterion, Risk Parity)
- Generate alerts for holdings
- Multi-channel notifications (Email, Telegram, SMS)

**Tech Stack**:
- FastAPI
- PostgreSQL for positions
- SMTP for email
- Telegram Bot API
- Twilio for SMS

**Endpoints**:
- `GET /api/v1/portfolio` - Get portfolio summary
- `POST /api/v1/portfolio/buy` - Execute buy
- `POST /api/v1/portfolio/sell` - Execute sell
- `POST /api/v1/portfolio/optimize` - Optimize allocation

---

### 6. **API Gateway** 🚪
**Port**: 8000
**Purpose**: Central entry point and request routing

**Responsibilities**:
- Route requests to services
- Load balancing
- Health monitoring
- Request/response logging
- Rate limiting (optional)

**Tech Stack**:
- FastAPI
- httpx for async requests

**Endpoints**:
- `GET /health` - Health check all services
- `/api/data/*` → Data Service
- `/api/news/*` → News Service
- `/api/ml/*` → ML Service
- `/api/signals/*` → Signal Service
- `/api/portfolio/*` → Portfolio Service

---

### 7. **GUI Service** 🖥️
**Port**: 8501
**Purpose**: Web-based dashboard

**Responsibilities**:
- Real-time signal display
- Portfolio visualization
- Backtesting results
- Performance charts
- Interactive filtering

**Tech Stack**:
- Streamlit
- Plotly for charts
- Calls API Gateway

**Features**:
- Live signals tab
- Portfolio tab
- Backtesting tab
- Performance tab

---

### 8. **Scheduler** ⏰
**Purpose**: Automated task execution

**Responsibilities**:
- Generate signals every 30 minutes
- Generate signals every hour
- BIST trading hours detection
- Turkish holiday handling
- Automated data collection

**Tech Stack**:
- APScheduler
- Calls Signal and Data services

---

## 🗄️ Infrastructure Services

### PostgreSQL
- Stores time series data (OHLCV)
- Stores signals and predictions
- Stores portfolios and transactions
- Stores news articles

### Redis
- Caches frequently accessed data
- Session management
- Temporary data storage
- Pub/sub for real-time updates

### RabbitMQ
- Message queue for async tasks
- Event streaming
- Service decoupling
- Background job processing

---

## 🚀 Deployment

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/BISTML.git
cd BISTML

# Configure environment
cp .env.example .env
nano .env  # Add API keys

# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/health

# Access dashboard
open http://localhost:8501
```

### Individual Service
```bash
# Start only data service
docker-compose up -d data-service postgres redis

# View logs
docker-compose logs -f data-service

# Restart service
docker-compose restart news-service
```

---

## 🔌 Service Communication

### REST API
- Synchronous request/response
- Used for: predictions, data queries, signal generation
- Protocol: HTTP/JSON

### Message Queue (RabbitMQ)
- Asynchronous processing
- Used for: background tasks, notifications, batch jobs
- Protocol: AMQP

### Redis Pub/Sub
- Real-time updates
- Used for: signal broadcasts, price updates
- Protocol: Redis protocol

---

## 📊 Data Flow Example

**Generating Trading Signals**:

1. **Scheduler** triggers signal generation (every 30 min)
2. **Signal Service** requests data:
   - Calls **Data Service** for OHLCV, fundamentals, whale data
   - Calls **News Service** for recent news and sentiment
3. **ML Service** receives features and returns predictions
4. **Signal Service** aggregates all inputs:
   - ML predictions (LSTM, XGBoost, LightGBM)
   - Sentiment scores
   - Whale Activity Index
   - Technical indicators
5. **Signal Service** generates final signal with confidence
6. **Portfolio Service** checks if signal affects user holdings
7. **Portfolio Service** sends alerts if needed
8. Results stored in **PostgreSQL**
9. **GUI Service** displays signals in dashboard

---

## 🔐 Security

- Services communicate on internal Docker network
- Only API Gateway (8000) and GUI (8501) exposed
- Database credentials in `.env` (not committed)
- SSL/TLS for external APIs
- Input validation on all endpoints
- Rate limiting on API Gateway

---

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale ML service for more prediction capacity
docker-compose up -d --scale ml-service=3

# Scale signal service for more throughput
docker-compose up -d --scale signal-service=2
```

### Vertical Scaling
Edit `docker-compose.yml` resource limits:
```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
```

---

## 🧪 Testing

```bash
# Test data collection
curl -X POST http://localhost:8000/api/data/v1/ohlcv \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["THYAO"], "timeframe": "1d", "period": "1mo"}'

# Test news collection
curl http://localhost:8000/api/news/v1/stock/THYAO

# Test signal generation
curl -X POST http://localhost:8000/api/signals/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["THYAO", "GARAN"]}'
```

---

## 📝 Monitoring

```bash
# Check all services
docker-compose ps

# View resource usage
docker stats

# Check health
curl http://localhost:8000/health

# View service logs
docker-compose logs -f --tail=100 signal-service
```

---

## 🔄 CI/CD

**GitHub Actions** (example workflow):
```yaml
name: Build and Deploy

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build images
        run: docker-compose build
      - name: Run tests
        run: docker-compose run --rm ml-service pytest
      - name: Push to registry
        run: docker-compose push
```

---

## 🎯 Benefits of Microservices Architecture

✅ **Independent Scaling** - Scale only the services that need it
✅ **Technology Freedom** - Each service can use different tech
✅ **Fault Isolation** - Failure in one service doesn't crash entire system
✅ **Easy Deployment** - Deploy services independently
✅ **Team Autonomy** - Different teams can own different services
✅ **Faster Development** - Parallel development of services
✅ **Better Maintainability** - Smaller, focused codebases

---

**Complete microservices implementation ready for production deployment!** 🚀
