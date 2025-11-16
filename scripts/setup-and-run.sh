#!/bin/bash
# Setup and run BIST AI Trading System

set -e

cd /home/user/BISTML

echo "=================================================="
echo "BIST AI Trading System - Setup and Run"
echo "=================================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo "Please run: sudo ./install-docker.sh"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "[1/4] Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your API keys:"
    echo "   - FRED_API_KEY"
    echo "   - EVDS_API_KEY"
    echo "   - OPENAI_API_KEY"
    echo "   - SMTP credentials (for alerts)"
    echo "   - TELEGRAM_BOT_TOKEN (optional)"
    echo ""
    read -p "Press Enter after you've edited .env file..."
else
    echo "[1/4] .env file already exists ✓"
fi

echo ""
echo "[2/4] Building Docker images (this may take 5-10 minutes)..."
docker compose build

echo ""
echo "[3/4] Starting all services..."
docker compose up -d

echo ""
echo "[4/4] Waiting for services to be ready..."
sleep 10

# Check service health
echo ""
echo "Checking service health..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API Gateway is running"
else
    echo "⚠️  API Gateway is starting..."
fi

if curl -s http://localhost:8501 > /dev/null 2>&1; then
    echo "✅ Dashboard is running"
else
    echo "⚠️  Dashboard is starting..."
fi

echo ""
echo "=================================================="
echo "✅ BIST AI Trading System Started!"
echo "=================================================="
echo ""
echo "Services:"
echo "  • Dashboard:    http://localhost:8501"
echo "  • API Gateway:  http://localhost:8000"
echo "  • RabbitMQ UI:  http://localhost:15672 (user: bistml, pass: from .env)"
echo ""
echo "Useful commands:"
echo "  • View logs:     docker compose logs -f"
echo "  • Stop services: docker compose down"
echo "  • Restart:       docker compose restart"
echo "  • Status:        docker compose ps"
echo ""
echo "Check health:"
echo "  curl http://localhost:8000/health"
echo ""
