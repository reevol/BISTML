#!/bin/bash
# Test all BIST AI Trading System services

echo "=================================================="
echo "Testing BIST AI Trading System Services"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_service() {
    local name=$1
    local url=$2

    echo -n "Testing $name... "

    if curl -s -f "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        return 1
    fi
}

# Test infrastructure
echo "Infrastructure Services:"
test_service "PostgreSQL     " "http://localhost:5432" || echo "  (normal if not HTTP)"
test_service "Redis          " "http://localhost:6379" || echo "  (normal if not HTTP)"
test_service "RabbitMQ       " "http://localhost:15672"

echo ""
echo "Microservices:"
test_service "API Gateway    " "http://localhost:8000/health"
test_service "Data Service   " "http://localhost:8001/health"
test_service "News Service   " "http://localhost:8002/health"
test_service "ML Service     " "http://localhost:8003/health"
test_service "Signal Service " "http://localhost:8004/health"
test_service "Portfolio Svc  " "http://localhost:8005/health"
test_service "GUI Dashboard  " "http://localhost:8501"

echo ""
echo "Detailed Health Check:"
curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "API Gateway not responding"

echo ""
echo "=================================================="
echo "Docker Container Status:"
echo "=================================================="
docker compose ps

echo ""
echo "To view logs: docker compose logs -f [service-name]"
echo "Example: docker compose logs -f signal-service"
