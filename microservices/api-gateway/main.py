"""
API Gateway for BIST AI Trading System
Routes requests to appropriate microservices
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
import os
from datetime import datetime

app = FastAPI(
    title="BIST AI Trading System - API Gateway",
    description="Central API gateway for all microservices",
    version="1.0.0"
)

# Service URLs
SERVICES = {
    "data": os.getenv("DATA_SERVICE_URL", "http://data-service:8000"),
    "news": os.getenv("NEWS_SERVICE_URL", "http://news-service:8000"),
    "ml": os.getenv("ML_SERVICE_URL", "http://ml-service:8000"),
    "signal": os.getenv("SIGNAL_SERVICE_URL", "http://signal-service:8000"),
    "portfolio": os.getenv("PORTFOLIO_SERVICE_URL", "http://portfolio-service:8000"),
}

@app.get("/")
async def root():
    return {
        "service": "BIST AI Trading System API Gateway",
        "version": "1.0.0",
        "services": list(SERVICES.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Check health of all services"""
    health_status = {}

    async with httpx.AsyncClient() as client:
        for service_name, service_url in SERVICES.items():
            try:
                response = await client.get(f"{service_url}/health", timeout=5.0)
                health_status[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "url": service_url
                }
            except Exception as e:
                health_status[service_name] = {
                    "status": "unreachable",
                    "error": str(e)
                }

    all_healthy = all(s["status"] == "healthy" for s in health_status.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": health_status,
        "timestamp": datetime.now().isoformat()
    }

# Proxy routes
@app.api_route("/api/data/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_data(path: str, request: Request):
    """Proxy to data service"""
    return await proxy_request("data", path, request)

@app.api_route("/api/news/{path:path}", methods=["GET", "POST"])
async def proxy_news(path: str, request: Request):
    """Proxy to news service"""
    return await proxy_request("news", path, request)

@app.api_route("/api/ml/{path:path}", methods=["GET", "POST"])
async def proxy_ml(path: str, request: Request):
    """Proxy to ML service"""
    return await proxy_request("ml", path, request)

@app.api_route("/api/signals/{path:path}", methods=["GET", "POST"])
async def proxy_signals(path: str, request: Request):
    """Proxy to signal service"""
    return await proxy_request("signal", path, request)

@app.api_route("/api/portfolio/{path:path}", methods=["GET", "POST", "PUT"])
async def proxy_portfolio(path: str, request: Request):
    """Proxy to portfolio service"""
    return await proxy_request("portfolio", path, request)

async def proxy_request(service: str, path: str, request: Request):
    """Generic proxy function"""
    service_url = SERVICES.get(service)
    if not service_url:
        raise HTTPException(status_code=404, detail=f"Service {service} not found")

    url = f"{service_url}/api/v1/{path}"

    async with httpx.AsyncClient() as client:
        try:
            # Get request body if present
            body = await request.body()

            # Forward request
            response = await client.request(
                method=request.method,
                url=url,
                content=body,
                headers=dict(request.headers),
                params=request.query_params,
                timeout=30.0
            )

            return JSONResponse(
                content=response.json() if response.text else {},
                status_code=response.status_code
            )

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Service timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
