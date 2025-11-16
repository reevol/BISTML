# BIST AI Trading System - Scripts

Quick access scripts for managing the system.

## Installation & Setup

### `install-docker.sh`
Installs Docker and Docker Compose on Ubuntu/Debian.

```bash
sudo ./install-docker.sh
```

### `setup-and-run.sh`
One-command setup and launch of the entire system.

```bash
./setup-and-run.sh
```

This will:
1. Create .env file if needed
2. Build all Docker images
3. Start all services
4. Check health

## Testing

### `test-services.sh`
Tests all microservices and checks health.

```bash
./test-services.sh
```

## Development

### `run-local.sh`
Run services locally without Docker (for development).

```bash
./run-local.sh
```

## Quick Reference

```bash
# Production deployment
./setup-and-run.sh

# Check status
./test-services.sh

# View logs
docker-compose logs -f

# Stop everything
docker-compose down
```
