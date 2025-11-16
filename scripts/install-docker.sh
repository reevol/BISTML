#!/bin/bash
# Install Docker and Docker Compose on Ubuntu/Debian

set -e

echo "=================================================="
echo "BIST AI Trading System - Docker Installation"
echo "=================================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

echo "[1/5] Updating package index..."
apt-get update

echo "[2/5] Installing prerequisites..."
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

echo "[3/5] Adding Docker's official GPG key..."
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo "[4/5] Setting up Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update

echo "[5/5] Installing Docker Engine..."
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo ""
echo "✅ Docker installed successfully!"
echo ""

# Add current user to docker group
if [ -n "$SUDO_USER" ]; then
    usermod -aG docker $SUDO_USER
    echo "✅ Added $SUDO_USER to docker group"
    echo "⚠️  Please log out and log back in for group changes to take effect"
fi

# Start Docker service
systemctl start docker
systemctl enable docker

echo ""
echo "Testing Docker installation..."
docker --version
docker compose version

echo ""
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Log out and log back in (for docker group)"
echo "2. Run: cd /home/user/BISTML"
echo "3. Run: ./setup-and-run.sh"
echo ""
