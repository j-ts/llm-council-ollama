#!/bin/bash

# LLM Council - Setup Script

echo "=========================================="
echo "LLM Council - Setup"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

# Check Python
echo "Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or later."
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python $PYTHON_VERSION found"
echo ""

# Check Node.js
echo "Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 20.19+ or 22.12+"
    exit 1
fi
NODE_VERSION=$(node --version)
echo "✓ Node.js $NODE_VERSION found"
echo ""

# Check Docker
echo "Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker."
    exit 1
fi
echo "✓ Docker found"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Detect OS for venv activation
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    VENV_PYTHON=".venv/Scripts/python"
    VENV_PIP=".venv/Scripts/pip"
else
    VENV_PYTHON=".venv/bin/python3"
    VENV_PIP=".venv/bin/pip"
fi

# Install dependencies in venv
if command -v uv &> /dev/null; then
    echo "Using uv..."
    uv sync
else
    echo "Using pip in virtual environment..."
    $VENV_PIP install --upgrade pip
    $VENV_PIP install fastapi uvicorn httpx pydantic python-dotenv redis rq
fi
echo "✓ Python dependencies installed"
echo ""

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..
echo "✓ Frontend dependencies installed"
echo ""

# Setup environment file
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your OPENROUTER_API_KEY"
    echo "   Get your API key from: https://openrouter.ai/keys"
    echo ""
else
    echo "✓ .env file already exists"
    echo ""
fi

# Load Redis port from .env if exists (default to 6380)
REDIS_PORT=6380
if [ -f ".env" ]; then
    REDIS_PORT=$(grep '^REDIS_PORT=' .env 2>/dev/null | cut -d'=' -f2)
    REDIS_PORT=${REDIS_PORT:-6380}
fi

# Check if Redis is running or create container
echo "Setting up Redis..."
if docker ps | grep -q llm-council-redis; then
    echo "✓ Redis container already running"
elif docker ps -a | grep -q llm-council-redis; then
    echo "Starting existing Redis container..."
    docker start llm-council-redis
    echo "✓ Redis container started"
else
    echo "Creating Redis container..."
    docker run -d --name llm-council-redis -p ${REDIS_PORT}:6379 --restart unless-stopped redis:7-alpine
    echo "✓ Redis container created and started"
fi
echo ""

echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OPENROUTER_API_KEY if not done yet"
echo "2. Run: ./start-background.sh"
echo "3. Open: http://localhost:5173"
echo ""
