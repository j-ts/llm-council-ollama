#!/bin/bash

# LLM Council - Start All Services (Background Mode)

echo "Starting LLM Council in background mode..."
echo ""

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found."
    echo "   Please run: ./setup.sh"
    exit 1
fi

# Check if Python dependencies are installed in venv
if ! .venv/bin/python3 -c "import fastapi" &> /dev/null; then
    echo "❌ Python dependencies not installed."
    echo "   Please run: ./setup.sh"
    exit 1
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "❌ Frontend dependencies not installed."
    echo "   Please run: ./setup.sh"
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found."
    echo "   Please run: ./setup.sh"
    exit 1
fi

# Load Redis port from .env (default to 6380)
REDIS_PORT=6380
if [ -f ".env" ]; then
    REDIS_PORT=$(grep '^REDIS_PORT=' .env 2>/dev/null | cut -d'=' -f2)
    REDIS_PORT=${REDIS_PORT:-6380}
fi

# Check if Redis container is running
if ! docker ps | grep -q llm-council-redis; then
    echo "⚠️  Redis container not running. Starting..."
    docker start llm-council-redis 2>/dev/null || \
    docker run -d --name llm-council-redis -p ${REDIS_PORT}:6379 --restart unless-stopped redis:7-alpine
    sleep 2
fi

# Start backend
echo "Starting backend..."
if command -v uv &> /dev/null; then
    nohup uv run python -m backend.main > backend.log 2>&1 &
else
    nohup .venv/bin/python3 -m backend.main > backend.log 2>&1 &
fi
BACKEND_PID=$!
sleep 2

# Start worker
echo "Starting worker..."
# Fix for macOS fork() issue with Objective-C runtime
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
if command -v uv &> /dev/null; then
    nohup uv run rq worker council --url redis://localhost:${REDIS_PORT} > worker.log 2>&1 &
else
    nohup .venv/bin/python3 -m rq.cli worker council --url redis://localhost:${REDIS_PORT} > worker.log 2>&1 &
fi
WORKER_PID=$!
sleep 2

# Start frontend
echo "Starting frontend..."
cd frontend
nohup npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo ""
echo "✓ LLM Council is running in background!"
echo ""
echo "Services:"
echo "  Backend:  http://localhost:8001 (PID: $BACKEND_PID)"
echo "  Worker:   Running (PID: $WORKER_PID)"
echo "  Frontend: http://localhost:5173 (PID: $FRONTEND_PID)"
echo "  Redis:    localhost:${REDIS_PORT} (Docker container)"
echo ""
echo "Logs:"
echo "  Backend:  tail -f backend.log"
echo "  Worker:   tail -f worker.log"
echo "  Frontend: tail -f frontend/frontend.log"
echo ""
echo "To stop: ./stop.sh"
