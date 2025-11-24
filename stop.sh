#!/bin/bash

# LLM Council - Stop All Services

echo "Stopping LLM Council services..."

# Stop backend (matches both uv and venv processes)
pkill -f "backend.main" && echo "✓ Backend stopped" || echo "  Backend was not running"

# Stop worker (matches both uv and venv processes)
pkill -f "worker.*council" && echo "✓ Worker stopped" || echo "  Worker was not running"

# Stop frontend
pkill -f "vite" && echo "✓ Frontend stopped" || echo "  Frontend was not running"

# Stop Redis container
if docker ps | grep -q llm-council-redis; then
    docker stop llm-council-redis > /dev/null 2>&1
    echo "✓ Redis stopped"
else
    echo "  Redis was not running"
fi

echo ""
echo "All services stopped."
