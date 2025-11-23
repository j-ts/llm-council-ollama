# Backend Dockerfile for LLM Council
FROM python:3.12-slim

WORKDIR /app

# Install uv - fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy Python project files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy backend source code
COPY backend/ ./backend/

# Create data directory for conversation storage
RUN mkdir -p data/conversations

# Expose backend port
EXPOSE 8001

# Run the backend using uv
CMD ["uv", "run", "python", "-m", "backend.main"]
