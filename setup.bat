@echo off
REM LLM Council - Setup Script for Windows

echo ==========================================
echo LLM Council - Setup
echo ==========================================
echo.

cd /d "%~dp0"

REM Check Python
echo Checking Python...
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [91mX Python is not installed. Please install Python 3.10 or later.[0m
    echo Download from: https://www.python.org/downloads/
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [92m✓ Python %PYTHON_VERSION% found[0m
echo.

REM Check Node.js
echo Checking Node.js...
where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [91mX Node.js is not installed. Please install Node.js 20.19+ or 22.12+[0m
    echo Download from: https://nodejs.org/
    exit /b 1
)
for /f "tokens=1" %%i in ('node --version') do set NODE_VERSION=%%i
echo [92m✓ Node.js %NODE_VERSION% found[0m
echo.

REM Check Docker
echo Checking Docker...
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [91mX Docker is not installed. Please install Docker Desktop.[0m
    echo Download from: https://www.docker.com/products/docker-desktop
    exit /b 1
)
echo [92m✓ Docker found[0m
echo.

REM Create virtual environment
echo Installing Python dependencies...
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Install dependencies
where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Using uv...
    uv sync
) else (
    echo Using pip in virtual environment...
    .venv\Scripts\pip install --upgrade pip
    .venv\Scripts\pip install fastapi uvicorn httpx pydantic python-dotenv redis rq
)
echo [92m✓ Python dependencies installed[0m
echo.

REM Install frontend dependencies
echo Installing frontend dependencies...
cd frontend
call npm install
cd ..
echo [92m✓ Frontend dependencies installed[0m
echo.

REM Setup environment file
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env
    echo [93m! Please edit .env and add your OPENROUTER_API_KEY[0m
    echo   Get your API key from: https://openrouter.ai/keys
    echo.
) else (
    echo [92m✓ .env file already exists[0m
    echo.
)

REM Load Redis port from .env (default to 6380)
set REDIS_PORT=6380
for /f "tokens=2 delims==" %%a in ('findstr /r "^REDIS_PORT=" .env 2^>nul') do set REDIS_PORT=%%a

REM Setup Redis
echo Setting up Redis...
docker ps | findstr llm-council-redis >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [92m✓ Redis container already running[0m
) else (
    docker ps -a | findstr llm-council-redis >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo Starting existing Redis container...
        docker start llm-council-redis
        echo [92m✓ Redis container started[0m
    ) else (
        echo Creating Redis container...
        docker run -d --name llm-council-redis -p %REDIS_PORT%:6379 --restart unless-stopped redis:7-alpine
        echo [92m✓ Redis container created and started[0m
    )
)
echo.

echo ==========================================
echo [92m✓ Setup Complete![0m
echo ==========================================
echo.
echo Next steps:
echo 1. Edit .env and add your OPENROUTER_API_KEY if not done yet
echo 2. Run: start-background.bat
echo 3. Open: http://localhost:5173
echo.
pause
