@echo off
REM LLM Council - Start All Services (Windows)

echo Starting LLM Council in background mode...
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv" (
    echo [91mX Virtual environment not found.[0m
    echo   Please run: setup.bat
    exit /b 1
)

REM Check if Python dependencies are installed
.venv\Scripts\python -c "import fastapi" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [91mX Python dependencies not installed.[0m
    echo   Please run: setup.bat
    exit /b 1
)

REM Check if frontend dependencies are installed
if not exist "frontend\node_modules" (
    echo [91mX Frontend dependencies not installed.[0m
    echo   Please run: setup.bat
    exit /b 1
)

REM Check if .env exists
if not exist ".env" (
    echo [91mX .env file not found.[0m
    echo   Please run: setup.bat
    exit /b 1
)

REM Load Redis port from .env (default to 6380)
set REDIS_PORT=6380
for /f "tokens=2 delims==" %%a in ('findstr /r "^REDIS_PORT=" .env 2^>nul') do set REDIS_PORT=%%a

REM Check if Redis container is running
docker ps | findstr llm-council-redis >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [93m! Redis container not running. Starting...[0m
    docker start llm-council-redis >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        docker run -d --name llm-council-redis -p %REDIS_PORT%:6379 --restart unless-stopped redis:7-alpine
    )
    timeout /t 2 /nobreak >nul
)

REM Start backend
echo Starting backend...
start "LLM Council Backend" /B .venv\Scripts\python -m backend.main > backend.log 2>&1

timeout /t 2 /nobreak >nul

REM Start worker
echo Starting worker...
REM Note: OBJC_DISABLE_INITIALIZE_FORK_SAFETY not needed on Windows
start "LLM Council Worker" /B .venv\Scripts\python -m rq.cli worker council --url redis://localhost:%REDIS_PORT% > worker.log 2>&1

timeout /t 2 /nobreak >nul

REM Start frontend
echo Starting frontend...
cd frontend
start "LLM Council Frontend" /B cmd /c "npm run dev > frontend.log 2>&1"
cd ..

echo.
echo [92m✓ LLM Council is running![0m
echo.
echo Services:
echo   Backend:  http://localhost:8001
echo   Frontend: http://localhost:5173
echo   Redis:    localhost:%REDIS_PORT% (Docker container)
echo.
echo Logs:
echo   Backend:  type backend.log
echo   Worker:   type worker.log
echo   Frontend: type frontend\frontend.log
echo.
echo To stop: stop.bat (or close the command windows)
echo.
