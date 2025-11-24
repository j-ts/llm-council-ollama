@echo off
REM LLM Council - Stop All Services (Windows)

echo Stopping LLM Council services...

REM Stop Python processes
taskkill /FI "WINDOWTITLE eq LLM Council Backend*" /F >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [92m✓ Backend stopped[0m
) else (
    echo   Backend was not running
)

taskkill /FI "WINDOWTITLE eq LLM Council Worker*" /F >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [92m✓ Worker stopped[0m
) else (
    echo   Worker was not running
)

taskkill /FI "WINDOWTITLE eq LLM Council Frontend*" /F >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [92m✓ Frontend stopped[0m
) else (
    echo   Frontend was not running
)

REM Stop Redis container
docker ps | findstr llm-council-redis >nul 2>nul
if %ERRORLEVEL% equ 0 (
    docker stop llm-council-redis >nul 2>nul
    echo [92m✓ Redis stopped[0m
) else (
    echo   Redis was not running
)

echo.
echo All services stopped.
pause
