@echo off
echo ========================================
echo   EdgeWriter - Dual Engine
echo ========================================
echo.
echo Starting server with Phi-3 model...
echo.
echo UI will be available at: http://127.0.0.1:8000
echo.
echo Features:
echo   - Base Model (MediaPipe in browser)
echo   - Phi-3 Mini (local GPU inference)
echo.
echo Press Ctrl+C to stop the server.
echo ========================================
echo.

python server.py
