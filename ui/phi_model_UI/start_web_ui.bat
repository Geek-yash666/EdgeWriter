@echo off
title EdgeWriter - Web UI (FastAPI + Phi-3)
echo.
echo ============================================
echo   EdgeWriter - Web UI Server
echo   Phi-3 Mini Local Writing Assistant
echo ============================================
echo.
echo Starting FastAPI server...
echo Browser will open automatically at http://127.0.0.1:8000
echo.
echo Press Ctrl+C to stop the server
echo.
cd /d "%~dp0"
py server.py
pause
