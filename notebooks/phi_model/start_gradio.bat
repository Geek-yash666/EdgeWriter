@echo off
title EdgeWriter - Gradio UI (Phi-3)
echo.
echo ============================================
echo   EdgeWriter - Gradio Interface
echo   Phi-3 Mini Local Writing Assistant
echo ============================================
echo.
echo Starting Gradio server...
echo Browser will open automatically at http://127.0.0.1:7860
echo.
echo Press Ctrl+C to stop the server
echo.
cd /d "%~dp0"
py gradio_app.py
pause
