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
py server.py 2>nul
if %errorlevel% neq 0 (
	echo py command not found, trying python...
	python server.py 2>nul
	if %errorlevel% neq 0 (
		echo python command not found, trying python3...
		python3 server.py 2>nul
		if %errorlevel% neq 0 (
			echo.
			echo ERROR: Python not found. Please install Python and add it to PATH.
			pause
			exit /b 1
		)
	)
)
