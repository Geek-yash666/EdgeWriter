@echo off
echo Starting local server at http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

py -m http.server 2>nul
if %errorlevel% neq 0 (
    echo py command not found, trying python...
    python -m http.server 2>nul
    if %errorlevel% neq 0 (
        echo python command not found, trying python3...
        python3 -m http.server 2>nul
        if %errorlevel% neq 0 (
            echo.
            echo ERROR: Python not found. Please install Python and add it to PATH.
            pause
            exit /b 1
        )
    )
)
pause
