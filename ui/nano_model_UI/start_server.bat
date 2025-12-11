@echo off
echo Starting local server at http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

REM Try to find a valid python command
where py >nul 2>nul
if %errorlevel% equ 0 (
    py server.py
    goto end
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    python server.py
    goto end
)

where python3 >nul 2>nul
if %errorlevel% equ 0 (
    python3 server.py
    goto end
)

echo.
echo ERROR: Python not found. Please install Python and add it to PATH.
echo.
echo Unable to start local server.
pause
exit /b 1

:end
echo.
echo Server session ended.
pause