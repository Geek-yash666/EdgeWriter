@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM --- Find a usable Python command (prefer the Windows launcher) ---
set "PYTHON_CMD="
where py >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py"
if not defined PYTHON_CMD (
    where python >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    where python3 >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python3"
)
if not defined PYTHON_CMD (
    echo.
    echo [ERROR] Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

title EdgeWriter - Integrated Dual UI
echo.
echo ========================================
echo   EdgeWriter - Dual Engine
echo ========================================
echo.
echo Starting Integrated UI server...
echo UI will be available at: http://127.0.0.1:8000
echo.
echo Features:
echo   - Base Model (browser / on-device)
echo   - Phi-3 Mini (local llama.cpp inference)
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"

REM --- GPU hinting for llama-cpp-python on Windows ---
set "HAS_NVIDIA=0"
wmic path win32_VideoController get Name | findstr /I "NVIDIA" >nul 2>&1 && set "HAS_NVIDIA=1"

%PYTHON_CMD% -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('llama_cpp') else 1)"
if errorlevel 1 (
    echo [llama_cpp_python] Not installed.
    echo Install dependencies first:
    echo   pip install -r requirements.txt
    if "%HAS_NVIDIA%"=="1" (
        echo.
        echo NVIDIA GPU detected. For CUDA acceleration, install the CUDA wheel ^(adjust for your Python/CUDA^):
        echo   pip install --upgrade --force-reinstall --no-cache-dir
        echo   https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp310-cp310-win_amd64.whl
    )
) else (
    echo [llama_cpp_python] Installed.
    call :detect_llama_cuda

    if /I "%LLAMA_CUDA%"=="CUDA_ON" (
        echo [llama_cpp_python] CUDA build detected.
    ) else (
        echo [llama_cpp_python] CPU-only build detected.
        if "%HAS_NVIDIA%"=="1" (
            echo Detected NVIDIA GPU. For CUDA acceleration, install the CUDA wheel ^(adjust for your Python/CUDA^):
            echo   pip install --upgrade --force-reinstall --no-cache-dir
            echo   https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp310-cp310-win_amd64.whl
        ) else (
            echo No NVIDIA GPU detected. This CPU build may be slow.
        )
    )
)

%PYTHON_CMD% server.py
if errorlevel 1 (
    echo [ERROR] server.py exited with a non-zero status.
)

pause
endlocal
exit /b

:detect_llama_cuda
set "LLAMA_CUDA=CUDA_OFF"
for /f %%i in ('%PYTHON_CMD% -c "from llama_cpp import Llama; import sys; info=Llama.build_info(); sys.stdout.write(\"CUDA_ON\" if str(info.get(\"cuda\",\"OFF\")).upper()==\"ON\" else \"CUDA_OFF\")" 2^>NUL') do set "LLAMA_CUDA=%%i"
exit /b