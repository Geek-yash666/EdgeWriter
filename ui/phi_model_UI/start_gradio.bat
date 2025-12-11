@echo off
setlocal

REM --- Ensure Python launcher is available ---
where py >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python launcher "py" not found in PATH.
    echo Install Python from https://www.python.org/downloads/ and ensure "py" is in PATH.
    pause
    exit /b 1
)
set "PYTHON_CMD=py"

title EdgeWriter - Phi-3 Gradio UI
echo.
echo ============================================
echo   EdgeWriter - Phi-3 Gradio Interface
echo ============================================
echo.
echo Starting Gradio server...
echo Browser will open automatically when ready.
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"

REM --- Check GPU type (looks for NVIDIA/CUDA capability) ---
set "HAS_NVIDIA=0"
wmic path win32_VideoController get Name | findstr /I "NVIDIA" >nul 2>&1 && set "HAS_NVIDIA=1"

REM --- Check if llama_cpp_python is installed ---
%PYTHON_CMD% -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('llama_cpp') else 1)"
if errorlevel 1 (
    echo [llama_cpp_python] Not installed.
    if "%HAS_NVIDIA%"=="1" (
        echo NVIDIA GPU detected. For CUDA acceleration, install the CUDA wheel:
        echo   pip install --upgrade --force-reinstall --no-cache-dir ^
https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp310-cp310-win_amd64.whl
    ) else (
        echo No NVIDIA GPU detected ^(likely integrated graphics^). Install llama_cpp_python if you want to run locally; the CPU build will be slower.
    )
) else (
    echo [llama_cpp_python] Installed.
    call :detect_llama_cuda

    if /I "%LLAMA_CUDA%"=="CUDA_ON" (
        echo [llama_cpp_python] CUDA build detected.
    ) else (
        echo [llama_cpp_python] CPU-only build detected.
        if "%HAS_NVIDIA%"=="1" (
            echo Detected NVIDIA GPU. For CUDA acceleration, install the CUDA wheel:
            echo   pip install --upgrade --force-reinstall --no-cache-dir ^
https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp310-cp310-win_amd64.whl
        ) else (
            echo No NVIDIA GPU detected ^(likely integrated graphics^). This CPU build may be slow and CUDA is not supported on your system.
        )
    )
)

%PYTHON_CMD% gradio_app.py
if errorlevel 1 (
    echo [ERROR] gradio_app.py exited with a non-zero status.
)

pause
endlocal
exit /b

:detect_llama_cuda
set "LLAMA_CUDA=CUDA_OFF"
for /f %%i in ('
    %PYTHON_CMD% -c "from llama_cpp import Llama; import sys; info=Llama.build_info(); sys.stdout.write('CUDA_ON' if str(info.get('cuda','OFF')).upper()=='ON' else 'CUDA_OFF')" 2^>NUL
') do set "LLAMA_CUDA=%%i"
exit /b
