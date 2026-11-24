@echo off
setlocal
title Patent Reproduce Env Setup

echo ================================================
echo   Patent Reproduce Environment Setup
echo   Python 3.11 + PyTorch + common deps
echo ================================================
echo.

REM ---- 1. Locate conda ----
set "CONDA_ROOT="
if exist "C:\ProgramData\miniconda3\Scripts\conda.exe" set "CONDA_ROOT=C:\ProgramData\miniconda3"
if exist "C:\ProgramData\Miniconda3\Scripts\conda.exe" set "CONDA_ROOT=C:\ProgramData\Miniconda3"
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_ROOT=%USERPROFILE%\miniconda3"
if exist "%USERPROFILE%\miniforge3\Scripts\conda.exe" set "CONDA_ROOT=%USERPROFILE%\miniforge3"
if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_ROOT=%USERPROFILE%\anaconda3"

if not defined CONDA_ROOT (
    echo [ERROR] conda not found!
    echo Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html
    echo Check "Add to PATH" during install, then re-run this script.
    pause
    exit /b 1
)

set "PATH=%CONDA_ROOT%;%CONDA_ROOT%\Scripts;%CONDA_ROOT%\Library\bin;%PATH%"
echo [OK] conda found: %CONDA_ROOT%

REM ---- 2. Detect GPU ----
set HAS_GPU=0
where nvidia-smi >nul 2>nul
if not errorlevel 1 set HAS_GPU=1

echo [1/4] Creating conda env "patent" (Python 3.11)...
call conda create -n patent python=3.11 -y
if errorlevel 1 echo [INFO] Env may already exist, reusing it.

echo.
echo [2/4] Activating env...
call conda activate patent

echo.
echo [3/4] Installing PyTorch...
if "%HAS_GPU%"=="1" (
    echo GPU detected, installing CUDA 12.8 build (RTX 50 series support)...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
) else (
    echo No GPU detected, installing CPU build...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo [4/4] Installing common deps...
python -m pip install numpy opencv-python pillow matplotlib tqdm scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo ================================================
echo   Verifying...
echo ================================================
python -c "import torch, cv2, numpy; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('OpenCV:', cv2.__version__)"

echo.
echo ================================================
echo   Done! Next time run: conda activate patent
echo ================================================
pause
