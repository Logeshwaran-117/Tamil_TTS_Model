@echo off
title Fish Speech S2 - CPU Fine-Tuning
echo ========================================================
echo   Fish Speech S2 Tamil Voice CPU Fine-Tuning
echo ========================================================
echo.

cd /d "%~dp0"

set HF_HOME=%~dp0cache\huggingface
set PIP_CACHE_DIR=%~dp0cache\pip
set TORCH_HOME=%~dp0cache\torch
set TRANSFORMERS_CACHE=%~dp0cache\huggingface
set HUGGINGFACE_HUB_CACHE=%~dp0cache\huggingface

.venv311\Scripts\python.exe train_local_cpu.py
echo.
echo ========================================================
echo   Fine-tuning finished! Checkpoint saved in checkpoints/
echo ========================================================
pause
