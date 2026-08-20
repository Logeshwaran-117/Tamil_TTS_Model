@echo off
title Indic F5 - CPU Fine-Tuning
echo ========================================================
echo   Indic F5 Tamil Voice CPU Fine-Tuning
echo ========================================================
echo.

cd /d "%~dp0"

set PIP_CACHE_DIR=%~dp0cache\pip
set TORCH_HOME=%~dp0cache\torch

.venv311\Scripts\python.exe train_local_cpu.py
echo.
echo ========================================================
echo   Fine-tuning finished! Checkpoint saved in checkpoints/
echo ========================================================
pause
