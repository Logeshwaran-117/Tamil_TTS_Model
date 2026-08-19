@echo off
title Fish Speech S2 Tamil TTS
echo ========================================================
echo   Starting Fish Speech S2 Tamil TTS Local Server
echo ========================================================
echo.

cd /d "%~dp0"

set HF_HOME=%~dp0cache\huggingface
set PIP_CACHE_DIR=%~dp0cache\pip
set TORCH_HOME=%~dp0cache\torch
set TRANSFORMERS_CACHE=%~dp0cache\huggingface
set HUGGINGFACE_HUB_CACHE=%~dp0cache\huggingface

start "" "tts_ui.html"
.venv311\Scripts\python.exe tts_backend.py
pause
