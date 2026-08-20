@echo off
title Indic F5 Tamil TTS Studio
echo ========================================================
echo   Starting Indic F5 Tamil TTS Local Server
echo ========================================================
echo.

cd /d "%~dp0"

set PIP_CACHE_DIR=%~dp0cache\pip
set TORCH_HOME=%~dp0cache\torch

start "" "tts_ui.html"
.venv311\Scripts\python.exe tts_backend.py
pause
