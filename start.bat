@echo off
setlocal
title Tag OCR Engine
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH. Install Python 3.10+ first.
    pause
    exit /b 1
)

if not exist ocr_config.json copy ocr_config.example.json ocr_config.json >nul

echo ============================================================
echo   Tag OCR Engine - local OCR + entity resolution
echo   Model: GLM-OCR via Ollama (MIT)
echo ============================================================
echo.
echo Starting at http://127.0.0.1:5005 ...
echo.

python app.py

echo.
echo Server stopped.
pause
