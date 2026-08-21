@echo off
title TruthLens AI Forensics Platform
cd /d "%~dp0"

set PORT=8000
echo =================================================================
echo   TRUTHLENS AI DIGITAL FORENSICS PLATFORM
echo   Starting Production Inference Server on http://localhost:%PORT%
echo =================================================================
echo   Web Dashboard: http://localhost:%PORT%/ui
echo   Swagger Docs:  http://localhost:%PORT%/docs
echo =================================================================
echo.

python main.py
if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with an error code.
    pause
)
