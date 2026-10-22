@echo off
setlocal
cd /d "%~dp0"
set "NAV_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%NAV_PYTHON%" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "PYTHONUTF8=1"
"%NAV_PYTHON%" -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
if errorlevel 1 pause
