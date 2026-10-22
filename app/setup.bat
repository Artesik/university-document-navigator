@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
if errorlevel 1 (
  echo Installation failed. Read the error above.
  pause
  exit /b 1
)
pause
