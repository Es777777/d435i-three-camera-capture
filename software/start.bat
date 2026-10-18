@echo off
setlocal
chcp 65001 >nul
title D435i - Starting
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\bootstrap.ps1" %*
set "capture_exit=%errorlevel%"
if not "%capture_exit%"=="0" (
  echo.
  echo Startup failed. See the message above and software\logs\bootstrap.log.
  pause
)
exit /b %capture_exit%
