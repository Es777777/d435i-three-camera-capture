@echo off
setlocal
call "%~dp0start.bat" -SetupOnly %*
set "capture_exit=%errorlevel%"
if "%capture_exit%"=="0" pause
exit /b %capture_exit%
