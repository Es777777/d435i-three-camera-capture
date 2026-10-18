@echo off
setlocal
call "%~dp0start.bat" -Mode capture %*
exit /b %errorlevel%
