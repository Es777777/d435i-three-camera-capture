@echo off
setlocal
call "%~dp0software\start.bat" %*
exit /b %errorlevel%
