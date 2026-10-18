@echo off
setlocal
call "%~dp0start.bat" -Mode settlement %*
exit /b %errorlevel%
