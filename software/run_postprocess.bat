@echo off
setlocal
call "%~dp0start.bat" -Mode legacy %*
exit /b %errorlevel%
