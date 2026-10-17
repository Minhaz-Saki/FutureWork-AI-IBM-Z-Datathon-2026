@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  set "PY=python"
)
echo Checking Python...
%PY% --version || (echo Install Python 3 from python.org and try again. & pause & exit /b 1)
echo Installing dependencies (may take a moment)...
%PY% -m pip install -r requirements-ui.txt || (echo Dependency installation failed. & pause & exit /b 1)
echo.
echo Open http://localhost:8899 in your browser.
echo Keep this window open. Press Ctrl+C to stop.
%PY% app.py --port 8899
pause
