@echo off
rem Aircraft Taxi Simulator launcher for Windows - double-click.
rem First run creates a private Python environment (.venv-win) and installs numpy + matplotlib.
cd /d "%~dp0"
set "VENV=.venv-win"
set "VPY=%VENV%\Scripts\python.exe"

if exist "%VENV%\ats_ready.txt" goto run

set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY goto nopython

echo First run: setting up Python environment (one minute)...
if not exist "%VPY%" (
    %PY% -m venv "%VENV%"
    if errorlevel 1 goto fail
)
"%VPY%" -m pip install --quiet --upgrade pip
"%VPY%" -m pip install --quiet -r requirements.txt
if errorlevel 1 goto fail
echo ok> "%VENV%\ats_ready.txt"

:run
"%VPY%" main.py
if errorlevel 1 goto fail
exit /b 0

:nopython
echo.
echo ERROR: Python 3 not found. Install it from https://www.python.org/downloads/
echo        (tick "Add python.exe to PATH" and keep "tcl/tk and IDLE" selected).
pause
exit /b 1

:fail
echo.
echo ERROR: setup or Aircraft Taxi Simulator failed - see messages above.
pause
exit /b 1
