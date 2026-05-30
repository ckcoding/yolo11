@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
set "CONSOLE_DIR=%SCRIPT_DIR%mmyolo_small_object"
set "VENV_DIR=%CONSOLE_DIR%\.venv-console"
set "PYTHON_EXE=py -3.12"
set "ACTION=%~1"

if "%ACTION%"=="" set "ACTION=run"

if /I "%ACTION%"=="help" goto :help
if /I "%ACTION%"=="-h" goto :help
if /I "%ACTION%"=="--help" goto :help
if /I "%ACTION%"=="setup" goto :setup
if /I "%ACTION%"=="run" goto :run
if /I "%ACTION%"=="status" goto :status
if /I "%ACTION%"=="open" goto :open

echo Unknown action: %ACTION%
echo.
goto :help

:setup
call :ensure_venv
if errorlevel 1 exit /b 1
call :install_deps
exit /b %errorlevel%

:run
call :ensure_venv
if errorlevel 1 exit /b 1
call :install_deps
if errorlevel 1 exit /b 1
echo Starting MMYOLO trainer console...
echo URL: http://127.0.0.1:18080
echo.
start "MMYOLO Trainer Console" cmd /k "cd /d ""%CONSOLE_DIR%"" && call ""%VENV_DIR%\Scripts\activate.bat"" && set PYTHONPATH=%CONSOLE_DIR% && set TRAINER_PROJECT_ROOT=%SCRIPT_DIR% && set TRAINER_STATE_ROOT=%CONSOLE_DIR%\runtime && set TRAINER_BROWSE_ROOTS=%SCRIPT_DIR%;%USERPROFILE%\Desktop;%USERPROFILE% && set TRAINER_DATASET_SCAN_ROOT=%USERPROFILE%\Desktop && set TRAINER_HOST=127.0.0.1 && set TRAINER_PORT=18080 && python -m trainer_console.main"
exit /b 0

:status
netstat -ano | findstr :18080
exit /b %errorlevel%

:open
start "" "http://127.0.0.1:18080"
exit /b 0

:ensure_venv
if exist "%VENV_DIR%\Scripts\python.exe" exit /b 0
echo Creating trainer console virtual environment...
%PYTHON_EXE% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo Failed to create venv with %PYTHON_EXE%
    exit /b 1
)
exit /b 0

:install_deps
call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install -r "%CONSOLE_DIR%\requirements-console.txt" psutil PyYAML Pillow
if errorlevel 1 exit /b 1
exit /b 0

:help
echo Usage:
echo   %~nx0 setup   ^(create venv and install web console deps^)
echo   %~nx0 run     ^(default, start console at http://127.0.0.1:18080^)
echo   %~nx0 status  ^(check whether port 18080 is listening^)
echo   %~nx0 open    ^(open console in browser^)
echo.
echo Console dir: %CONSOLE_DIR%
exit /b 0
