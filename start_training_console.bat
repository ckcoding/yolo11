@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "DISTRO=Ubuntu-24.04"
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%"
set "COMPOSE_FILE=mmyolo_small_object/docker-compose.console.yml"
set "ACTION=%~1"

if "%ACTION%"=="" set "ACTION=up"

if /I "%ACTION%"=="help" goto :help
if /I "%ACTION%"=="-h" goto :help
if /I "%ACTION%"=="--help" goto :help

call :resolve_wsl_dir
if errorlevel 1 exit /b 1

if /I "%ACTION%"=="up" goto :up
if /I "%ACTION%"=="build" goto :build
if /I "%ACTION%"=="rebuild" goto :rebuild
if /I "%ACTION%"=="down" goto :down
if /I "%ACTION%"=="logs" goto :logs
if /I "%ACTION%"=="status" goto :status
if /I "%ACTION%"=="shell" goto :shell

echo Unknown action: %ACTION%
echo.
goto :help

:up
call :run "docker compose -f %COMPOSE_FILE% up -d --build"
if errorlevel 1 exit /b 1
echo.
echo Training console should be starting.
echo URL: http://127.0.0.1:18080
echo Logs: %~nx0 logs
exit /b 0

:build
call :run "docker compose -f %COMPOSE_FILE% build --no-cache"
exit /b %errorlevel%

:rebuild
call :run "docker compose -f %COMPOSE_FILE% down && docker compose -f %COMPOSE_FILE% up -d --build"
exit /b %errorlevel%

:down
call :run "docker compose -f %COMPOSE_FILE% down"
exit /b %errorlevel%

:logs
call :run "docker compose -f %COMPOSE_FILE% logs -f --tail=200"
exit /b %errorlevel%

:status
call :run "docker compose -f %COMPOSE_FILE% ps"
exit /b %errorlevel%

:shell
call :run "bash"
exit /b %errorlevel%

:resolve_wsl_dir
set "WSL_DIR="
for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = (Resolve-Path '%PROJECT_DIR%').Path; $drive = $p.Substring(0,1).ToLower(); $rest = $p.Substring(2).Replace('\','/'); Write-Output ('/mnt/' + $drive + $rest)"`) do (
    set "WSL_DIR=%%I"
)

if not defined WSL_DIR (
    echo Failed to resolve WSL path for:
    echo   %PROJECT_DIR%
    exit /b 1
)
exit /b 0

:run
set "WSL_CMD=%~1"
echo Running in %DISTRO%:
echo   cd %WSL_DIR%
echo   %WSL_CMD%
echo.
wsl -d %DISTRO% --cd "%WSL_DIR%" bash -lc "%WSL_CMD%"
exit /b %errorlevel%

:help
echo Usage:
echo   %~nx0 up       ^(default, build and start training console^)
echo   %~nx0 build    ^(build only, no cache^)
echo   %~nx0 rebuild  ^(down, then rebuild and start^)
echo   %~nx0 down     ^(stop containers^)
echo   %~nx0 logs     ^(follow logs^)
echo   %~nx0 status   ^(show container status^)
echo   %~nx0 shell    ^(open a shell in the project directory^)
echo.
echo Target distro: %DISTRO%
echo Target path:   %PROJECT_DIR%
exit /b 0
