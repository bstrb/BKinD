@echo off
set SCRIPT_DIR=%~dp0
rem Remove trailing backslash
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
rem Convert Windows path to WSL path
for /f "tokens=* usebackq" %%i in (`wsl wslpath "%SCRIPT_DIR%"`) do set WSL_SCRIPT_PATH=%%i/setup/setup_and_run.sh
wsl -e bash -ic "source ~/.bashrc; bash %WSL_SCRIPT_PATH%"
if %errorlevel% neq 0 (
    echo.
    echo There was an error running the script. Press any key to close this window.
) else (
    echo.
    echo Script execution finished successfully. Press any key to close this window.
)
pause >nul
