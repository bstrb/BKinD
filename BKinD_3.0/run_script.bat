@echo off
wsl -e bash -ic "source ~/.bashrc; /mnt/c/Users/bubl3932/Desktop/BKinD/BKinD_3.0/setup_and_run.sh"
if %errorlevel% neq 0 (
    echo.
    echo There was an error running the script. Press any key to close this window.
) else (
    echo.
    echo Script execution finished successfully. Press any key to close this window.
)
pause >nul
