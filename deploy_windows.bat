@echo off
REM Windows deployment script for RunPod
REM This script creates an archive and deploys to the pod

set POD_HOST=69.30.85.20
set POD_PORT=22018
set POD_USER=root

echo ==================================================================
echo SPD Deployment Script for RunPod (Windows)
echo ==================================================================
echo Target: %POD_USER%@%POD_HOST%:%POD_PORT%
echo.

REM Test SSH connection
echo Testing SSH connection...
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p %POD_PORT% %POD_USER%@%POD_HOST% "echo SSH connection successful"
if %ERRORLEVEL% NEQ 0 (
    echo Failed to connect to pod. Please check connection details.
    exit /b 1
)
echo SSH connection verified!
echo.

REM Create archive name with timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set ARCHIVE_NAME=spd_deploy_%datetime:~0,8%_%datetime:~8,6%.tar.gz

echo Creating archive: %ARCHIVE_NAME%
tar -czf %ARCHIVE_NAME% ^
    --exclude=.git ^
    --exclude=.venv ^
    --exclude=__pycache__ ^
    --exclude=*.pyc ^
    --exclude=.pytest_cache ^
    --exclude=wandb ^
    --exclude=outputs ^
    --exclude=*.egg-info ^
    --exclude=docs/coverage ^
    --exclude=*.tar.gz ^
    --exclude=deploy_windows.bat ^
    .

if %ERRORLEVEL% NEQ 0 (
    echo Failed to create archive
    exit /b 1
)
echo Archive created successfully!
echo.

REM Copy archive to pod
echo Copying archive to pod...
scp -o StrictHostKeyChecking=no -P %POD_PORT% %ARCHIVE_NAME% %POD_USER%@%POD_HOST%:/workspace/
if %ERRORLEVEL% NEQ 0 (
    echo Failed to copy archive to pod
    del %ARCHIVE_NAME%
    exit /b 1
)
echo Archive copied successfully!
echo.

REM Extract and setup on pod
echo Setting up on pod...
ssh -o StrictHostKeyChecking=no -p %POD_PORT% %POD_USER%@%POD_HOST% "cd /workspace && tar -xzf %ARCHIVE_NAME% -C /workspace/spd/sjcs-spd/ 2>/dev/null || (mkdir -p /workspace/spd/sjcs-spd && tar -xzf %ARCHIVE_NAME% -C /workspace/spd/sjcs-spd/) && rm %ARCHIVE_NAME% && cd /workspace/spd/sjcs-spd && chmod +x setup_runpod_a40.sh && echo 'Extraction complete!'"

echo Running setup script on pod...
ssh -o StrictHostKeyChecking=no -p %POD_PORT% %POD_USER%@%POD_HOST% "cd /workspace/spd/sjcs-spd && ./setup_runpod_a40.sh"

REM Clean up local archive
echo Cleaning up...
del %ARCHIVE_NAME%

echo.
echo ==================================================================
echo DEPLOYMENT COMPLETE!
echo ==================================================================
echo.
echo To start training, SSH into the pod:
echo   ssh -p %POD_PORT% %POD_USER%@%POD_HOST%
echo.
echo Then run:
echo   cd /workspace/spd/sjcs-spd
echo   ./launch_training.sh
echo.
pause