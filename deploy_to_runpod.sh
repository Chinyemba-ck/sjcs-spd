#!/bin/bash
# Deployment script to copy code to RunPod and execute setup
# This script runs from your local machine to deploy to the pod

set -euo pipefail

# Pod connection details
POD_HOST="69.30.85.20"
POD_PORT="22018"
POD_USER="root"
POD_NAME="aware_fuchsia_guanaco"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Functions for colored output
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "==================================================================="
echo "SPD Deployment Script for RunPod"
echo "==================================================================="
echo "Target Pod: $POD_NAME"
echo "Connection: $POD_USER@$POD_HOST:$POD_PORT"
echo ""

# Step 1: Test SSH connection
log_info "Testing SSH connection to pod..."
ssh -o ConnectTimeout=5 -p $POD_PORT $POD_USER@$POD_HOST "echo 'SSH connection successful'" || {
    log_error "Failed to connect to pod. Please check connection details."
    exit 1
}
log_success "SSH connection verified"

# Step 2: Create archive of current repository
log_info "Creating archive of SPD repository..."
ARCHIVE_NAME="spd_deployment_$(date +%Y%m%d_%H%M%S).tar.gz"

# Create archive excluding unnecessary files
tar -czf "$ARCHIVE_NAME" \
    --exclude=".git" \
    --exclude=".venv" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude=".pytest_cache" \
    --exclude="wandb" \
    --exclude="outputs" \
    --exclude="*.egg-info" \
    --exclude="docs/coverage" \
    --exclude="*.tar.gz" \
    .

log_success "Archive created: $ARCHIVE_NAME"

# Step 3: Copy archive to pod
log_info "Copying archive to pod..."
scp -P $POD_PORT "$ARCHIVE_NAME" $POD_USER@$POD_HOST:/workspace/ || {
    log_error "Failed to copy archive to pod"
    rm "$ARCHIVE_NAME"
    exit 1
}
log_success "Archive copied to pod"

# Step 4: Extract and setup on pod
log_info "Extracting archive and running setup on pod..."
ssh -p $POD_PORT $POD_USER@$POD_HOST << 'REMOTE_SCRIPT'
set -euo pipefail

# Color codes for remote output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[REMOTE]${NC} $1"; }
log_success() { echo -e "${GREEN}[REMOTE]${NC} $1"; }
log_error() { echo -e "${RED}[REMOTE]${NC} $1"; }

# Navigate to workspace
cd /workspace

# Find the latest archive
LATEST_ARCHIVE=$(ls -t spd_deployment_*.tar.gz 2>/dev/null | head -1)
if [ -z "$LATEST_ARCHIVE" ]; then
    log_error "No deployment archive found"
    exit 1
fi

log_info "Found archive: $LATEST_ARCHIVE"

# Create SPD directory if it doesn't exist
mkdir -p /workspace/spd/sjcs-spd

# Extract archive
log_info "Extracting archive..."
tar -xzf "$LATEST_ARCHIVE" -C /workspace/spd/sjcs-spd/
log_success "Archive extracted"

# Clean up archive
rm "$LATEST_ARCHIVE"

# Navigate to project directory
cd /workspace/spd/sjcs-spd

# Make setup script executable
chmod +x setup_runpod_a40.sh

# Run setup script
log_info "Running setup script..."
./setup_runpod_a40.sh

log_success "Remote setup completed!"
REMOTE_SCRIPT

# Step 5: Clean up local archive
log_info "Cleaning up local archive..."
rm "$ARCHIVE_NAME"
log_success "Local cleanup completed"

# Step 6: Final instructions
echo ""
echo "==================================================================="
echo "DEPLOYMENT COMPLETE!"
echo "==================================================================="
echo ""
echo "The SPD environment has been deployed and set up on the RunPod."
echo ""
echo "To connect to the pod and start training:"
echo "  ssh -p $POD_PORT $POD_USER@$POD_HOST"
echo ""
echo "Once connected:"
echo "  cd /workspace/spd/sjcs-spd"
echo "  ./launch_training.sh          # Start distributed training"
echo ""
echo "In another terminal (for monitoring):"
echo "  ssh -p $POD_PORT $POD_USER@$POD_HOST"
echo "  cd /workspace/spd/sjcs-spd"
echo "  ./monitor_training.sh         # Monitor GPU usage and logs"
echo ""
echo "W&B Dashboard:"
echo "  https://wandb.ai/SJCS-SPD"
echo ""
log_success "Deployment script completed successfully!"