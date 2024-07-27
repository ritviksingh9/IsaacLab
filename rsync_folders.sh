#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="/home/ritviks/workspace/git/IsaacLab/"
DEST_DIR="/home/ritviks/workspace/git/osmo_IsaacLab/"

# Run rsync with the specified exclusions
rsync -av --exclude='__pycache__' --exclude='runs' --exclude='.git' --exclude='logs' --exclude _isaac_sim "$SOURCE_DIR" "$DEST_DIR"

# Print completion message
echo "Sync completed from $SOURCE_DIR to $DEST_DIR"