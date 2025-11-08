#!/bin/bash

# Get the list of unique PIDs using the GPU
PIDS=$(nvidia-smi pmon -c 1 | grep -v '^#' | awk '{print $2}' | sort -u)

# Check if any PIDs were found
if [ -z "$PIDS" ]; then
    echo "No GPU processes found."
else
    # Loop through each PID and kill it
    for PID in $PIDS; do
        echo "Killing PID $PID"
        kill -9 "$PID"
    done
    echo "All GPU processes terminated."
fi