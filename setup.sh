#!/bin/bash

# Path to the Isaac Lab repository
ISAACLAB_PATH="/home/shuosha/Desktop/IsaacLab/IsaacLab"

# Create a symbolic link
ln -s "$ISAACLAB_PATH" ./isaaclab_packages

echo "Symbolic link to Isaac Lab packages created."
