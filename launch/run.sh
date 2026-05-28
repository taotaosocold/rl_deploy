#!/bin/bash
# ============================================================
# RL Deploy — Launch script for BeyondMimic pipeline
# Usage:
#   bash launch/run.sh                    # use default config
#   bash launch/run.sh path/to/config.yaml # use custom config
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"

# Source ROS2
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

# ONNX Runtime library path (from hl_motion package)
# ONNX Runtime — adjust this path to match your robot's hl_motion location
HL_MOTION_LIB="${HL_MOTION_LIB:-/workspace/hl_motion/lib}"
export LD_LIBRARY_PATH="${HL_MOTION_LIB}:${LD_LIBRARY_PATH}"

# Source own workspace if built with colcon
if [ -f "${BUILD_DIR}/install/setup.bash" ]; then
    source "${BUILD_DIR}/install/setup.bash"
fi

# First build
if [ ! -f "${BUILD_DIR}/rl_deploy_node" ]; then
    echo "=== First run: building ==="
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cd "${PROJECT_DIR}"
fi

CONFIG="${1:-${PROJECT_DIR}/config/rl_config.yaml}"
echo "=== Starting RL Deploy Node ==="
echo "Config: ${CONFIG}"

# Run from project root so relative paths in config resolve correctly
cd "${PROJECT_DIR}"
"${BUILD_DIR}/rl_deploy_node" "${CONFIG}"
