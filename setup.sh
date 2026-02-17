#!/bin/bash
#
# Highway-env Environment Setup Script
# Removes existing venv, creates a fresh one, installs dependencies, and tests.
#

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "=== Highway-env Setup Script ==="
echo ""

# Step 1: Remove existing virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "[1/5] Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
    echo "      Done."
else
    echo "[1/5] No existing virtual environment found."
fi

# Step 2: Create new virtual environment
echo "[2/5] Creating new virtual environment..."
python3 -m venv "$VENV_DIR"
echo "      Done."

# Step 3: Install dependencies
echo "[3/5] Installing dependencies..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
echo "      Done."

# Step 4: Test basic environment
echo "[4/5] Testing basic environment..."
python3 -c "
import gymnasium as gym
import highway_env

env = gym.make('highway-v0', render_mode='rgb_array')
obs, info = env.reset()

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
print('      Basic test passed!')
"

# Step 5: Test video recording
echo "[5/5] Testing video recording..."
python3 -c "
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
import shutil
import os

# Clean test folder
test_folder = 'test_video'
if os.path.exists(test_folder):
    shutil.rmtree(test_folder)

env = gym.make('highway-v0', render_mode='rgb_array')
env = RecordVideo(env, video_folder=test_folder, episode_trigger=lambda e: True)
env.unwrapped.set_record_video_wrapper(env)

obs, info = env.reset()
for _ in range(3):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()

# Verify video was created
videos = [f for f in os.listdir(test_folder) if f.endswith('.mp4')]
assert len(videos) > 0, 'No video file created'

# Clean up
shutil.rmtree(test_folder)
print('      Video recording test passed!')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Examples:"
echo "  python example.py              # Basic highway example"
echo "  python example_intersection.py # Intersection with video recording"
