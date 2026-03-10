#!/bin/bash
#
# Experiment 6 Setup — installs rl-agents and patches gymnasium compatibility
#
# Run from the project root with the venv already activated:
#   source venv/bin/activate
#   bash experiments/6_Social_attention_with_rl_agents/setup.sh
#

set -e

echo "=== Experiment 6 Setup ==="
echo ""

# Verify venv is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: No virtual environment active."
    echo "Run:  source venv/bin/activate"
    exit 1
fi

SITE_PACKAGES="$VIRTUAL_ENV/lib/python3.10/site-packages"

# Step 1: Install rl-agents and dependencies
echo "[1/2] Installing rl-agents and tensorboard..."
pip install git+https://github.com/eleurent/rl-agents tensorboard --quiet
echo "      Done."

# Step 2: Patch rl-agents for gymnasium 1.x compatibility
echo "[2/2] Patching rl-agents for gymnasium compatibility..."
python3 -c "
import pathlib

site = pathlib.Path('$SITE_PACKAGES')
rl_dir = site / 'rl_agents' / 'trainer'

# Patch 1: evaluation.py — capped_cubic_video_schedule removed in gymnasium 1.x
eval_py = rl_dir / 'evaluation.py'
src = eval_py.read_text()
old = 'from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, capped_cubic_video_schedule'
new = '''from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
try:
    from gymnasium.wrappers import capped_cubic_video_schedule
except ImportError:
    def capped_cubic_video_schedule(episode_id: int) -> bool:
        if episode_id < 1000:
            return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
        else:
            return episode_id % 1000 == 0'''
if old in src:
    eval_py.write_text(src.replace(old, new))
    print('      Patched evaluation.py')
else:
    print('      evaluation.py already patched or unchanged')

# Patch 2: logger.py — gym.logger.INFO and set_level removed in gymnasium 1.x
logger_py = rl_dir / 'logger.py'
src = logger_py.read_text()
changed = False
old1 = 'def configure(config={}, gym_level=gym.logger.INFO):'
new1 = 'def configure(config={}, gym_level=getattr(gym.logger, \"INFO\", __import__(\"logging\").INFO)):'
if old1 in src:
    src = src.replace(old1, new1)
    changed = True
old2 = '    gym.logger.set_level(gym_level)'
new2 = '''    if hasattr(gym.logger, 'set_level'):
        gym.logger.set_level(gym_level)
    elif hasattr(gym.logger, 'min_level'):
        gym.logger.min_level = gym_level'''
if old2 in src:
    src = src.replace(old2, new2)
    changed = True
if changed:
    logger_py.write_text(src)
    print('      Patched logger.py')
else:
    print('      logger.py already patched or unchanged')

# Patch 3: np.infty removed in NumPy 2.0 — replace with np.inf across all files
rl_base = site / 'rl_agents'
count = 0
for py in rl_base.rglob('*.py'):
    src = py.read_text()
    if 'np.infty' in src:
        py.write_text(src.replace('np.infty', 'np.inf'))
        count += 1
if count:
    print(f'      Patched np.infty -> np.inf in {count} file(s)')
else:
    print('      np.infty already patched or not found')

# Patch 4: graphics.py — agent.env needs .unwrapped to reach through gymnasium wrappers
gfx_py = site / 'rl_agents' / 'agents' / 'deep_q_network' / 'graphics.py'
src = gfx_py.read_text()
changed = False
for old, new in [
    ('agent.env.observation_type', 'agent.env.unwrapped.observation_type'),
    ('agent.env.vehicle', 'agent.env.unwrapped.vehicle'),
    ('agent.env.road', 'agent.env.unwrapped.road'),
]:
    if old in src and new not in src:
        src = src.replace(old, new)
        changed = True
if changed:
    gfx_py.write_text(src)
    print('      Patched graphics.py')
else:
    print('      graphics.py already patched or unchanged')
"

# Verify
echo ""
echo "Verifying imports..."
python3 -c "
from rl_agents.agents.common.factory import load_environment, load_agent
from rl_agents.trainer.evaluation import Evaluation
print('All rl-agents imports OK')
"

echo ""
echo "=== Experiment 6 Setup Complete ==="
echo ""
echo "Usage:"
echo "  python experiments/6_Social_attention_with_rl_agents/train_rl_agents.py"
