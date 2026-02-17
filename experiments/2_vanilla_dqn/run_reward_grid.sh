#!/usr/bin/env bash
#
# Grid search over high_speed_reward x arrived_reward for experiment 2 (Vanilla DQN).
#
# Usage:
#   bash run_reward_grid.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for hs in 0.75 0.5 0.25; do
  for ar in 1 10 25 50 100; do
    hs_label=$(echo "$hs" | tr -d '.')
    desc="hs${hs_label}_ar${ar}"

    config=$(mktemp /tmp/grid_XXXXXX.json)
    cat > "$config" <<CONF
{
  "description": "${desc}",
  "env": {
    "high_speed_reward": ${hs},
    "arrived_reward": ${ar}
  }
}
CONF

    echo ""
    echo "============================================"
    echo "  Grid run: high_speed_reward=$hs  arrived_reward=$ar"
    echo "  Description: $desc"
    echo "============================================"

    python "$SCRIPT_DIR/train_dqn_intersection.py" --config "$config"

    rm "$config"
  done
done

echo ""
echo "Grid search complete."
