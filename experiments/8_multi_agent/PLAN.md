# Experiment 8 — Implementation Plan

## Files to Create/Modify

All paths relative to `experiments/8_multi_agent/`.

Copy from Experiment 7: `train.py`, `default_config.json`, `setup.sh`, and entire `src/` directory. Then modify as described below.

---

### 1. CREATE `src/multi_agent_intersection.py` — Custom multi-agent environment

Subclass `RandomSpawnIntersectionEnv` (from exp 7, also copied into `src/`).

**No `MultiAgentObservation` / `MultiAgentAction`** — we bypass highway-env's multi-agent wrappers entirely. The env itself manages the tuple packing/unpacking of observations and actions. This avoids the need for placeholder vehicles, since unborn agents simply get zero-padded observations.

**New config keys:**
- `controlled_vehicles`: 4
- `spawn_times`: `null` — when null, assign each agent a random spawn time uniformly distributed across `[0, duration * 0.6]` (first 60% of episode, leaving time to reach destination)
- `spawn_clearance`: 20 — minimum distance to existing vehicles before spawning

**State tracking (reset each episode):**
- `self._agent_spawn_times`: list of 4 random times
- `self._agent_spawned`: list of 4 booleans (vehicle placed on road)
- `self._agent_done`: list of 4 booleans (arrived or crashed)
- `self._agent_entries`: list of 4 random entry indices
- `self._agent_destinations`: list of 4 destination strings

**Override `_reset()`:**
- Call `_make_road()` and spawn NPC traffic as usual (parent `_make_vehicles` for NPCs only)
- Initialize `self.controlled_vehicles = []` (empty — no controlled vehicles yet)
- For each of the 4 agent slots, pre-assign:
  - Random spawn time from `[0, duration * 0.6]`
  - Random entry (0–3)
  - Random destination (excluding own entry)
- Set all `_agent_spawned` and `_agent_done` to False

**Override `step(action_tuple)`:**
- `action_tuple` is a tuple of 4 actions (from DQN)
- Before physics step, check each unborn agent:
  - If `self.time >= spawn_time[i]` and entry lane is clear (no vehicle within `spawn_clearance` meters):
    - Create vehicle at assigned entry with assigned destination
    - Add to `self.road.vehicles` and `self.controlled_vehicles`
    - Mark `_agent_spawned[i] = True`
  - If lane not clear, defer to next step
- For spawned & active agents: apply their action
- For unborn or done agents: ignore their action
- Run physics step
- Build per-agent observations, rewards, info
- Return `(obs_tuple, avg_reward, terminated, truncated, info)`

**Override `_observe()`:**
- For each agent slot (0–3):
  - If spawned and not done: build Kinematics observation from that vehicle's perspective (reuse the observation factory or build manually)
  - If unborn or done: return zero array of shape `(vehicles_count, 9)`
- Return as tuple of 4 arrays

**Override `_reward(action)`:**
- Average reward across active (spawned and not done) agents only
- Unborn and done agents contribute 0

**Override `_is_terminated()`:**
- All spawned agents are done (arrived or crashed)
- OR: no agents spawned yet → not terminated (wait for first spawn)

**Override observation_space / action_space:**
- `observation_space`: `spaces.Tuple` of 4 × `Box(shape=(15, 9))`
- `action_space`: `spaces.Tuple` of 4 × `Discrete(3)`

**Register as `"intersection-multi-agent-v0"`.**

---

### 2. MODIFY `src/ego_centric_wrapper.py` — Handle tuple observations

The current wrapper assumes a single numpy array observation. In multi-agent mode, the env returns a tuple of arrays.

**Changes to `observation(self, obs)`:**
```python
def observation(self, obs):
    if isinstance(obs, tuple):
        # Multi-agent: transform each agent's observation independently
        controlled = self.env.unwrapped.controlled_vehicles
        spawned = self.env.unwrapped._agent_spawned
        return tuple(
            self._transform_single(agent_obs, controlled[i])
            if spawned[i] and i < len(controlled)
            else agent_obs  # unborn: pass through zeros unchanged
            for i, agent_obs in enumerate(obs)
        )
    else:
        # Single-agent: backward compatible
        return self._transform_single(obs, self.env.unwrapped.vehicle)
```

**Extract current logic into `_transform_single(self, obs, vehicle)`:**
- Takes one observation array and the corresponding vehicle
- Injects that vehicle's `destination_direction` into row 0
- Rotates into that vehicle's heading frame
- Returns transformed array

**Update `observation_space`** property to handle `spaces.Tuple` when multi-agent.

---

### 3. MODIFY `src/training_tracker.py` — Per-agent logging

In multi-agent mode, write one row per agent per episode:
```
episode, agent_id, reward, steps, crashed, arrived, destination, spawn_entry
```

The step callback detects multi-agent by checking if the observation is a tuple. On episode end, query the env for each agent's status.

---

### 4. MODIFY `default_config.json` — Multi-agent config

```json
{
  "description": "multi_agent_4v_ego_centric",
  "train_episodes": 10000,
  "eval_episodes": 100,
  "demo_episodes": 10,
  "demo_on_train_end": true,
  "env": {
    "id": "intersection-multi-agent-v0",
    "import_module": "highway_env",
    "controlled_vehicles": 4,
    "observation": {
      "type": "Kinematics",
      "vehicles_count": 15,
      "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h", "cos_d", "sin_d"],
      "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20]},
      "absolute": true,
      "flatten": false,
      "order": "shuffled",
      "observe_intentions": false
    },
    "action": {
      "type": "DiscreteMetaAction",
      "longitudinal": true,
      "lateral": false
    },
    "destination": null,
    "duration": 13,
    "initial_vehicle_count": 10,
    "spawn_probability": 0.6,
    "collision_reward": -5,
    "high_speed_reward": 1,
    "arrived_reward": 1,
    "reward_speed_range": [7.0, 9.0],
    "normalize_reward": false,
    "spawn_entry": null,
    "spawn_clearance": 20
  },
  "agent": {
    "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
    "model": {
      "type": "EgoAttentionNetwork",
      "embedding_layer": {"type": "MultiLayerPerceptron", "layers": [64, 64], "reshape": false, "in": 9},
      "others_embedding_layer": {"type": "MultiLayerPerceptron", "layers": [64, 64], "reshape": false, "in": 9},
      "self_attention_layer": null,
      "attention_layer": {"type": "EgoAttention", "feature_size": 64, "heads": 2},
      "output_layer": {"type": "MultiLayerPerceptron", "layers": [64, 64], "reshape": false}
    },
    "gamma": 0.95,
    "n_steps": 1,
    "batch_size": 64,
    "memory_capacity": 15000,
    "target_update": 512,
    "exploration": {
      "method": "EpsilonGreedy",
      "tau": 15000,
      "temperature": 1.0,
      "final_temperature": 0.05
    }
  }
}
```

Note: observation and action types are standard single-agent types in the config. The multi-agent tuple handling is done entirely by our custom env class, not by highway-env's MultiAgent wrappers.

Agent config is identical to exp 7 — the DQN already handles tuple observations natively.

---

### 5. MODIFY `train.py` — Multi-agent evaluation and demo

**`make_env()`:** Same as exp 7 but imports the new multi-agent env registration.

**`run_episodes()`:**
- After reset, capture destination for EACH controlled vehicle (from env state)
- Results include per-agent info: agent_id, destination, arrived, crashed

**`evaluate_agent()`:**
- Per-agent per-episode CSV rows
- Summary includes per-destination breakdown across all agents

**`demo()`:** Should work as-is since rl-agents DQN handles tuples. May need minor adjustments for video recording.

**Phase 1:** Use `--recover-from <exp7_checkpoint>` — same agent architecture, DQN auto-handles tuple observations.

---

### 6. MODIFY `src/__init__.py` — Register new env

Add import for `multi_agent_intersection` to trigger `gym.register`.

---

## Implementation Order

1. Copy exp 7 files into exp 8
2. Create `src/multi_agent_intersection.py` with staggered spawn and manual tuple obs/action
3. Update `src/ego_centric_wrapper.py` for tuple observations
4. Update `default_config.json` for multi-agent
5. Update `src/__init__.py` for new env registration
6. Update `src/training_tracker.py` for per-agent logging
7. Update `train.py` for multi-agent eval/demo
8. Test Phase 1: load exp 7 checkpoint, evaluate in multi-agent
9. Test Phase 2: short training run, verify metrics

---

## Testing Plan

### Test 1: Environment creates and resets
```python
env = gym.make("intersection-multi-agent-v0", config={...})
obs, info = env.reset()
assert isinstance(obs, tuple) and len(obs) == 4
assert each obs has shape (15, 9)
```

### Test 2: Staggered spawn timing
- Reset 10 times, verify spawn times vary per agent
- Verify not all agents appear at step 0
- Verify unborn agents return zero observations

### Test 3: Collision-safe spawn
- Verify no immediate crash on spawn (distance check works)
- Verify spawn is deferred if lane is occupied

### Test 4: Ego-centric wrapper with tuples
```python
wrapped_env = EgoCentricWrapper(env)
obs, _ = wrapped_env.reset()
# Spawned agent's obs[0]: x=0, y=0, cos_h=1, sin_h=0
# Unborn agent's obs: all zeros (passed through unchanged)
```

### Test 5: Phase 1 — Zero-shot eval
- Load exp 7 checkpoint
- Run 100 episodes in multi-agent env
- Verify DQN handles tuple observations correctly
- Compare arrival/crash rates to single-agent baseline

### Test 6: Phase 2 — Short training run
```bash
python train.py --episodes 50
```

### Test 7: Per-agent logging
- Verify CSV has per-agent rows with correct destinations
- Verify summary breakdown per destination and per agent
