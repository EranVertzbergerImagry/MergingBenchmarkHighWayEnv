# High Level Depiction
 The idea is to explore what capabilities we can reach with reinforcement learning.
1. base line is running the environment with a random action as policy.
To run the environment with a random policy, use the following command:

```bash
python experiments/1_no_policy/eval_random_policy.py --config experiments/1_no_policy/default_config.json
```
crash percentage: 50-60 %
arrive percentage: 50-40%

video example:

<video src="https://raw.githubusercontent.com/EranVertzbergerImagry/MergingBenchmarkHighWayEnv/modified_env_and_RL_algos_implementation/ExperimentDocumentation/random_policy_example.mp4" controls width="600"></video>

2. We start off experimenting with off-the shelf models in the gymnasium models. The environment was kept constant in terms of observation and action.

```python
    "observation": {
      "type": "Kinematics",
      "vehicles_count": 15,
      "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
      "absolute": true,
      "flatten": true
    },
    "action": {
      "type": "DiscreteMetaAction",
      "target_speeds": [0, 4.5, 9]
    }
```
I experimented with DQN, SAC, and PPO (experiments 2-4). tried different Net structures, target speeds, wights, episode lengths.
A bash script  in exch experiment (for example experiments/2_vanilla_dqn/run_reward_grid.sh) runs a grid search to find the best hyper weights for the reward function. Basically the resulting policy was about 50% crash-arrive. the agent learns a simple constant speed approach collecting as much high speed points risking crash. When setting high speed reward to zero, the agent stops to avoide crash but never arrives at target.


3. Experiment with DQN + transformer atchitecture [Social Attention](http://arxiv.org/abs/1911.12250):
This approach was implemented using the gymnasium library in experiment 5 and using [rl-agents](https://github.com/eleurent/rl-agents) in experiment 6.
experiment 5: crash-21%, arrive-72%. we do see that the agent learns to stop and yield and choose the correct moment to enter the junction:
video example:

<video src="https://raw.githubusercontent.com/EranVertzbergerImagry/MergingBenchmarkHighWayEnv/modified_env_and_RL_algos_implementation/ExperimentDocumentation/social_attn_dqn_example.mp4" controls width="600"></video>

experiment 6: crash-12%, arrive-80%. The visualization shows the attention calculated for each agent and the probability for each decision.:
video example:

<video src="https://raw.githubusercontent.com/EranVertzbergerImagry/MergingBenchmarkHighWayEnv/modified_env_and_RL_algos_implementation/ExperimentDocumentation/rl_agents_ego_attn_example.mp4" controls width="600"></video>

when crashes occure, in all examined cases, the cause is other agents crashing in to the EGO
video example:

<video src="https://raw.githubusercontent.com/EranVertzbergerImagry/MergingBenchmarkHighWayEnv/modified_env_and_RL_algos_implementation/ExperimentDocumentation/rl_agents_ego_attn_crash_example.mp4" controls width="600"></video>

training was performed only on the left turn scenario, it seems agent learns to generelize to straight and right turns also.

right turn example:

<video src="https://raw.githubusercontent.com/EranVertzbergerImagry/MergingBenchmarkHighWayEnv/modified_env_and_RL_algos_implementation/ExperimentDocumentation/rl_agents_ego_attn_right_turn_example.mp4" controls width="600"></video>

straight example:

<video src="https://raw.githubusercontent.com/EranVertzbergerImagry/MergingBenchmarkHighWayEnv/modified_env_and_RL_algos_implementation/ExperimentDocumentation/rl_agents_ego_attn_straight_example.mp4" controls width="600"></video>
