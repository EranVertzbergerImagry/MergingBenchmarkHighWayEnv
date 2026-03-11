# High Level Depiction
 The idea is to explore what capabilities we can reach with reinforcement learning.
## 1. base line 
running the environment with a random action as policy.
To run the environment with a random policy, use the following command:

```bash
python experiments/1_no_policy/eval_random_policy.py --config experiments/1_no_policy/default_config.json
```
crash percentage: 50-60 %
arrive percentage: 50-40%

video example:

[random_policy_example.mp4](https://github.com/EranVertzbergerImagry/MergingBenchmarkHighWayEnv/blob/modified_env_and_RL_algos_implementation/ExperimentDocumentation/random_policy_example.mp4)

<video src="random_policy_example.mp4" controls width="600"></video>

## 2. experimenting with off-the shelf models in the gymnasium models. 

The environment was kept constant in terms of observation and action.

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


## 3. Experiment with DQN + transformer atchitecture 
based on [Social Attention](http://arxiv.org/abs/1911.12250):
This approach was implemented using the gymnasium library in experiment 5.
experiment 5: crash-21%, arrive-72%. we do see that the agent learns to stop and yield and choose the correct moment to enter the junction:
video example:

[social_attn_dqn_example.mp4](https://github.com/EranVertzbergerImagry/MergingBenchmarkHighWayEnv/blob/modified_env_and_RL_algos_implementation/ExperimentDocumentation/social_attn_dqn_example.mp4)

<video src="social_attn_dqn_example.mp4" controls width="600"></video>

## 4. experiment 6: using [rl-agents](https://github.com/eleurent/rl-agents) 
results based on a 1000 episodes evaluation:
crash-12%, arrive-80%. The visualization shows the attention calculated for each agent and the probability for each decision.:
video example:

[rl_agents_ego_attn_example.mp4](https://github.com/EranVertzbergerImagry/MergingBenchmarkHighWayEnv/blob/modified_env_and_RL_algos_implementation/ExperimentDocumentation/rl_agents_ego_attn_example.mp4)

<video src="rl_agents_ego_attn_example.mp4" controls width="600"></video>

when crashes occure, in all examined cases, the cause is other agents crashing in to the EGO
video example:

[rl_agents_ego_attn_crash_example.mp4](https://github.com/EranVertzbergerImagry/MergingBenchmarkHighWayEnv/blob/modified_env_and_RL_algos_implementation/ExperimentDocumentation/rl_agents_ego_attn_crash_example.mp4)

<video src="rl_agents_ego_attn_crash_example.mp4" controls width="600"></video>

training was performed only on the left turn scenario, it seems agent learns to generelize to straight and right turns also.

right turn example:

[rl_agents_ego_attn_right_turn_example.mp4](https://github.com/EranVertzbergerImagry/MergingBenchmarkHighWayEnv/blob/modified_env_and_RL_algos_implementation/ExperimentDocumentation/rl_agents_ego_attn_right_turn_example.mp4)

<video src="rl_agents_ego_attn_right_turn_example.mp4" controls width="600"></video>

straight example:

[rl_agents_ego_attn_straight_example.mp4](https://github.com/EranVertzbergerImagry/MergingBenchmarkHighWayEnv/blob/modified_env_and_RL_algos_implementation/ExperimentDocumentation/rl_agents_ego_attn_straight_example.mp4)

<video src="rl_agents_ego_attn_straight_example.mp4" controls width="600"></video>

## experiment 7: trained rl_agend with random destination and ego based observation enabling evaluation with random initial position.
The idea is to make the model more general. If agent learns to reach all destinations from all initial locations, we could then experiment with a multi agent environment. 
results: 

**Episode Summary**

| steps | reward | crashed | arrived |
|-------|--------|---------|---------|
| 1000  | 6.47   | 7%      | 63%     |

**Destination Results**

| destination | episodes | arrived% | crashed% | stall% |
|-------------|---------|----------|----------|--------|
| o1          | 348     | 49%      | 8%       | 43%    |
| o2          | 317     | 51%      | 7%       | 43%    |
| o3          | 335     | 90%      | 6%       | 4%     |

examples:

[experiment_7_random_destination_and_spawn.mp4] (https://github.com/EranVertzbergerImagry/MergingBenchmarkHighWayEnv/blob/modified_env_and_RL_algos_implementation/ExperimentDocumentation/experiment_7_random_destination_and_spawn.mp4)
<video src="experiment_7_random_destination_and_spawn.mp4" controls width="600"></video>

[experiment_7_random_destination_and_spawn_2.mp4] (https://github.com/EranVertzbergerImagry/MergingBenchmarkHighWayEnv/blob/modified_env_and_RL_algos_implementation/ExperimentDocumentation/experiment_7_random_destination_and_spawn_2.mp4)
<video src="experiment_7_random_destination_and_spawn_2.mp4" controls width="600"></video>

## Experiment 8: multi agent setup
In this example, we deploy a model trained in expereiment 7 on multiple agents.
 
[multi_agent_using_model_trained_on_single_agent_setup.mp4] (https://github.com/EranVertzbergerImagry/MergingBenchmarkHighWayEnv/blob/modified_env_and_RL_algos_implementation/ExperimentDocumentation/multi_agent_using_model_trained_on_single_agent_setup.mp4)

<video src="multi_agent_using_model_trained_on_single_agent_setup.mp4" controls width="600"></video>

## experiment 8: multi agent training.
The model trained in 7 is good. but still there are some crashes caused by the EGO. using the visualization of the attention I can inspect that in some cases
a close vehicle is not being attended or attended to late. My hypothesis is that interaction with the IDM from the environment is problematic because in some cases, crashes occure because an IDM crashes to the EGO. My initiative is to train using many EGO (controlled) agents. My first attempt (lets call it exp0) was to train experiment 8 with 8 agents. The sult is a stall behavior, EGO stops to avoide collision but never get to the destination. See results in experiments/8_multi_agent/data/runs/2026-03-10_21-28-15_multi_agent_8_controlled_ego_centric. you n see the evaluation and also that reward converges to zero. My next test (now progressing call it exp1) is training with one controlled agent to verify experiment 7 is reproduced and theres no bug (see 
experiments/8_multi_agent/data/runs/2026-03-11_14-45-38_multi_agent_1_controlled_vehicle). I can already see reward at 3-4. So maybe it was I have some oughts and ideas (assuming there's no bug): maybe the result in exp0 was caused by too much traffic (10 iitial IDMs + 0.6 spawn probability + 8 more controlled). So because in most cases the intersection gets ocked the training collapses to the "stall mode". 1. I'm thinking to train with only controlled (10 initial + 0.6 spawn probability like in experiment 7). 2. start training with an initial model hieved in 7.
