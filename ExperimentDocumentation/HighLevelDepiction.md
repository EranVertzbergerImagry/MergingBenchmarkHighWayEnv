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

### experiment to test the effect of policy frequency 
I have noticed that 1Hz policy is problematic because not responsive enough. The base model I'm trying to improve is trained in experiment 7  with parameters 1 Hz policy frequency, gamma = 0.95,high_speed_reward = 1. I tested the  model (evaluation_only) 5Hz at the same environment. indeed crashes reduced and overall performance looked better. Now train a new model in 5Hz. I changed
the parameters to: gamma = 0.99. The gamma was chosen so that the discounted return will propagate the same new_gamma = (0.95 ^ 13) ^ [1/(13*5)] = ~0.99. 

(trained in 1 Hz)
Arrived: 67/100 (67%)  Crashed: 12/100 (12%)  Avg reward: 6.57

  
  Dest   Episodes  Arrived  Crashed    Stall
  --------------------------------------
  o1           41      46%      17%      37%
  o2           29      62%      14%      24%
  o3           30      100%       0%       0%

Also changed the high speed reward from 1 -> to 0.2 because in 5Hz agent collects times 5 more speed rewards relative to the same terminal reward. changed to 5 hz

Arrived: 56/100 (56%)  Crashed: 9/100 (9%)  Stall: 35/100 (35%)  Avg reward: 6.15

  Dest      Agents  Arrived  Crashed    Stall
  -----------------------------------------
  o1            26       4%      19%      77%
  o2            31      55%      10%      35%
  o3            43      88%       2%       9%

stalling! 
incresed HS reward to 0.5
  Arrived: 67/100 (67%)  Crashed: 32/100 (32%)  Stall: 1/100 (1%)  Avg reward: 16.55

  Dest      Agents  Arrived  Crashed    Stall
  -----------------------------------------
  o1            30      57%      40%       3%
  o2            39      54%      46%       0%
  o3            31      94%       6%       0%
reduced to 0.35 

Arrived: 30/100 (30%)  Crashed: 23/100 (23%)  Stall: 47/100 (47%)  Avg reward: 9.22

  Dest      Agents  Arrived  Crashed    Stall
  -----------------------------------------
  o1            40      20%      30%      50%
  o2            36      14%      28%      58%
  o3            24      71%       4%      25%

reduced to 0.275
Arrived: 63/100 (63%)  Crashed: 26/100 (26%)  Stall: 11/100 (11%)  Avg reward: 10.14

  Dest      Agents  Arrived  Crashed    Stall
  -----------------------------------------
  o1            39      49%      38%      13%
  o2            35      54%      29%      17%
  o3            26      96%       4%       0%
  High Speed Reward (HSR)=1, Arival Rewrd (AR) = 5, Crash Penalty (CP) = -25
  Arrived: 40/100 (40%)  Crashed: 10/100 (10%)  Stall: 50/100 (50%)  Avg reward: 21.59

  Dest      Agents  Arrived  Crashed    Stall
  -----------------------------------------
  o1            37       0%       8%      92%
  o2            31      26%      23%      52%
  o3            32     100%       0%       0%

  **I can't find a good set of parameters for training in 5Hz. I will continue to explore possibilities with the current setup:** 
  **High Speed Reward (HSR)=1, Arival Rewrd (AR) = 1, Crash Penalty (CP) = -5, policy frequency = 1Hz, gamma =0.95**
  **at evaluation increase policy frequency to 5 Hz**
## experiment 8: multi agent training.
The model trained in 7 is good. but still there are some crashes caused by the EGO. using the visualization of the attention I can inspect that in some cases where a close vehicle is not being attended or attended to late. My hypothesis is that interaction with the IDM from the environment is problematic because in some cases, crashes occure because an IDM crashes to the EGO. 
My initiative is to train using many EGO (controlled) agents. My first attempt (lets call it exp0) was to train experiment 8 with 8 agents. The result is a stall behavior, EGO stops to avoide collision but never get to the destination. See results in experiments/8_multi_agent/data/runs/2026-03-10_21-28-15_multi_agent_8_controlled_ego_centric. you can see the evaluation and also that reward converges to zero. My next test (call it exp1) is training with one controlled agent to verify experiment 7 is reproduced and theres no bug (see 
experiments/8_multi_agent/data/runs/2026-03-11_14-45-38_multi_agent_1_controlled_vehicle). I can already see reward at 3-4. I have some thoughts and ideas (assuming there's no bug): maybe the result in exp0 was caused by too much traffic (10 iitial IDMs + 0.6 spawn probability + 8 more controlled). So because in most cases the intersection gets blocked the training collapses to the "stall mode". 
1. I'm thinking to train with only controlled (10 initial + 0.6 spawn probability like in experiment 7). 
2. start training with an initial model achieved in 7.
* train a model with a different exploration profile with tau raised 15K->500K
  tested on the 4 agent benchmark with a 5 hz policy frequency
  results:
  SUMMARY,100 episodes,394 agents spawned,avg_reward=22.76,arrived=78%,crashed=12%,stall=9%
  increase in crash rate but maybe we need to train longer with the larger tau because maybe it would keep improving. 
* try to fine tune the model acheived in 7 using multi agent setup meaning initial_temp=final_temp = 0.05 and see if performance    improves. we start with the best model by now: experiments/7_Social_attention_generalization/data/runs/2026-03-10_11-18-09_ego_attention_2h_random_dest_ego_centric_reference/checkpoint-final.tar
  **I stopped this experiment in the middle as I didnt see improvement and testing in the middle showed model has degraded. 
* test fine tune with a smaller learning rate lr = 5e-4 -> 5e-5. If this doesnt work I can try to see if I can use an established policy freezed as a replacement to IDM. **didn't work training collapsed** 

**conclusion training multiple agents is unstable**
## experiment 9: improved enviromental drivers
basically replace the the IDM with a model achieved in exp7 (Model0). then train a new model using it as the environment drivers (ED). 
opening point model achieved in exp7 (experiments/7_Social_attention_generalization/data/runs/2026-03-10_11-18-09_ego_attention_2h_random_dest_ego_centric_reference) deployed in the ED. same model evaluated against them. 
* baseline results M0 VS M0 1 Hz: 
Arrived: 55/100 (55%)  Crashed: 12/100 (12%)  Stall: 33/100 (33%)  Avg reward: 5.55

  Dest      Agents  Arrived  Crashed    Stall
  -----------------------------------------
  o1            30      17%      20%      63%
  o2            31      58%       6%      35%
  o3            39      82%      10%       8%

Arr/Crsh = 55/12 = 4.58
* baseline results M0 VS M0 5 Hz:
Arrived: 56/100 (56%)  Crashed: 15/100 (15%)  Stall: 29/100 (29%)  Avg reward: 27.16

  Dest      Agents  Arrived  Crashed    Stall
  -----------------------------------------
  o1            41      24%      22%      54%
  o2            32      62%      19%      19%
  o3            27      96%       0%       4%

Arr/Crsh = 56/15 = 3.73

* model (M1) fine tuned for 4000 episodes from M0 Vs M0 :

  Arrived: 82/100 (82%)  Crashed: 13/100 (13%)  Stall: 5/100 (5%)  Avg reward: 7.37

  Dest      Agents  Arrived  Crashed    Stall
  -----------------------------------------
  o1            28      75%      21%       4%
  o2            38      79%      16%       5%
  o3            34      91%       3%       6%

Arr/Crsh = 56/15 = 6.3
* model (M2) fine tuned for 4000 episodes from M1 Vs M1 :
Arrived: 53/100 (53%)  Crashed: 40/100 (40%)  Stall: 7/100 (7%)  Avg reward: 4.45

  Dest      Agents  Arrived  Crashed    Stall
  -----------------------------------------
  o1            39      36%      51%      13%
  o2            29      41%      52%       7%
  o3            32      84%      16%       0%
Arr/Crsh = 53/40 = 1.325 

* evaluation M1 Vs M1
  Arrived: 59/100 (59%)  Crashed: 40/100 (40%)  Stall: 1/100 (1%)  Avg reward: 4.79

  Dest      Agents  Arrived  Crashed    Stall
  -----------------------------------------
  o1            32      38%      59%       3%
  o2            27      41%      59%       0%
  o3            41      88%      12%       0%
Arr/Crsh = 59/50 = 1.18
* evaluation M1 Vs M0
  