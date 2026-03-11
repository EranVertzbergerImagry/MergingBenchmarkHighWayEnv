# Experiment 8 — Multi-Agent Intersection

## Overview

Building on Experiment 7's ego-centric generalization, this experiment introduces **multi-agent training and evaluation** at an intersection. Four controlled vehicles navigate the intersection simultaneously, each using the same learned policy independently. The goal is to test whether a policy trained in a multi-agent setting — where the ego must account for other intelligent agents, not just rule-based NPC traffic — produces more robust driving behavior.

## Background

Experiment 7 trained a single agent with ego-centric observations and random destinations, achieving heading-invariant generalization across all intersection entries. However, the training environment contained only one controlled vehicle — all other traffic was scripted NPC behavior. In real intersections, multiple decision-making agents interact simultaneously, creating more complex and less predictable dynamics.

## Design Decisions

### 4 Controlled Vehicles

We use 4 controlled vehicles (one potential agent per intersection arm). All share the same DQN policy weights — the network processes each agent's observation independently and returns an action. This means more diverse training transitions per episode without changing the architecture.

### Staggered Random Spawning

A naive approach of spawning all 4 agents at t=0 at fixed locations would allow the policy to memorize the initial configuration rather than learn general behavior. To prevent this:

- **Random spawn times** — Each agent is assigned a random spawn time spread across the episode. Agents that haven't spawned yet are kept idle (action forced to IDLE, zero reward contribution).
- **Random entry points** — Each agent spawns at a randomly selected intersection entry.
- **Collision-safe spawn** — Before placing a new agent, we verify the entry lane is clear. If the lane is occupied, spawning is deferred or skipped.
- **Post-arrival/crash handling** — Agents that have arrived at their destination or crashed become idle for the remainder of the episode.

This approach maintains the fixed-size observation/action tuples that highway-env's `MultiAgentObservation`/`MultiAgentAction` require (tuple of 4 at every step), while achieving the randomness needed for genuine generalization.

### Ego-Centric Observations (from Experiment 7)

Each agent's observation is independently rotated into its own heading frame. The 9-feature vector `[presence, x, y, vx, vy, cos_h, sin_h, cos_d, sin_d]` is transformed so that from each agent's perspective: `x=0, y=0, cos_h=1, sin_h=0`. This makes the shared policy heading-invariant — each agent sees the world from its own frame of reference.

### Cooperative Reward

highway-env's multi-agent intersection uses an averaged reward across all active agents. This is a cooperative setting — the policy is incentivized to learn behavior that works well for all agents simultaneously, not just one.

### Per-Agent Logging

Evaluation logs per-agent per-episode statistics:
- Each agent's destination, arrival status, crash status
- Per-destination breakdown across all agents
- Aggregate episode-level metrics

This allows us to analyze whether certain entry/destination combinations are harder in multi-agent traffic.

## Two Evaluation Phases

### Phase 1: Zero-Shot Transfer

Evaluate the Experiment 7 checkpoint (trained single-agent) directly in the multi-agent scene with no retraining. This tests whether ego-centric generalization transfers to multi-agent dynamics.

### Phase 2: Multi-Agent Training

Train from scratch in the multi-agent setting. Compare results against Phase 1 to measure whether multi-agent training improves robustness.

## Usage

```bash
# Phase 1: Evaluate exp7 checkpoint in multi-agent scene
python train.py --evaluation-only --recover-from <exp7_checkpoint.tar>

# Phase 2: Train in multi-agent setting
python train.py

# Evaluate with random spawn
python train.py --evaluation-only --recover-from <checkpoint.tar> --random-spawn
```
