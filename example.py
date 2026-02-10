"""
Highway-env Example Script
A simple demonstration of the highway driving environment.
"""
import gymnasium as gym
import highway_env

def main():
    # Create the highway environment
    # Available environments: 'highway-v0', 'merge-v0', 'roundabout-v0',
    # 'parking-v0', 'intersection-v0', 'racetrack-v0'
    env = gym.make('highway-v0', render_mode='rgb_array')

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print("Actions: 0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER")

    total_reward = 0
    done = False
    steps = 0

    while not done and steps < 100:
        # Random action - replace with your own policy
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    print(f"\nEpisode finished after {steps} steps")
    print(f"Total reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
