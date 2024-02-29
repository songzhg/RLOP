import gymnasium as gym
import numpy as np

env = gym.make_vec("LunarLander-v2", num_envs=16)

env.reset()

for i in range(5000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated.any() or truncated.any():
        obs = info['final_observation']
        mask = info['_final_observation']
        print(obs[mask])
        print(obs[mask][0, 0])
        break