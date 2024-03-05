import gymnasium as gym
import numpy as np
import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F

random.seed(0)
np.random.seed(0)
th.manual_seed(0)

class SimpleFCNet(nn.Module):
    def __init__(self):
        super(SimpleFCNet, self).__init__()
        self.fc1 = nn.Linear(3, 3)

    def forward(self, x):
        return self.fc1(x)

net = SimpleFCNet()

for name, param in net.named_parameters():
    if param.requires_grad:
        print(name, param.data)
    

# env = gym.make_vec("LunarLander-v2", num_envs=1, vectorization_mode='sync', render_mode='human')
# env = gym.make("LunarLander-v2", render_mode='human')
# env.action_space.seed(0)
# env.reset(seed=0)
 
# for i in range(1000):
    # action = env.action_space.sample()
    # # obs, reward, terminated, truncated, info = env.step(action)
    # if terminated or truncated:
        # env.reset()
    # if terminated.any() or truncated.any():
    #     obs = info['final_observation']
    #     mask = info['_final_observation']
    #     print(obs[mask])
    #     print(obs[mask][0, 0])
    #     break
