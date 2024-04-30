import time
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from typing import Callable

if __name__ == '__main__':
    env_id = "LunarLander-v2"
    num_cpu = 16
    n_timesteps = 5e6
    n_experiments = 5
    path = 'data/dqn/lunar_lander/sb3'

    set_random_seed(0)
    
    with open(path + '_eval.txt', 'w') as f:
        pass
    
    for i in range(n_experiments):
        env = make_vec_env(env_id, seed=i, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

        model = DQN("MlpPolicy", env, verbose=1,
                # learning_rate=6.3e-4,
                learning_rate=5e-4,
                buffer_size=50000,
                learning_starts=100,
                # batch_size=128,
                batch_size=64,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=250,
                exploration_fraction=0.12,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.1,
                max_grad_norm=10,
                device='cuda',
                tensorboard_log=path,
                seed=i
                )
        start_time = time.time()
        model.learn(total_timesteps=n_timesteps)
        duration = time.time() - start_time
        # model.save(path + '_' + str(i) +  '.pth')
        
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
        with open(path + '_eval.txt', 'a') as f:
            print(str(mean_reward) + '\t' + str(std_reward) + '\t' + str(duration), file=f)
        env.close()
