import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from typing import Callable

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    env_id = "LunarLanderContinuous-v2"
    num_cpu = 16
    n_timesteps = 1e6
    
    # env = make_vec_env(env_id, n_envs=num_cpu)
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    model = PPO("MlpPolicy", env, verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.98,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=False,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,
            device="cuda",
            tensorboard_log="python/tensorboard_logs/"
            )
    start_time = time.time()
    model.learn(total_timesteps=n_timesteps)
    duration = time.time() - start_time
        
    print(f"Took {duration:.2f}s - {n_timesteps / duration:.2f} FPS")

    
    # eval_env = gym.make(env_id, render_mode='human')
    eval_env = gym.make(env_id)
    eval_env = Monitor(eval_env, "monitor_logs")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000)
    print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")
    env.close()
    eval_env.close()
