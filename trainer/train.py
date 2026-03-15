import os
import sys

# Use relative path to import wuziqi_gym
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from wuziqi_gym.env import WuziqiEnv
from trainer.config import TrainConfig


class ActionMaskEnvWrapper(gym.Env):
    """Wrapper that provides action masks to stable-baselines3.

    This wrapper uses the environment's legal_actions_mask() method
    to filter out invalid actions during training using MaskablePPO.

    Usage:
        from stable_baselines3 import MaskablePPO
        from stable_baselines3.common.maskable_policies import MaskableMlpPolicy

        env = ActionMaskEnvWrapper(WuziqiEnv())
        model = MaskablePPO(MaskableMlpPolicy, env, verbose=1)
        model.learn(total_timesteps=100000)
    """

    def __init__(self, env: WuziqiEnv):
        super().__init__()
        self.env = env
        self.action_space = spaces.Discrete(225)
        self.observation_space = env.observation_space

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info['action_mask'] = self.env.legal_actions_mask()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['action_mask'] = self.env.legal_actions_mask()
        return obs, reward, terminated, truncated, info

    def legal_actions_mask(self):
        return self.env.legal_actions_mask()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


def train(config: TrainConfig, use_action_mask: bool = True):
    """训练主函数

    Args:
        config: 训练配置
        use_action_mask: 是否使用动作掩码过滤非法动作。启用后可以避免
                        智能体选择无效动作并获得-0.1惩罚，提高学习效率。

    Note:
        要完全启用动作掩码，需要安装stable-baselines3的掩码支持:
        - 使用 MaskablePPO 和 MaskableMlpPolicy
        - 当前环境已实现legal_actions_mask()方法
    """
    # 创建环境，使用动作掩码包装器
    base_env = WuziqiEnv()
    if use_action_mask:
        env = ActionMaskEnvWrapper(base_env)
    else:
        env = base_env

    # 创建模型
    from trainer.config import get_model_config
    if config.model_kwargs is None:
        config.model_kwargs = get_model_config(config.model_size)
    model_kwargs = config.model_kwargs.copy() if config.model_kwargs else {}
    policy = model_kwargs.pop('policy', 'MlpPolicy')

    # 尝试使用MaskablePPO以支持动作掩码
    try:
        from stable_baselines3 import MaskablePPO
        from stable_baselines3.common.maskable_policies import MaskableMlpPolicy

        if use_action_mask and config.algorithm == "PPO":
            # 使用MaskablePPO以支持动作掩码
            if policy == 'MlpPolicy':
                policy = MaskableMlpPolicy
            model = MaskablePPO(policy, env, verbose=1, **model_kwargs)
        elif config.algorithm == "PPO":
            model = PPO(policy, env, verbose=1, **model_kwargs)
        elif config.algorithm == "A2C":
            model = A2C(policy, env, verbose=1, **model_kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")
    except ImportError:
        # 如果MaskablePPO不可用，使用普通PPO
        if config.algorithm == "PPO":
            model = PPO(policy, env, verbose=1, **model_kwargs)
        elif config.algorithm == "A2C":
            model = A2C(policy, env, verbose=1, **model_kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")

    # 设置日志目录
    os.makedirs(config.log_dir, exist_ok=True)

    # 训练循环，支持保存频率
    total_timesteps = config.total_timesteps
    save_freq = config.save_freq
    max_steps = config.max_steps

    import time

    # 分步训练以支持保存频率
    if save_freq > 0 and save_freq < total_timesteps:
        num_saves = total_timesteps // save_freq
        remaining = total_timesteps % save_freq
        start_time = time.time()

        for i in range(num_saves):
            step_start = time.time()
            model.learn(total_timesteps=save_freq, reset_num_timesteps=False)
            step_time = time.time() - step_start
            elapsed = time.time() - start_time
            progress = (i + 1) * save_freq
            eta = (elapsed / progress) * (total_timesteps - progress) if progress > 0 else 0

            # 保存检查点
            os.makedirs(config.save_path, exist_ok=True)
            model.save(os.path.join(config.save_path, f"{config.algorithm}_wuziqi_step_{i+1}"))

            print(f"\n[训练] 进度: {progress}/{total_timesteps} ({100*progress/total_timesteps:.1f}%) | "
                  f"本轮用时: {step_time:.1f}s | 已用时: {elapsed:.1f}s | 预计剩余: {eta:.1f}s")

        if remaining > 0:
            model.learn(total_timesteps=remaining, reset_num_timesteps=False)
    else:
        # 一次性训练
        start_time = time.time()
        model.learn(total_timesteps=total_timesteps)
        total_time = time.time() - start_time
        print(f"\n[训练完成] 总用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")

    # 最终保存
    os.makedirs(config.save_path, exist_ok=True)
    model.save(os.path.join(config.save_path, f"{config.algorithm}_wuziqi"))

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='训练五子棋AI')
    parser.add_argument('--model-size', type=str, default='medium',
                        choices=['small', 'medium', 'large'],
                        help='模型大小: small(快), medium, large(强)')
    parser.add_argument('--timesteps', type=int, default=10000000,
                        help='总训练步数 (默认1000万)')
    args = parser.parse_args()

    config = TrainConfig(
        algorithm="PPO",
        model_size=args.model_size,
        total_timesteps=args.timesteps,
    )
    train(config)
