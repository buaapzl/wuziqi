import os
import sys
sys.path.insert(0, '/Users/rr2017/code/wuziqi')

from stable_baselines3 import PPO, A2C
from wuziqi_gym.env import WuziqiEnv
from trainer.config import TrainConfig


def train(config: TrainConfig):
    """训练主函数"""
    # 创建环境
    env = WuziqiEnv()

    # 创建模型
    model_kwargs = config.model_kwargs.copy() if config.model_kwargs else {}
    policy = model_kwargs.pop('policy', 'MlpPolicy')

    if config.algorithm == "PPO":
        model = PPO(policy, env, verbose=1, **model_kwargs)
    elif config.algorithm == "A2C":
        model = A2C(policy, env, verbose=1, **model_kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    # 训练
    model.learn(total_timesteps=config.total_timesteps)

    # 保存
    os.makedirs(config.save_path, exist_ok=True)
    model.save(os.path.join(config.save_path, f"{config.algorithm}_wuziqi"))

    return model


if __name__ == '__main__':
    config = TrainConfig(
        algorithm="PPO",
        total_timesteps=100000,
    )
    train(config)
