from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# 小模型配置 - 训练快，效果一般
SMALL_PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
}

# 中等模型配置
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
}

# 大模型配置 - 用于训练更强的AI (使用CNN)
LARGE_PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 4096,  # 更大的batch
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "policy": "CnnPolicy",  # 使用CNN处理图像输入
}


# 模型配置映射
MODEL_CONFIGS = {
    "small": SMALL_PPO_CONFIG,
    "medium": PPO_CONFIG,
    "large": LARGE_PPO_CONFIG,
}


@dataclass
class TrainConfig:
    algorithm: str = "PPO"
    total_timesteps: int = 10000000
    max_steps: int = 200  # Max steps per episode (used by environment wrapper)
    env_make_kwargs: Optional[dict] = None
    model_size: str = "large"  # small, medium, large
    model_kwargs: Optional[Dict[str, Any]] = None
    save_freq: int = 1000000  # Save checkpoint every N steps (0 to disable)
    eval_freq: int = 5000000  # Evaluation frequency (placeholder for future use)
    save_path: str = "./models"
    log_dir: str = "./logs"


def get_model_config(model_size: str = "medium") -> Dict[str, Any]:
    """获取模型配置"""
    return MODEL_CONFIGS.get(model_size, PPO_CONFIG).copy()
