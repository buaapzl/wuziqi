from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    algorithm: str = "PPO"
    total_timesteps: int = 100_000
    max_steps: int = 200
    env_make_kwargs: Optional[dict] = None
    model_kwargs: Optional[dict] = None
    save_freq: int = 10000
    eval_freq: int = 50000
    save_path: str = "./models"
    log_dir: str = "./logs"


PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
}
