from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    algorithm: str = "PPO"
    total_timesteps: int = 100000000
    max_steps: int = 200  # Max steps per episode (used by environment wrapper)
    env_make_kwargs: Optional[dict] = None
    model_kwargs: Optional[dict] = None
    save_freq: int = 1000000  # Save checkpoint every N steps (0 to disable)
    eval_freq: int = 5000000  # Evaluation frequency (placeholder for future use)
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
