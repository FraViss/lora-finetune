"""Training configuration for LoRA fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """Hyperparameters and runtime settings for LoRA fine-tuning."""

    rank: int = 8
    alpha: float = 16.0
    block_size: int = 128
    batch_size: int = 8
    learning_rate: float = 3e-4
    num_epochs: int = 3
    val_every_n_steps: int = 50
    save_every_n_epochs: int = 1
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        """Ensure the checkpoint directory exists."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def __str__(self) -> str:
        """Return a multi-line summary of all configuration fields."""
        return (
            "TrainingConfig(\n"
            f"  rank={self.rank},\n"
            f"  alpha={self.alpha},\n"
            f"  block_size={self.block_size},\n"
            f"  batch_size={self.batch_size},\n"
            f"  learning_rate={self.learning_rate},\n"
            f"  num_epochs={self.num_epochs},\n"
            f"  val_every_n_steps={self.val_every_n_steps},\n"
            f"  save_every_n_epochs={self.save_every_n_epochs},\n"
            f"  checkpoint_dir={self.checkpoint_dir!r},\n"
            f"  device={self.device!r},\n"
            f"  seed={self.seed},\n"
            ")"
        )


if __name__ == "__main__":
    config = TrainingConfig()
    print(config)
