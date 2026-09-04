"""Training loop for LoRA fine-tuning on Tiny Shakespeare."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ShakespeareDataset
from evaluate.evaluator import compute_val_metrics
from model.base_model import LoRAGPT2
from train.config import TrainingConfig


class Trainer:
    """Fine-tune a :class:`LoRAGPT2` model on the Tiny Shakespeare corpus."""

    def __init__(self, config: TrainingConfig) -> None:
        """Set up the model, data loaders, optimizer, and metric history.

        Args:
            config: Training hyperparameters and runtime settings.
        """
        self.config = config
        self.device = config.device

        torch.manual_seed(config.seed)

        self.lora_model = LoRAGPT2(rank=config.rank, alpha=config.alpha)
        self.lora_model.model.to(self.device)

        train_dataset = ShakespeareDataset(
            self.lora_model.tokenizer,
            block_size=config.block_size,
            split="train",
        )
        val_dataset = ShakespeareDataset(
            self.lora_model.tokenizer,
            block_size=config.block_size,
            split="val",
        )

        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=config.batch_size,
        )
        self.val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=config.batch_size,
        )

        trainable_params = [
            p for p in self.lora_model.model.parameters() if p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
        )

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_perplexity": [],
        }

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the language-modeling loss for a single batch.

        Args:
            batch: Dictionary containing ``input_ids`` and ``labels`` tensors.

        Returns:
            Scalar loss tensor.
        """
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        outputs = self.lora_model.forward(input_ids=input_ids, labels=labels)
        return outputs.loss

    def _validate(self) -> tuple[float, float]:
        """Evaluate the model on the validation set.

        Returns:
            A tuple of ``(mean_val_loss, val_perplexity)``.
        """
        return compute_val_metrics(self.lora_model, self.val_loader, self.device)

    def _save_checkpoint(self, epoch: int) -> None:
        """Save the model state dict for the given epoch.

        Args:
            epoch: Epoch number used in the checkpoint filename.
        """
        checkpoint_path = (
            Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save(self.lora_model.model.state_dict(), checkpoint_path)

    def train(
        self,
        start_epoch: int = 1,
        checkpoint: Path | str | None = None,
    ) -> None:
        """Run the full training loop with periodic validation and checkpointing.

        Args:
            start_epoch: First epoch to run (1-based). Use with ``checkpoint`` to resume.
            checkpoint: Optional path to a saved ``state_dict`` to load before training.
        """
        print(self.config)

        if checkpoint is not None:
            state = torch.load(
                checkpoint,
                map_location=self.device,
                weights_only=True,
            )
            self.lora_model.model.load_state_dict(state)
            print(f"Resumed weights from {checkpoint}")

        global_step = 0
        latest_val_loss = float("nan")
        latest_val_perplexity = float("nan")

        for epoch in range(start_epoch, self.config.num_epochs + 1):
            self.lora_model.model.train()
            train_losses: list[float] = []
            validated_this_epoch = False

            progress = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{self.config.num_epochs}",
            )
            for batch in progress:
                self.optimizer.zero_grad()
                loss = self._compute_loss(batch)
                loss.backward()
                self.optimizer.step()

                loss_value = loss.item()
                train_losses.append(loss_value)
                global_step += 1
                progress.set_postfix(loss=f"{loss_value:.4f}")

                if global_step % self.config.val_every_n_steps == 0:
                    mean_train_loss = sum(train_losses) / len(train_losses)
                    val_loss, val_perplexity = self._validate()
                    latest_val_loss = val_loss
                    latest_val_perplexity = val_perplexity
                    validated_this_epoch = True

                    self.history["train_loss"].append(mean_train_loss)
                    self.history["val_loss"].append(val_loss)
                    self.history["val_perplexity"].append(val_perplexity)

                    self.lora_model.model.train()

            if not validated_this_epoch:
                latest_val_loss, latest_val_perplexity = self._validate()
            mean_train_loss = sum(train_losses) / len(train_losses)

            if epoch % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)

            print(
                f"Epoch {epoch} summary | "
                f"train_loss={mean_train_loss:.4f} | "
                f"val_loss={latest_val_loss:.4f} | "
                f"val_perplexity={latest_val_perplexity:.4f}"
            )

    def plot_history(self) -> None:
        """Plot training and validation metrics and save the figure to disk."""
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].plot(self.history["train_loss"])
        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Validation step")
        axes[0].set_ylabel("Loss")

        axes[1].plot(self.history["val_loss"])
        axes[1].set_title("Validation Loss")
        axes[1].set_xlabel("Validation step")
        axes[1].set_ylabel("Loss")

        axes[2].plot(self.history["val_perplexity"])
        axes[2].set_title("Validation Perplexity")
        axes[2].set_xlabel("Validation step")
        axes[2].set_ylabel("Perplexity")

        fig.tight_layout()
        fig.savefig(output_dir / "training_history.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    config = TrainingConfig()
    trainer = Trainer(config)

    checkpoint_dir = Path(config.checkpoint_dir)
    last_epoch = config.num_epochs
    resume_ckpt = checkpoint_dir / f"checkpoint_epoch_{last_epoch - 1}.pt"
    final_ckpt = checkpoint_dir / f"checkpoint_epoch_{last_epoch}.pt"

    if (
        last_epoch > 1
        and resume_ckpt.exists()
        and not final_ckpt.exists()
    ):
        trainer.train(start_epoch=last_epoch, checkpoint=resume_ckpt)
    else:
        trainer.train()

    trainer.plot_history()
