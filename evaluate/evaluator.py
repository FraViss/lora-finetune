"""Evaluate and compare base GPT-2 against a LoRA fine-tuned checkpoint."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import ShakespeareDataset
from model.base_model import LoRAGPT2
from train.config import TrainingConfig


def compute_val_metrics(
    lora_gpt2: LoRAGPT2,
    data_loader: DataLoader,
    device: str,
) -> tuple[float, float]:
    """Compute mean loss and perplexity for a model over a data loader.

    Args:
        lora_gpt2: Model wrapper to evaluate.
        data_loader: Batches of ``input_ids``/``labels`` to evaluate on.
        device: Device to move batches to before the forward pass.

    Returns:
        A tuple of ``(mean_loss, perplexity)``.
    """
    lora_gpt2.model.eval()
    losses: list[float] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = lora_gpt2.forward(input_ids=input_ids, labels=labels)
            losses.append(outputs.loss.item())

    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)


class Evaluator:
    """Compare perplexity and text generation between base and fine-tuned GPT-2."""

    def __init__(self, config: TrainingConfig, checkpoint_path: str) -> None:
        """Load a fine-tuned checkpoint and prepare validation data.

        Args:
            config: Training hyperparameters and runtime settings.
            checkpoint_path: Path to a saved model ``state_dict`` checkpoint.
        """
        self.config = config
        self.device = config.device

        self.finetuned_model = LoRAGPT2(rank=config.rank, alpha=config.alpha)
        self.finetuned_model.model.load_state_dict(
            torch.load(checkpoint_path, map_location=config.device, weights_only=True)
        )
        self.finetuned_model.model.to(self.device)
        self.finetuned_model.model.eval()

        self.base_model = LoRAGPT2(rank=config.rank, alpha=config.alpha)
        self.base_model.model.to(self.device)
        self.base_model.model.eval()

        val_dataset = ShakespeareDataset(
            self.finetuned_model.tokenizer,
            block_size=config.block_size,
            split="val",
        )
        self.val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=config.batch_size,
        )

    def _compute_perplexity(self, lora_gpt2: LoRAGPT2) -> float:
        """Compute mean validation perplexity for a :class:`LoRAGPT2` model.

        Args:
            lora_gpt2: Model wrapper to evaluate on the validation set.

        Returns:
            Perplexity computed as ``exp(mean_cross_entropy_loss)``.
        """
        _, perplexity = compute_val_metrics(lora_gpt2, self.val_loader, self.device)
        return perplexity

    def compare_perplexity(self) -> dict[str, float]:
        """Compare validation perplexity for base and fine-tuned models.

        Returns:
            Dictionary with keys ``"base"`` and ``"finetuned"`` mapping to
            perplexity values.
        """
        base_perplexity = self._compute_perplexity(self.base_model)
        finetuned_perplexity = self._compute_perplexity(self.finetuned_model)

        print("Perplexity comparison (validation set)")
        print("=" * 44)
        print(f"{'Model':<12} {'Perplexity':>12}")
        print("-" * 44)
        print(f"{'Base':<12} {base_perplexity:>12.4f}")
        print(f"{'Finetuned':<12} {finetuned_perplexity:>12.4f}")
        print("=" * 44)

        return {"base": base_perplexity, "finetuned": finetuned_perplexity}

    def compare_generation(
        self,
        prompts: list[str],
        max_new_tokens: int = 120,
    ) -> None:
        """Print side-by-side generations for base and fine-tuned models.

        Args:
            prompts: Input prompts to continue with each model.
            max_new_tokens: Maximum number of new tokens to generate per prompt.
        """
        for index, prompt in enumerate(prompts, start=1):
            base_text = self.base_model.generate(prompt, max_new_tokens=max_new_tokens)
            finetuned_text = self.finetuned_model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
            )

            print(f"\n{'=' * 60}")
            print(f"Prompt {index}")
            print(f"{'=' * 60}")
            print(f"Prompt:\n{prompt}\n")
            print("--- Base model ---")
            print(base_text)
            print("\n--- Fine-tuned model ---")
            print(finetuned_text)


if __name__ == "__main__":
    config = TrainingConfig()
    evaluator = Evaluator(
        config,
        checkpoint_path=f"{config.checkpoint_dir}/checkpoint_epoch_{config.num_epochs}.pt",
    )
    evaluator.compare_perplexity()
    evaluator.compare_generation(
        prompts=[
            "To be or not to be",
            "Shall I compare thee",
            "What light through yonder window",
        ],
    )
