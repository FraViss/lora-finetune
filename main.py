"""Main entry point for the LoRA fine-tuning project.

Supports two modes:

- ``train``: fine-tune GPT-2 with LoRA on Tiny Shakespeare, save checkpoints,
  and plot training history.
- ``evaluate``: compare base GPT-2 against the fine-tuned checkpoint on
  validation perplexity and sample text generation.
"""

from __future__ import annotations

import argparse

from evaluate.evaluator import Evaluator
from train.config import TrainingConfig
from train.trainer import Trainer

EVALUATION_PROMPTS = [
    "To be or not to be",
    "Shall I compare thee",
    "What light through yonder window",
]


def _default_checkpoint(config: TrainingConfig) -> str:
    """Path to the checkpoint saved after the final configured epoch."""
    return f"{config.checkpoint_dir}/checkpoint_epoch_{config.num_epochs}.pt"


def run_train() -> None:
    """Run the LoRA fine-tuning loop and save a training history plot."""
    config = TrainingConfig()
    trainer = Trainer(config)
    trainer.train()
    trainer.plot_history()


def run_evaluate() -> None:
    """Compare base and fine-tuned models on perplexity and generation."""
    config = TrainingConfig()
    evaluator = Evaluator(config, checkpoint_path=_default_checkpoint(config))
    evaluator.compare_perplexity()
    evaluator.compare_generation(prompts=EVALUATION_PROMPTS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the project entry point.

    Args:
        argv: Optional argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace with a ``mode`` field set to ``"train"`` or
        ``"evaluate"``.
    """
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for GPT-2 on Tiny Shakespeare.",
    )
    parser.add_argument(
        "mode",
        choices=["train", "evaluate"],
        help='run mode: "train" to fine-tune, "evaluate" to compare models',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the selected run mode.

    Args:
        argv: Optional argument list. Defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)

    if args.mode == "train":
        run_train()
    elif args.mode == "evaluate":
        run_evaluate()


if __name__ == "__main__":
    main()
