"""Low-Rank Adaptation (LoRA) layer for fine-tuning frozen linear layers."""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """Wraps a frozen :class:`nn.Linear` with a low-rank trainable update.

    The adapted output is the original linear map plus a scaled low-rank term:

        y = W @ x + b + (alpha / rank) * (x @ A^T @ B^T)

    where ``lora_A`` has shape ``(rank, in_features)`` and ``lora_B`` has shape
    ``(out_features, rank)``.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        """Build a LoRA adapter around an existing linear layer.

        Args:
            linear: Pre-existing linear layer whose weights are frozen.
            rank: Rank of the low-rank decomposition.
            alpha: Scaling hyperparameter; effective scale is ``alpha / rank``.
        """
        super().__init__()

        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        for param in self.linear.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the frozen linear map plus the scaled LoRA update.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        original_output = self.linear(x)
        lora_output = x @ self.lora_A.T @ self.lora_B.T
        return original_output + self.scaling * lora_output

    def merge_weights(self) -> nn.Linear:
        """Fuse LoRA weights into a standalone linear layer for inference.

        Returns:
            A new :class:`nn.Linear` with weight
            ``W + scaling * (lora_B @ lora_A)`` and the same bias as the wrapped layer.
        """
        merged = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=self.linear.bias is not None,
        )

        with torch.no_grad():
            delta = self.scaling * (self.lora_B @ self.lora_A)
            merged.weight.copy_(self.linear.weight + delta)
            if self.linear.bias is not None:
                merged.bias.copy_(self.linear.bias)

        return merged

    @property
    def trainable_parameters(self) -> int:
        """Return the number of trainable parameters in the LoRA matrices."""
        return sum(
            p.numel() for p in (self.lora_A, self.lora_B) if p.requires_grad
        )
