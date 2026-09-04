"""GPT-2 language model with LoRA adapters on attention output projections."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.pytorch_utils import Conv1D

from model.lora_layer import LoRALayer


def _conv1d_to_linear(conv: Conv1D) -> nn.Linear:
    """Convert a GPT-2 :class:`Conv1D` layer to an equivalent :class:`nn.Linear`."""
    linear = nn.Linear(conv.nx, conv.nf, bias=True)
    linear.weight.data.copy_(conv.weight.T)
    linear.bias.data.copy_(conv.bias.data)
    return linear


class LoRAGPT2:
    """GPT-2 with LoRA adapters on each attention block's ``c_proj`` projection.

    All base GPT-2 weights are frozen. Only the low-rank LoRA matrices attached
    to the attention output projections are trainable.
    """

    def __init__(self, rank: int = 8, alpha: float = 16.0) -> None:
        """Load GPT-2, freeze it, and attach LoRA adapters to attention ``c_proj`` layers.

        Args:
            rank: LoRA rank for each adapted projection.
            alpha: LoRA scaling hyperparameter passed to each :class:`LoRALayer`.
        """
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        for param in self.model.parameters():
            param.requires_grad = False

        for block in self.model.transformer.h:
            original_c_proj = block.attn.c_proj
            linear_c_proj = _conv1d_to_linear(original_c_proj)
            block.attn.c_proj = LoRALayer(linear_c_proj, rank=rank, alpha=alpha)

    def trainable_parameters(self) -> int:
        """Print and return the number of trainable parameters in the model."""
        count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Trainable parameters: {count:,}")
        return count

    def frozen_parameters(self) -> int:
        """Print and return the number of frozen parameters in the model."""
        count = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        print(f"Frozen parameters: {count:,}")
        return count

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        """Run a forward pass through the underlying GPT-2 model.

        Args:
            input_ids: Token indices of shape ``(batch_size, sequence_length)``.
            attention_mask: Optional mask of shape ``(batch_size, sequence_length)``.
            labels: Optional labels for language-modeling loss computation.

        Returns:
            The :class:`transformers.modeling_outputs.CausalLMOutputWithCrossAttentions`
            returned by the underlying model.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate text from a prompt using sampling-based decoding.

        Args:
            prompt: Input text to continue.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            The decoded generated text, including the original prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    lora_gpt2 = LoRAGPT2()
    lora_gpt2.trainable_parameters()
    lora_gpt2.frozen_parameters()

    generated = lora_gpt2.generate("To be or not to be")
    print(f"\nGenerated text:\n{generated}")
