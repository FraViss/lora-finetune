"""Dataset utilities for character-level language modeling on Tiny Shakespeare."""

from __future__ import annotations

import urllib.request

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


class ShakespeareDataset(Dataset):
    """Non-overlapping token blocks from the Tiny Shakespeare corpus.

    Each sample provides ``block_size`` input tokens and ``block_size`` label
    tokens shifted by one position for next-token prediction.
    """

    def __init__(
        self,
        tokenizer: GPT2Tokenizer,
        block_size: int = 128,
        split: str = "train",
    ) -> None:
        """Load, split, tokenize, and chunk the Tiny Shakespeare text.

        Args:
            tokenizer: GPT-2 tokenizer used to encode the corpus.
            block_size: Number of input tokens per training example.
            split: Which split to use, either ``"train"`` or ``"val"``.
        """
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.split = split

        with urllib.request.urlopen(SHAKESPEARE_URL) as response:
            text = response.read().decode("utf-8")

        split_index = int(len(text) * 0.9)
        if split == "train":
            text = text[:split_index]
        else:
            text = text[split_index:]

        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of non-overlapping blocks in the tokenized split."""
        return len(self.tokens) // (self.block_size + 1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one input/label pair for next-token language modeling.

        Args:
            index: Index of the block to retrieve.

        Returns:
            A dictionary with ``input_ids`` and ``labels``, each of shape
            ``(block_size,)`` and dtype ``torch.long``.
        """
        start = index * (self.block_size + 1)
        end = start + self.block_size + 1
        chunk = self.tokens[start:end]

        return {
            "input_ids": chunk[: self.block_size],
            "labels": chunk[1 : self.block_size + 1],
        }


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    train_dataset = ShakespeareDataset(tokenizer, split="train")
    val_dataset = ShakespeareDataset(tokenizer, split="val")

    first_sample = train_dataset[0]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"First input_ids shape: {tuple(first_sample['input_ids'].shape)}")
    print(
        "First 10 decoded tokens: "
        f"{tokenizer.decode(first_sample['input_ids'][:10].tolist())!r}"
    )
