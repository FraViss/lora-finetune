# LoRA Fine-Tuning on Tiny Shakespeare

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red)

## Overview

This project fine-tunes **GPT-2 small** (117M parameters) on the **Tiny Shakespeare** corpus using **LoRA** (Low-Rank Adaptation). Only **0.12%** of the model is trained—**147,456** parameters out of **124M**—while the frozen base weights stay fixed. On the validation set, perplexity drops from **10,157** to **241**, a **42×** improvement.

## What is LoRA?

LoRA keeps the pretrained base model frozen and learns a low-rank update to selected weight matrices. Instead of fine-tuning the full weight matrix $W$, the adaptation is decomposed as:

$$
\Delta W = B \cdot A
$$

where $A \in \mathbb{R}^{r \times d_\text{in}}$ and $B \in \mathbb{R}^{d_\text{out} \times r}$ with rank $r \ll \min(d_\text{in}, d_\text{out})$. Only **A** and **B** are trained; at inference time the update can be merged back into $W$.

## Project structure

```
lora-finetune/
├── main.py                      # CLI entry point (train / evaluate)
├── demo.ipynb                   # Jupyter demo with plots and comparisons
├── requirements.txt             # Pinned Python dependencies (PyTorch CPU)
├── data/
│   └── dataset.py               # Tiny Shakespeare download, split, and token blocks
├── model/
│   ├── __init__.py              # Package exports (LoRALayer)
│   ├── lora_layer.py            # LoRA wrapper around frozen nn.Linear layers
│   └── base_model.py            # GPT-2 with LoRA on attention c_proj projections
├── train/
│   ├── config.py                # TrainingConfig dataclass (hyperparameters)
│   └── trainer.py               # Training loop, validation, checkpoints, plotting
├── evaluate/
│   └── evaluator.py             # Perplexity and generation comparison (base vs fine-tuned)
└── outputs/
    ├── plot_history.py          # Standalone script to plot epoch-level metrics
    └── training_history.png     # Saved training curves (train/val loss, perplexity)
```

Checkpoints are written to `checkpoints/` during training (e.g. `checkpoint_epoch_3.pt`).

## Installation

```bash
git clone https://github.com/FraViss/lora-finetune
cd lora-finetune
uv venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements.txt
```

## Usage

Train LoRA adapters on Tiny Shakespeare:

```bash
python main.py train
```

Compare base GPT-2 against the fine-tuned checkpoint:

```bash
python main.py evaluate
```

## Results

| Epoch | Train loss | Val loss | Val perplexity |
|------:|-----------:|---------:|---------------:|
| 1 | 6.3400 | 5.6966 | 297.86 |
| 2 | 5.8498 | 5.5484 | 256.83 |
| 3 | 5.7372 | 5.4841 | 240.83 |

Validation perplexity improved from **~10,157** (base GPT-2) to **~241** (fine-tuned)—roughly a **42×** reduction—after three epochs of LoRA training.

## Key design choices

- **GPT-2 small** — Small enough to train on CPU in reasonable time, yet large enough to show meaningful adaptation with LoRA.
- **Tiny Shakespeare** — A classic, compact language-modeling benchmark with distinctive style; easy to download and split for train/validation.
- **LoRA on `c_proj` only** — Adapters are attached to each attention block’s output projection, targeting where attention outputs are mixed before the next sublayer.
- **Asymmetric initialization** — `lora_A` is Gaussian (std 0.02) and `lora_B` is zero, so the LoRA path starts inactive and training gradually introduces the low-rank update.

## License

MIT
