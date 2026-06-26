"""Plot training history across all three epochs."""

from pathlib import Path

import matplotlib.pyplot as plt

train_loss = [6.3400, 5.8498, 5.7372]
val_loss = [5.6966, 5.5484, 5.4841]
val_perplexity = [297.86, 256.83, 240.83]

epochs = [1, 2, 3]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(epochs, train_loss, marker="o")
axes[0].set_title("Train Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_xticks(epochs)
axes[0].grid(True)

axes[1].plot(epochs, val_loss, marker="o")
axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_xticks(epochs)
axes[1].grid(True)

axes[2].plot(epochs, val_perplexity, marker="o")
axes[2].set_title("Validation Perplexity")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Perplexity")
axes[2].set_xticks(epochs)
axes[2].grid(True)

fig.tight_layout()
output_path = Path(__file__).parent / "training_history.png"
fig.savefig(output_path, dpi=150)
plt.close(fig)
