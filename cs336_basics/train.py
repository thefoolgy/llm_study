from __future__ import annotations
import os
import math
import torch
import numpy as np
from torch import nn

from cs336_basics.bpe import get_tokenizer
from cs336_basics.layer import TransformerLM
from cs336_basics.loss import CrossEntropy
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import (
    get_batch,
    gradient_clipping,
    cosine_annealing_scheduler,
    save_checkpoint,
    load_checkpoint,
)

# -------------------------------
# Hyperparameters
# -------------------------------
DATA_PATH = "./data/tiny_shakespeare.txt"
CHECKPOINT_PATH = "./checkpoints/model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
CONTEXT_LENGTH = 128
D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 8
D_FF = 1024
ROPE_THETA = 10000.0

MAX_LR = 3e-4
MIN_LR = 1e-5
WARMUP_ITERS = 500
COSINE_ITERS = 10000
MAX_ITERS = 20000
GRAD_CLIP_NORM = 1.0
EVAL_INTERVAL = 1000
SAVE_INTERVAL = 5000
PRINT_INTERVAL = 100

# -------------------------------
# Dataset loading
# -------------------------------
def load_dataset():
    """Load and encode the dataset."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}

    data = np.array([stoi[ch] for ch in text], dtype=np.int64)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    vocab_size = len(vocab)
    return train_data, val_data, vocab_size


# -------------------------------
# Evaluation helper
# -------------------------------
@torch.no_grad()
def estimate_loss(model, dataset, context_length, batch_size):
    """Compute average loss on validation set."""
    model.eval()
    losses = []
    for _ in range(10):
        x, y = get_batch(dataset, batch_size, context_length, DEVICE)
        logits = model(x)
        loss = CrossEntropy()(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return np.mean(losses)


# -------------------------------
# Main training function
# -------------------------------
def train():
    print(f"Training on {DEVICE}...")

    # Load data
    train_data, val_data, vocab_size = load_dataset()

    # Initialize model and optimizer
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=CONTEXT_LENGTH,
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=ROPE_THETA,
        device=DEVICE,
        dtype=torch.float32,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=MAX_LR)
    criterion = CrossEntropy()

    # Resume if checkpoint exists
    start_iter = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        start_iter = load_checkpoint(CHECKPOINT_PATH, model, optimizer)

    print("Starting training loop...")

    for it in range(start_iter, MAX_ITERS):
        # Adjust learning rate (cosine schedule)
        lr = cosine_annealing_scheduler(it, MAX_LR, MIN_LR, WARMUP_ITERS, COSINE_ITERS)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get a batch
        x, y = get_batch(train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)

        # Forward pass
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        gradient_clipping(model.parameters(), GRAD_CLIP_NORM)

        optimizer.step()

        # Logging
        if it % PRINT_INTERVAL == 0:
            print(f"Iter {it:6d} | loss {loss.item():.4f} | lr {lr:.6f}")

        # Evaluation
        if it % EVAL_INTERVAL == 0 and it > 0:
            val_loss = estimate_loss(model, val_data, CONTEXT_LENGTH, BATCH_SIZE)
            print(f"[Eval] Iter {it:6d} | val_loss {val_loss:.4f}")

        # Save checkpoint
        if it % SAVE_INTERVAL == 0 and it > 0:
            print(f"Saving checkpoint at iteration {it}...")
            save_checkpoint(model, optimizer, it, CHECKPOINT_PATH)

    print("Training complete.")
    save_checkpoint(model, optimizer, MAX_ITERS, CHECKPOINT_PATH)
    print(f"Final model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
