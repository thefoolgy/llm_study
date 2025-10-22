from __future__ import annotations
import os
import argparse
import math
import json
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from torch import nn
from bpe import Tokenizer, train_bpe
from layer import TransformerLM
from loss import CrossEntropy
from optimizer import AdamW
from utils import (
    get_batch,
    gradient_clipping,
    cosine_annealing_scheduler,
    save_checkpoint,
    load_checkpoint,
)


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train a Transformer Language Model with BPE tokenization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data file")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    parser.add_argument("--use_memmap", action="store_true",
                        help="Use memory-mapped files for large datasets")
    
    # BPE Tokenizer arguments
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="BPE vocabulary size")
    parser.add_argument("--vocab_file", type=str, default=None,
                        help="Path to pre-trained vocab file (if available)")
    parser.add_argument("--merges_file", type=str, default=None,
                        help="Path to pre-trained merges file (if available)")
    parser.add_argument("--special_tokens", type=str, nargs="+", 
                        default=["<|endoftext|>"],
                        help="Special tokens for BPE")
    
    # Model architecture arguments
    parser.add_argument("--d_model", type=int, default=256,
                        help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1024,
                        help="Feed-forward dimension")
    parser.add_argument("--context_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--rope_theta", type=float, default=10000.0,
                        help="RoPE theta parameter")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--max_iters", type=int, default=20000,
                        help="Maximum training iterations")
    parser.add_argument("--eval_interval", type=int, default=1000,
                        help="Evaluate every N iterations")
    parser.add_argument("--eval_iters", type=int, default=10,
                        help="Number of batches for evaluation")
    parser.add_argument("--save_interval", type=int, default=5000,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Log training metrics every N iterations")
    
    # Optimizer arguments
    parser.add_argument("--max_lr", type=float, default=3e-4,
                        help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="Minimum learning rate")
    parser.add_argument("--warmup_iters", type=int, default=500,
                        help="Number of warmup iterations")
    parser.add_argument("--cosine_iters", type=int, default=10000,
                        help="Number of cosine annealing iterations")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay coefficient")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Adam beta2")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--experiment_name", type=str, default="transformer_lm",
                        help="Name for this experiment")
    
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="transformer-lm",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (username or team)")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu", "mps"],
                        help="Device to use for training")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Data type for model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """Determine the device to use for training."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_str]


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_tokenizer(args) -> Tokenizer:
    """
    Prepare BPE tokenizer - either load from files or train new one.
    """
    print("\n" + "="*50)
    print("TOKENIZER PREPARATION")
    print("="*50)
    
    if args.vocab_file and args.merges_file:
        print(f"Loading pre-trained tokenizer from:")
        print(f"  Vocab: {args.vocab_file}")
        print(f"  Merges: {args.merges_file}")
        tokenizer = Tokenizer.from_files(
            args.vocab_file,
            args.merges_file,
            special_tokens=args.special_tokens
        )
    else:
        print(f"Training new BPE tokenizer on: {args.data_path}")
        print(f"Target vocab size: {args.vocab_size}")
        print(f"Special tokens: {args.special_tokens}")
        
        vocab, merges = train_bpe(
            input_path=args.data_path,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens
        )
        tokenizer = Tokenizer(vocab, merges, special_tokens=args.special_tokens)
        
        # Save the trained tokenizer
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        vocab_path = os.path.join(args.checkpoint_dir, "vocab.json")
        merges_path = os.path.join(args.checkpoint_dir, "merges.txt")
        
        # Save vocab
        with open(vocab_path, 'w') as f:
            vocab_serializable = {k: v.decode('latin1') for k, v in vocab.items()}
            json.dump(vocab_serializable, f)
        
        # Save merges
        with open(merges_path, 'w') as f:
            for pair in merges:
                f.write(f"{pair[0].decode('utf-8')} {pair[1].decode('utf-8')}\n")
        
        print(f"Saved tokenizer to {args.checkpoint_dir}")
    
    actual_vocab_size = len(tokenizer.vocab)
    print(f"Tokenizer ready! Vocabulary size: {actual_vocab_size}")
    return tokenizer


def prepare_data(args, tokenizer: Tokenizer):
    """
    Load and tokenize data, optionally using memory mapping for large files.
    Returns train and validation data as numpy arrays or memmaps.
    """
    print("\n" + "="*50)
    print("DATA PREPARATION")
    print("="*50)
    
    # Check if we have pre-tokenized data
    tokenized_path = args.data_path + ".tokenized.npy"
    
    if os.path.exists(tokenized_path):
        print(f"Loading pre-tokenized data from: {tokenized_path}")
        if args.use_memmap:
            data = np.memmap(tokenized_path, dtype=np.int32, mode='r')
        else:
            data = np.load(tokenized_path)
    else:
        print(f"Tokenizing data from: {args.data_path}")
        print("This may take a while for large files...")
        
        # Read and tokenize
        with open(args.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Text length: {len(text):,} characters")
        tokens = tokenizer.encode(text)
        data = np.array(tokens, dtype=np.int32)
        
        # Save tokenized data
        print(f"Saving tokenized data to: {tokenized_path}")
        np.save(tokenized_path, data)
        
        # If using memmap, reload as memmap
        if args.use_memmap:
            data = np.memmap(tokenized_path, dtype=np.int32, mode='r')
    
    print(f"Total tokens: {len(data):,}")
    
    # Split into train and validation
    split_idx = int(len(data) * (1 - args.val_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    
    return train_data, val_data


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    dataset: np.ndarray,
    context_length: int,
    batch_size: int,
    eval_iters: int,
    device: str,
    criterion: nn.Module
) -> dict[str, float]:
    """
    Estimate loss and perplexity on a dataset.
    """
    model.eval()
    losses = []
    
    for _ in range(eval_iters):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    
    avg_loss = np.mean(losses)
    perplexity = math.exp(avg_loss)
    
    model.train()
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity
    }


def train(args):
    """Main training loop."""
    
    # Set up device and seed
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    set_seed(args.seed)
    
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Seed: {args.seed}")
    
    # Initialize W&B if requested
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.experiment_name,
                config=vars(args)
            )
            print("Weights & Biases logging enabled")
        except ImportError:
            print("WARNING: wandb not installed, disabling W&B logging")
            args.use_wandb = False
    
    # Prepare tokenizer
    tokenizer = prepare_tokenizer(args)
    vocab_size = len(tokenizer.vocab)
    
    # Prepare data
    train_data, val_data = prepare_data(args, tokenizer)
    
    # Initialize model
    print("\n" + "="*50)
    print("MODEL INITIALIZATION")
    print("="*50)
    
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=dtype,
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Model size: {n_params * dtype.itemsize / 1024**2:.2f} MB")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    # Initialize loss function
    criterion = CrossEntropy()
    
    # Set up checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"{args.experiment_name}_latest.pt"
    )
    
    # Resume from checkpoint if requested
    start_iter = 0
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resuming from iteration {start_iter}")
    
    # Training loop
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    print(f"Total iterations: {args.max_iters}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    print(f"Tokens per iteration: {args.batch_size * args.context_length:,}")
    print("="*50 + "\n")
    
    best_val_loss = float('inf')
    
    for iteration in range(start_iter, args.max_iters):
        # Learning rate schedule
        lr = cosine_annealing_scheduler(
            iteration,
            args.max_lr,
            args.min_lr,
            args.warmup_iters,
            args.cosine_iters
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # Get batch
        x, y = get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            device
        )
        
        # Forward pass
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        gradient_clipping(model.parameters(), args.grad_clip_norm)
        
        # Optimizer step
        optimizer.step()
        
        # Logging
        if iteration % args.log_interval == 0:
            print(
                f"Iter {iteration:6d}/{args.max_iters} | "
                f"loss {loss.item():.4f} | "
                f"lr {lr:.6f}"
            )
            
            if args.use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/perplexity": math.exp(loss.item()),
                    "train/learning_rate": lr,
                    "iteration": iteration,
                })
        
        # Evaluation
        if iteration % args.eval_interval == 0 and iteration > 0:
            print(f"\n{'='*50}")
            print(f"EVALUATION AT ITERATION {iteration}")
            print('='*50)
            
            val_metrics = estimate_loss(
                model,
                val_data,
                args.context_length,
                args.batch_size,
                args.eval_iters,
                device,
                criterion
            )
            
            print(f"Validation loss: {val_metrics['loss']:.4f}")
            print(f"Validation perplexity: {val_metrics['perplexity']:.2f}")
            print('='*50 + '\n')
            
            if args.use_wandb:
                wandb.log({
                    "val/loss": val_metrics['loss'],
                    "val/perplexity": val_metrics['perplexity'],
                    "iteration": iteration,
                })
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_path = os.path.join(
                    args.checkpoint_dir,
                    f"{args.experiment_name}_best.pt"
                )
                save_checkpoint(model, optimizer, iteration, best_path)
                print(f"New best model saved! Val loss: {best_val_loss:.4f}\n")
        
        # Save checkpoint
        if iteration % args.save_interval == 0 and iteration > 0:
            print(f"Saving checkpoint at iteration {iteration}...")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            
            # Also save a numbered checkpoint
            numbered_path = os.path.join(
                args.checkpoint_dir,
                f"{args.experiment_name}_iter_{iteration}.pt"
            )
            save_checkpoint(model, optimizer, iteration, numbered_path)
            print(f"Checkpoint saved to {numbered_path}\n")
    
    # Final evaluation and save
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    
    final_metrics = estimate_loss(
        model,
        val_data,
        args.context_length,
        args.batch_size,
        args.eval_iters,
        device,
        criterion
    )
    
    print(f"Final validation loss: {final_metrics['loss']:.4f}")
    print(f"Final validation perplexity: {final_metrics['perplexity']:.2f}")
    
    # Save final model
    final_path = os.path.join(
        args.checkpoint_dir,
        f"{args.experiment_name}_final.pt"
    )
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"Final model saved to {final_path}")
    
    if args.use_wandb:
        wandb.log({
            "final/val_loss": final_metrics['loss'],
            "final/val_perplexity": final_metrics['perplexity'],
        })
        wandb.finish()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    args = parse_args()
    train(args)