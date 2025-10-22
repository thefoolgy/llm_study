"""
Inference script for text generation using trained Transformer LM.

Usage:
    python generate.py --checkpoint ./checkpoints/model.pt --prompt "Once upon a time"
"""

import argparse
import torch
import numpy as np

from bpe import Tokenizer
from layer import TransformerLM


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with trained model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--vocab_file", type=str, required=True,
                        help="Path to vocab.json")
    parser.add_argument("--merges_file", type=str, required=True,
                        help="Path to merges.txt")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt to start generation")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling (1.0 = disabled)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    # Model architecture (needed if not saved in checkpoint metadata)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """Determine device to use."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Filter a distribution of logits using top-k filtering."""
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    return logits


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Filter a distribution of logits using nucleus (top-p) filtering."""
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
    return logits


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: str,
) -> str:
    """Generate text from a prompt."""
    
    model.eval()
    
    # Encode prompt
    if prompt:
        tokens = tokenizer.encode(prompt)
    else:
        tokens = []
    
    # Convert to tensor
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Prompt tokens: {len(tokens)}")
    print(f"Generating {max_length} tokens...")
    print("-" * 70)
    
    generated_tokens = tokens.copy()
    
    for i in range(max_length):
        # Get logits for next token
        # Use only the last context_length tokens if sequence is too long
        context_length = model.context_length
        if input_ids.shape[1] > context_length:
            input_ids_context = input_ids[:, -context_length:]
        else:
            input_ids_context = input_ids
        
        logits = model(input_ids_context)
        
        # Get logits for the last position
        next_token_logits = logits[0, -1, :] / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            next_token_logits = top_k_filtering(next_token_logits, top_k)
        
        # Apply top-p filtering
        if top_p < 1.0:
            next_token_logits = top_p_filtering(next_token_logits, top_p)
        
        # Sample from the filtered distribution
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        generated_tokens.append(next_token.item())
        
        # Print progress every 10 tokens
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{max_length} tokens...", end='\r')
    
    print(f"Generated {max_length}/{max_length} tokens... Done!")
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def load_model(checkpoint_path: str, args, device: str) -> TransformerLM:
    """Load model from checkpoint."""
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load tokenizer to get vocab size
    tokenizer = Tokenizer.from_files(
        args.vocab_file,
        args.merges_file,
        special_tokens=["<|endoftext|>"]
    )
    vocab_size = len(tokenizer.vocab)
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32,
    )
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Loaded from iteration: {checkpoint.get('iteration', 'unknown')}")
    
    return model, tokenizer


def main():
    args = parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.checkpoint, args, device)
    
    # Generate text
    print("\n" + "="*70)
    print("GENERATION")
    print("="*70)
    
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )
    
    print("\n" + "="*70)
    print("GENERATED TEXT")
    print("="*70)
    print(generated_text)
    print("="*70)


if __name__ == "__main__":
    main()