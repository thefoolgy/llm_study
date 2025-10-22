"""
Utility script for analyzing training checkpoints and generating reports.

Usage:
    python analyze_training.py --checkpoint_dir ./checkpoints --experiment_name my_model
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze training results")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing checkpoints")
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Name of the experiment")
    parser.add_argument("--output_dir", type=str, default="./analysis",
                        help="Directory to save analysis results")
    return parser.parse_args()


def load_checkpoint_info(checkpoint_path: str) -> Dict:
    """Load basic information from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    info = {
        "iteration": checkpoint.get("iteration", 0),
        "path": checkpoint_path,
    }
    
    # Count parameters
    if "model_state" in checkpoint:
        n_params = sum(
            p.numel() for p in checkpoint["model_state"].values()
            if isinstance(p, torch.Tensor)
        )
        info["n_params"] = n_params
    
    return info


def analyze_checkpoints(checkpoint_dir: str, experiment_name: str) -> List[Dict]:
    """Analyze all checkpoints for an experiment."""
    checkpoint_dir = Path(checkpoint_dir)
    pattern = f"{experiment_name}_*.pt"
    
    checkpoint_files = sorted(checkpoint_dir.glob(pattern))
    
    if not checkpoint_files:
        print(f"No checkpoints found matching pattern: {pattern}")
        return []
    
    print(f"Found {len(checkpoint_files)} checkpoint(s)")
    
    checkpoints_info = []
    for ckpt_file in checkpoint_files:
        try:
            info = load_checkpoint_info(str(ckpt_file))
            info["filename"] = ckpt_file.name
            checkpoints_info.append(info)
            print(f"  ✓ {ckpt_file.name} - Iteration {info['iteration']}")
        except Exception as e:
            print(f"  ✗ {ckpt_file.name} - Error: {e}")
    
    return checkpoints_info


def load_tokenizer_info(checkpoint_dir: str) -> Dict:
    """Load tokenizer information if available."""
    vocab_path = Path(checkpoint_dir) / "vocab.json"
    merges_path = Path(checkpoint_dir) / "merges.txt"
    
    info = {}
    
    if vocab_path.exists():
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
            info["vocab_size"] = len(vocab)
            info["vocab_path"] = str(vocab_path)
    
    if merges_path.exists():
        with open(merges_path, 'r') as f:
            merges = f.readlines()
            info["n_merges"] = len([l for l in merges if l.strip() and not l.startswith('#')])
            info["merges_path"] = str(merges_path)
    
    return info


def generate_report(
    checkpoint_dir: str,
    experiment_name: str,
    output_dir: str
):
    """Generate a comprehensive training report."""
    
    print("\n" + "="*70)
    print(f"TRAINING ANALYSIS: {experiment_name}")
    print("="*70)
    
    # Analyze checkpoints
    print("\nCheckpoint Analysis:")
    print("-" * 70)
    checkpoints = analyze_checkpoints(checkpoint_dir, experiment_name)
    
    if not checkpoints:
        print("No checkpoints found. Exiting.")
        return
    
    # Sort by iteration
    checkpoints = sorted(checkpoints, key=lambda x: x["iteration"])
    
    # Print summary
    print(f"\nTotal checkpoints: {len(checkpoints)}")
    print(f"First iteration: {checkpoints[0]['iteration']}")
    print(f"Last iteration: {checkpoints[-1]['iteration']}")
    
    if "n_params" in checkpoints[0]:
        n_params = checkpoints[0]["n_params"]
        print(f"Model parameters: {n_params:,}")
        print(f"Model size (fp32): {n_params * 4 / 1024**2:.2f} MB")
    
    # Tokenizer info
    print("\nTokenizer Information:")
    print("-" * 70)
    tokenizer_info = load_tokenizer_info(checkpoint_dir)
    if tokenizer_info:
        for key, value in tokenizer_info.items():
            print(f"{key}: {value}")
    else:
        print("No tokenizer files found")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON report
    report = {
        "experiment_name": experiment_name,
        "checkpoint_dir": checkpoint_dir,
        "checkpoints": checkpoints,
        "tokenizer_info": tokenizer_info,
    }
    
    report_path = Path(output_dir) / f"{experiment_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to: {report_path}")
    
    # Checkpoint timeline
    if len(checkpoints) > 1:
        iterations = [c["iteration"] for c in checkpoints]
        
        plt.figure(figsize=(10, 4))
        plt.scatter(iterations, range(len(iterations)), alpha=0.6)
        plt.xlabel("Iteration")
        plt.ylabel("Checkpoint Index")
        plt.title(f"Checkpoint Timeline: {experiment_name}")
        plt.grid(True, alpha=0.3)
        
        timeline_path = Path(output_dir) / f"{experiment_name}_timeline.png"
        plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Timeline saved to: {timeline_path}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")


def compare_checkpoints(checkpoint_paths: List[str]):
    """Compare multiple checkpoints."""
    print("\nComparing Checkpoints:")
    print("-" * 70)
    
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location="cpu")
        iteration = ckpt.get("iteration", "unknown")
        
        # Get optimizer state info
        if "optimizer_state" in ckpt:
            opt_state = ckpt["optimizer_state"]
            if "state" in opt_state:
                n_states = len(opt_state["state"])
                print(f"{Path(path).name}:")
                print(f"  Iteration: {iteration}")
                print(f"  Optimizer states: {n_states}")
        else:
            print(f"{Path(path).name}: Iteration {iteration}")


def extract_model_config(checkpoint_path: str) -> Dict:
    """Extract model configuration from checkpoint if possible."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    config = {}
    
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        
        # Try to infer d_model from embedding weights
        if "token_emb.weight" in state_dict:
            vocab_size, d_model = state_dict["token_emb.weight"].shape
            config["vocab_size"] = vocab_size
            config["d_model"] = d_model
        
        # Count transformer blocks
        block_keys = [k for k in state_dict.keys() if k.startswith("blocks.")]
        if block_keys:
            max_block_idx = max(int(k.split(".")[1]) for k in block_keys)
            config["num_layers"] = max_block_idx + 1
        
        # Try to infer num_heads from attention weights
        for key in state_dict.keys():
            if "attn.q_proj.weight" in key:
                attn_dim = state_dict[key].shape[0]
                if "d_model" in config:
                    config["num_heads"] = config["d_model"] // (attn_dim // config["d_model"])
                break
    
    return config


if __name__ == "__main__":
    args = parse_args()
    
    try:
        generate_report(
            args.checkpoint_dir,
            args.experiment_name,
            args.output_dir
        )
        
        # Also try to extract model config
        latest_ckpt = Path(args.checkpoint_dir) / f"{args.experiment_name}_latest.pt"
        if latest_ckpt.exists():
            print("\nModel Configuration (inferred from checkpoint):")
            print("-" * 70)
            config = extract_model_config(str(latest_ckpt))
            for key, value in config.items():
                print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()