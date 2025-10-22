#!/bin/bash
# Example training commands for the Transformer Language Model

# ==============================================================================
# Basic Training Example (Small Model)
# ==============================================================================
uv run python train.py \
    --data_path ../data/owt_train.txt \
    --experiment_name "small_model" \
    --d_model 256 \
    --num_layers 4 \
    --num_heads 8 \
    --d_ff 1024 \
    --batch_size 32 \
    --context_length 128 \
    --max_iters 20000 \
    --checkpoint_dir ./checkpoints/small

# ==============================================================================
# Medium Model with Memory Mapping (for large datasets)
# ==============================================================================
# echo "Example 2: Medium model with memory-mapped data"
# python train_improved.py \
#     --data_path ./data/large_corpus.txt \
#     --use_memmap \
#     --experiment_name "medium_model" \
#     --d_model 512 \
#     --num_layers 8 \
#     --num_heads 8 \
#     --d_ff 2048 \
#     --batch_size 64 \
#     --context_length 256 \
#     --max_iters 100000 \
#     --checkpoint_dir ./checkpoints/medium

# ==============================================================================
# Large Model with W&B Logging
# ==============================================================================
# echo "Example 3: Large model with Weights & Biases logging"
# python train_improved.py \
#     --data_path ./data/owt_train.txt \
#     --use_wandb \
#     --wandb_project "my-transformer-lm" \
#     --experiment_name "large_model_v1" \
#     --d_model 768 \
#     --num_layers 12 \
#     --num_heads 12 \
#     --d_ff 3072 \
#     --batch_size 32 \
#     --context_length 512 \
#     --max_iters 200000 \
#     --checkpoint_dir ./checkpoints/large

# ==============================================================================
# Resume Training from Checkpoint
# ==============================================================================
# echo "Example 4: Resume training from checkpoint"
# python train_improved.py \
#     --data_path ./data/owt_train.txt \
#     --resume_from ./checkpoints/medium/medium_model_iter_50000.pt \
#     --experiment_name "medium_model" \
#     --checkpoint_dir ./checkpoints/medium \
#     --max_iters 100000

# ==============================================================================
# Custom Learning Rate Schedule
# ==============================================================================
# echo "Example 5: Custom learning rate schedule"
# python train_improved.py \
#     --data_path ./data/owt_train.txt \
#     --experiment_name "custom_lr" \
#     --max_lr 6e-4 \
#     --min_lr 6e-5 \
#     --warmup_iters 2000 \
#     --cosine_iters 50000 \
#     --batch_size 64 \
#     --max_iters 100000 \
#     --checkpoint_dir ./checkpoints/custom_lr

# ==============================================================================
# Use Pre-trained Tokenizer
# ==============================================================================
# echo "Example 6: Use pre-trained BPE tokenizer"
# python train_improved.py \
#     --data_path ./data/owt_train.txt \
#     --vocab_file ./checkpoints/vocab.json \
#     --merges_file ./checkpoints/merges.txt \
#     --experiment_name "pretrained_tokenizer" \
#     --checkpoint_dir ./checkpoints/pretrained_tok

# ==============================================================================
# GPU Training with Mixed Precision (Float16)
# ==============================================================================
# echo "Example 7: GPU training with float16"
# python train_improved.py \
#     --data_path ./data/owt_train.txt \
#     --device cuda \
#     --dtype float16 \
#     --experiment_name "fp16_model" \
#     --batch_size 128 \
#     --checkpoint_dir ./checkpoints/fp16

# ==============================================================================
# Hyperparameter Sweep Configuration
# ==============================================================================
# echo "Example 8: Training for hyperparameter sweep"
# # Vary learning rates
# for lr in 1e-4 3e-4 6e-4; do
#     python train_improved.py \
#         --data_path ./data/owt_train.txt \
#         --experiment_name "sweep_lr_${lr}" \
#         --max_lr $lr \
#         --checkpoint_dir ./checkpoints/sweep_lr
# done

# Vary model sizes
# for d_model in 128 256 512; do
#     python train_improved.py \
#         --data_path ./data/owt_train.txt \
#         --experiment_name "sweep_dmodel_${d_model}" \
#         --d_model $d_model \
#         --checkpoint_dir ./checkpoints/sweep_dmodel
# done