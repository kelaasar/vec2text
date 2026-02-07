#!/bin/bash

# Mistral Embed Inverter Training Script
# This trains the inverter model for mistral-embed (1024 dimensions)
# Based on the successful Gemini training configuration

export TMPDIR=/scratch/tmp
export VEC2TEXT_CACHE=/scratch/kelaasar/vec2text_cache

CUDA_VISIBLE_DEVICES=0,2 python vec2text/run.py \
    --per_device_train_batch_size 96 \
    --per_device_eval_batch_size 96 \
    --max_seq_length 128 \
    --model_name_or_path t5-base \
    --dataset_name msmarco \
    --embedder_model_name mistral-embed \
    --num_repeat_tokens 16 \
    --embedder_no_grad True \
    --num_train_epochs 2 \
    --max_eval_samples 500 \
    --eval_steps 20000 \
    --warmup_steps 625 \
    --bf16=True \
    --use_wandb=True \
    --learning_rate 1e-3 \
    --output_dir ./saves/mistral-embed-inverter \
    --save_steps 50000 \
    --logging_steps 50 \
    --use_frozen_embeddings_as_input True \
    --experiment inversion \
    --embedder_model_api mistral-embed
