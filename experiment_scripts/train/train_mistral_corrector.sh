#!/bin/bash

# Mistral Embed Corrector Training Script
# This trains the corrector model for mistral-embed (1024 dimensions)
# Based on the successful Gemini corrector training configuration

export TMPDIR=/scratch/tmp
export VEC2TEXT_CACHE=/scratch/kelaasar/vec2text_cache

# First, add the inverter to aliases.py if not already done:
# "mistral_embed_msmarco__msl128__2epoch": "/scratch/kelaasar/vec2text/saves/mistral-embed-inverter"

CUDA_VISIBLE_DEVICES=0,2 python vec2text/run.py \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 3 \
    --per_device_eval_batch_size 16 \
    --max_seq_length 128 \
    --num_train_epochs 2 \
    --max_eval_samples 500 \
    --eval_steps 20000 \
    --warmup_steps 20000 \
    --learning_rate 1e-3 \
    --dataset_name msmarco \
    --embedder_model_name mistral-embed \
    --model_name_or_path t5-base \
    --experiment corrector \
    --bf16=True \
    --use_wandb=True \
    --exp_group_name mistral_embed \
    --exp_name mistral_embed \
    --output_dir ./saves/mistral-embed-corrector \
    --num_repeat_tokens 16 \
    --embedder_no_grad True \
    --save_steps 50000 \
    --logging_steps 50 \
    --use_frozen_embeddings_as_input True \
    --embedder_model_api mistral-embed \
    --corrector_load_from_pretrained_inversion mistral_embed_msmarco__msl128__2epoch
