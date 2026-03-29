#!/bin/bash
# 进入 TOFU 目录
cd TOFU

# 设置端口
MASTER_PORT=28765

# 只使用 GPU 0
CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 \
  --master_port=${MASTER_PORT} \
  finetune.py \
  --config-name=my_finetune.yaml \
  model_family=tinyllama \
  split=full \
  batch_size=1 \
  gradient_accumulation_steps=1 \
  lr=2e-5