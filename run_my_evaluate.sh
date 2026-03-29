cd TOFU
MASTER_PORT=28765
MODEL=tinyllama
LR1=2e-5
METHOD=IHL
BSZ=8
EPOCHS=5
LORA_TARGET=all
SPLIT=forget10
RANK=32
DEVICE=0

# 1. Base FT model evaluation (forget10_perturbed)
CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
  --nproc_per_node=1 \
  --master_port=${MASTER_PORT} \
  evaluate_util.py \
  --config-name=eval_everything.yaml \
  batch_size=${BSZ} \
  model_family=${MODEL} \
  split=forget10_perturbed \
  model_path=./llm_weights/ft_epoch5_lr${LR1}_${MODEL}_full_wd0.01/checkpoint-625

# 2. Unlearned model evaluation（根据你的 forget.sh checkpoint 步数调整 STEP，通常 60）
STEP=60   # 如果你改了 epochs 或 GAS，这里需对应修改

CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
  --nproc_per_node=1 \
  --master_port=${MASTER_PORT} \
  evaluate_util.py \
  --config-name=eval_everything.yaml \
  batch_size=${BSZ} \
  model_family=${MODEL} \
  split=${SPLIT}_perturbed \
  model_path=./llm_weights/ft_epoch5_lr${LR1}_${MODEL}_full_wd0.01/checkpoint-625/${METHOD}_target-${LORA_TARGET}_r-${RANK}_${LR2}_${SPLIT}_${EPOCHS}/checkpoint-${STEP} \
  base_model_path=./llm_weights/ft_epoch5_lr${LR1}_${MODEL}_full_wd0.01/checkpoint-625/${METHOD}_target-${LORA_TARGET}_r-${RANK}_${LR2}_${SPLIT}_${EPOCHS}