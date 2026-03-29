MASTER_PORT=28765
MODEL=tinyllama
LR=2e-5

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 \
  --master_port=${MASTER_PORT} \
  measure_importance.py \
  --config-name=forget.yaml \
  split=forget10 \
  batch_size=8 \
  model_family=${MODEL} \
  model_path=./llm_weights/ft_epoch5_lr${LR}_${MODEL}_full_wd0.01/checkpoint-625