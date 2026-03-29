#!/bin/bash
# =========================================
# 单卡 GA/LoRA 遗忘训练脚本（纯本地权重 + TOFU/config/my_forget.yaml）
# =========================================
cd "$(dirname "$0")"
HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
MASTER_PORT=28765

# -------------------------
# 模型、学习率、方法
# -------------------------
MODELS=( tinyllama )         # 可以换成你的模型名
LR1S=( 2e-05 )               # 用于命名 checkpoint 的学习率
LR2S=( 0.0002 )              # 实际训练学习率

METHODS=( grad_ascent )               # GA / IHL / KL / grad_ascent / grad_diff
BSZS=( 1 )                    # batch size 对应方法
GASS=( 1 )                   # gradient accumulation steps 对应方法

SPLITS=( forget10 )           # forget 数据集划分
LORA_TARGETS=( all )          # LoRA 目标

RANKS=( 2 )           # LoRA r 值
DEVICES=( 0 0 0 0 )           # 单卡跑全部用 GPU 0

# -------------------------
# 循环执行
# -------------------------
for model_idx in "${!MODELS[@]}"; do
    MODEL=${MODELS[$model_idx]}
    LR1=${LR1S[$model_idx]}
    LR2=${LR2S[$model_idx]}

    for method_idx in "${!METHODS[@]}"; do
        METHOD=${METHODS[$method_idx]}
        BSZ=${BSZS[$method_idx]}
        GAS=${GASS[$method_idx]}

        for lora_target_idx in "${!LORA_TARGETS[@]}"; do
            LORA_TARGET=${LORA_TARGETS[$lora_target_idx]}

            for split_idx in "${!SPLITS[@]}"; do
                SPLIT=${SPLITS[$split_idx]}

                for rank_idx in "${!RANKS[@]}"; do
                    RANK=${RANKS[$rank_idx]}
                    DEVICE=${DEVICES[$rank_idx]}

                    echo "========================================"
                    echo "MODEL=${MODEL} METHOD=${METHOD} SPLIT=${SPLIT} LoRA.r=${RANK}"
                    echo "========================================"

                    CUDA_VISIBLE_DEVICES=${DEVICE} torchrun \
                        --nproc_per_node=1 \
                        --master_port=${MASTER_PORT} \
                        forget.py \
                        --config-path=config \
                        --config-name=my_forget.yaml \
                        ++model_path=../llm_weights/my_tinyllama_finetune/checkpoint-4000 \
                        ++save_dir=../llm_weights/my_${MODEL}_${METHOD}_${RANK}_forget_${SPLIT} \
                        ++forget_loss=${METHOD} \
                        ++importance_file=null \
                        data_path=./data/ \
                        LoRA.targets=${LORA_TARGET} \
                        LoRA.r=${RANK} \
                        LoRA.alpha=$(( ${RANK} * 2 )) \
                        LoRA.dropout=0 \
                        num_epochs=5 \
                        split=${SPLIT} \
                        batch_size=${BSZ} \
                        gradient_accumulation_steps=${GAS} \
                        model_family=${MODEL} \
                        lr=${LR2}

                done
            done
        done
    done
done