#!/bin/bash
task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
backbone_type=${6:-""}
backbone=${7:-""}
backbone_pretrained=${8:-"true"}
backbone_image_size=${9:-""}
use_distillation=${10:-""}
freeze_last_layer_and_cls=${11:-""}
resume_ckpt=${12:-""}

DEBUG=False
save_ckpt=True

export CUDA_VISIBLE_DEVICES=${gpu_id}

CMD="python3 imitate_episodes.py \
    --task_name sim-${task_name}-${task_config}-${expert_data_num} \
    --ckpt_dir ./act_ckpt/act-${task_name}/${task_config}-${expert_data_num} \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 6000 \
    --lr 1e-5 \
    --save_freq 2000 \
    --state_dim 14 \
    --seed ${seed}"

# Add backbone arguments if provided
if [ -n "$backbone_type" ]; then
    CMD="$CMD --backbone_type ${backbone_type}"
fi

if [ -n "$backbone" ]; then
    CMD="$CMD --backbone ${backbone}"
fi

if [ -n "$backbone_pretrained" ]; then
    CMD="$CMD --backbone_pretrained ${backbone_pretrained}"
fi

if [ -n "$backbone_image_size" ]; then
    CMD="$CMD --backbone_image_size ${backbone_image_size}"
fi

if [ -n "$use_distillation" ]; then
    CMD="$CMD --use_distillation ${use_distillation}"
fi

if [ -n "$freeze_last_layer_and_cls" ]; then
    CMD="$CMD --freeze_last_layer_and_cls ${freeze_last_layer_and_cls}"
fi

if [ -n "$resume_ckpt" ]; then
    CMD="$CMD --resume_ckpt ${resume_ckpt}"
fi

# Execute the command
$CMD

# bash train.sh beat_block_hammer demo_clean 50 0 0 dinov2 vit_base_patch14_dinov2 true 224 true
