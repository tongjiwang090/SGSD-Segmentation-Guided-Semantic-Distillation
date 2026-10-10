#!/bin/bash

# == keep unchanged ==
policy_name=ACT
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
backbone_type=${7:-""}
backbone=${8:-""}
backbone_pretrained=${9:-""}
backbone_image_size=${10:-""}
use_distillation=${11:-""}
temporal_agg=${12:-"true"}
ckpt_name=${13:-"auto"}
DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}
export PYTHONWARNINGS=ignore::UserWarning
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

CMD="python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --ckpt_dir policy/ACT/act_ckpt/act-${task_name}/${ckpt_setting}-${expert_data_num} \
    --seed ${seed} \
    --temporal_agg ${temporal_agg} \
    --ckpt_name ${ckpt_name}"

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

$CMD

# Example:
# bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0 dinov2 vit_base_patch14_dinov2 true 224 true true auto
# bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0 "" "" "" "" "" true policy_last.ckpt
