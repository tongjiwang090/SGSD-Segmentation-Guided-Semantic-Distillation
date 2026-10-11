#!/bin/bash
# single-arm success-rate evaluation runner
#
#   bash run_eval_arm.sh <arm_name> <ckpt_dir> [test_num] [seed] [gpu_id] [ckpt_name]
#
# example:
#   bash run_eval_arm.sh A-fullft-nodistill \
#       policy/ACT/act_ckpt/act-beat_block_hammer/demo_clean-50_dino 100 0 0
#
# fixed: dinov2 vit-b/14, beat_block_hammer/demo_clean, temporal_agg=true
#
# backbone args must be passed explicitly: backbone_* are all null in deploy_policy.yml
# and would fall back to resnet18 from backbone_config.yml, building the wrong backbone.
# ckpt_name must be passed explicitly: the default 'auto' looks for policy_best.ckpt, which we removed
set -euo pipefail
# activates its own env: the caller's shell is bare under cron / setsid nohup
# activates its own env: the caller's shell is bare under cron / setsid nohup
if ! command -v python >/dev/null 2>&1 || ! python -c "import torch" >/dev/null 2>&1; then
  CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
  [ -f "$CONDA_SH" ] || { echo "FATAL: $CONDA_SH not found" >&2; exit 1; }
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate RoboTwin
fi
python -c "import torch, sapien" 2>/dev/null || { echo "FATAL: robotwin env unusable (torch/sapien import failed)" >&2; exit 1; }

ARM=${1:?arm_name required}
CKPT_DIR=${2:?ckpt_dir required, relative to the robotwin root}
TEST_NUM=${3:-100}
SEED=${4:-0}
GPU_ID=${5:-0}
CKPT_NAME=${6:-policy_epoch_4000_seed_0.ckpt}
TASK_CFG=${7:-demo_clean}     # demo_clean | demo_randomized (out-of-domain)

TASK=${ACT_TASK:-beat_block_hammer}

# backbone overridable by env, same style as run_freeze_ablation.sh. dinov2 default.
BACKBONE_TYPE=${BACKBONE_TYPE:-dinov2}
BACKBONE_NAME=${BACKBONE_NAME:-vit_base_patch14_dinov2}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # robotwin root
cd "$ROOT"

if [ ! -f "$CKPT_DIR/$CKPT_NAME" ]; then
  echo "FATAL: $CKPT_DIR/$CKPT_NAME not found" >&2
  exit 1
fi
if [ ! -f "$CKPT_DIR/dataset_stats.pkl" ]; then
  echo "FATAL: $CKPT_DIR/dataset_stats.pkl not found (normalization stats; without them every action is wrong)" >&2
  exit 1
fi

LOGDIR=$ROOT/eval_result/_logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/${ARM}_${TASK_CFG}_seed${SEED}.log
if [ -f "$LOG" ]; then
  mkdir -p "$LOGDIR/_aborted"
  mv "$LOG" "$LOGDIR/_aborted/$(basename "$LOG" .log)_$(date +%Y%m%d%H%M%S).log"
  echo "[note] moved a same-named old log into _aborted/ so tee -a does not pollute parse_log" >&2
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore::UserWarning

{
echo "=========================================================="
echo " arm       : $ARM"
echo " ckpt      : $CKPT_DIR/$CKPT_NAME"
echo " task_cfg  : $TASK_CFG"
echo " test_num  : $TEST_NUM   seed: $SEED   gpu: $GPU_ID"
echo " backbone  : $BACKBONE_TYPE / $BACKBONE_NAME"
echo " intervene : ${ACT_INTERVENE_SITE:-<none>} mode=${ACT_INTERVENE_MODE:--} sigma=${ACT_INTERVENE_SIGMA:--}"
echo " started   : $(date '+%F %T')"
echo "=========================================================="
} | tee -a "$LOG"

python script/eval_policy.py --config policy/ACT/deploy_policy.yml \
    --overrides \
    --task_name "$TASK" \
    --task_config "$TASK_CFG" \
    --ckpt_setting "$ARM" \
    --ckpt_dir "$CKPT_DIR" \
    --ckpt_name "$CKPT_NAME" \
    --seed "$SEED" \
    --test_num "$TEST_NUM" \
    --temporal_agg true \
    --backbone_type "$BACKBONE_TYPE" \
    --backbone "$BACKBONE_NAME" \
    --backbone_pretrained false \
    --backbone_image_size 224 \
    --use_distillation false 2>&1 | tee -a "$LOG"

echo "finished  : $(date '+%F %T')" | tee -a "$LOG"
echo "--- result ---" | tee -a "$LOG"
find "$ROOT/eval_result/$TASK/ACT/$TASK_CFG/$ARM" -name _result.txt -newermt '-1 day' \
    -exec sh -c 'echo "{}"; cat "{}"' \; | tee -a "$LOG"
