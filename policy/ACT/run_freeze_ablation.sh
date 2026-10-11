#!/bin/bash
# freeze-position ablation runner (v1)
#
#   bash run_freeze_ablation.sh <unfreeze_spec> [num_epochs] [gpu_id] [seed] [save_freq]
#
# examples:
#   bash run_freeze_ablation.sh none 4000 0 0 2000    # fully-frozen reference
#   bash run_freeze_ablation.sh cls  4000 0 0 2000    # cls_token only
#   bash run_freeze_ablation.sh b8   4000 0 0 2000    # only transformer block 8
#
# fixed: no distillation | beat_block_hammer/demo_clean-50 | batch 4 | lr 1e-5
# backbone: dinov2 vit-b/14 by default; override with BACKBONE_TYPE / BACKBONE_NAME
set -euo pipefail

# activates its own env: the caller's shell is bare under cron / setsid nohup
if ! command -v python >/dev/null 2>&1 || ! python -c "import torch" >/dev/null 2>&1; then
  CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
  [ -f "$CONDA_SH" ] || { echo "FATAL: $CONDA_SH not found" >&2; exit 1; }
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate RoboTwin
fi
python -c "import torch, sapien" 2>/dev/null || { echo "FATAL: robotwin env unusable (torch/sapien import failed)" >&2; exit 1; }

SPEC=${1:?need unfreeze spec, e.g. none / cls / b0 / b8 / b9-11}
NUM_EPOCHS=${2:-4000}
GPU_ID=${3:-0}
SEED=${4:-0}
SAVE_FREQ=${5:-2000}

EXP=${EXP_NAME:-freeze_ablation_v1}
# backbone overridable by env, same style as EXP_NAME. dinov2 default, backward compatible.
# pretrained-weight variants (augreg2 / mae / dino ...) are kept apart by EXP_NAME, not by dir name.
BACKBONE_TYPE=${BACKBONE_TYPE:-dinov2}
BACKBONE_NAME=${BACKBONE_NAME:-vit_base_patch14_dinov2}
TASK=${ACT_TASK:-beat_block_hammer}
CFG=${ACT_TASK_CFG:-demo_clean}
NUM=50

# spec -> sortable arm name
ARM=$(python - "$SPEC" <<'PYARM'
import sys, re
s = sys.argv[1].strip().lower()
if s == "none":
    print("frozen")
elif s == "all":
    print("fullft")
elif s == "cls":
    print("clstoken")
elif re.fullmatch(r"b\d+", s):
    print("b%02d" % int(s[1:]))
else:
    print(re.sub(r"[^a-z0-9]+", "-", s).strip("-"))
PYARM
)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CKPT_DIR=$ROOT/act_ckpt/$EXP/${BACKBONE_TYPE}_nodistill_${ARM}_seed${SEED}
mkdir -p "$CKPT_DIR"
LOG=$CKPT_DIR/train_log.txt

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONUNBUFFERED=1

{
echo "=========================================================="
echo " arm       : $ARM   unfreeze_spec=$SPEC"
echo " backbone  : $BACKBONE_TYPE / $BACKBONE_NAME"
echo " ckpt_dir  : $CKPT_DIR"
echo " epochs    : $NUM_EPOCHS   save_freq: $SAVE_FREQ   seed: $SEED"
echo " gpu       : $GPU_ID"
echo " started   : $(date '+%F %T')"
echo "=========================================================="
} | tee -a "$LOG"

cd "$ROOT"
python imitate_episodes.py \
    --task_name sim-${TASK}-${CFG}-${NUM} \
    --ckpt_dir "$CKPT_DIR" \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 4 \
    --dim_feedforward 3200 \
    --num_epochs "$NUM_EPOCHS" \
    --lr 1e-5 \
    --lr_backbone 1e-5 \
    --save_freq "$SAVE_FREQ" \
    --state_dim 14 \
    --seed "$SEED" \
    --backbone_type "$BACKBONE_TYPE" \
    --backbone "$BACKBONE_NAME" \
    --backbone_pretrained true \
    --backbone_image_size 224 \
    --use_distillation false \
    --unfreeze_blocks "$SPEC" 2>&1 | tee -a "$LOG"

echo "finished  : $(date '+%F %T')" | tee -a "$LOG"
