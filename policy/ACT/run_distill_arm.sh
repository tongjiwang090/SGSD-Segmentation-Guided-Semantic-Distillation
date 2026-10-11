#!/bin/bash
# single-arm training runner with distillation
#
#   bash run_distill_arm.sh <unfreeze_spec> [num_epochs] [gpu_id] [seed] [save_freq] [exp_name]
#
# example:
#   bash run_distill_arm.sh 'b11,cls,norm' 4000 0 0 4000 method_clean_v1
#
# the only difference from run_freeze_ablation.sh is --use_distillation true.
# teacher is the up4 output of Segmentation/checkpoints/best_unet_model.pth (768ch, 224x224);
# the student is upsampled to meet it -- the teacher's H,W must not change
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

# backbone: dinov2 vit-b/14 by default; override with BACKBONE_TYPE / BACKBONE_NAME.
# must match run_freeze_ablation.sh, otherwise the frozen and distilled arms build
# different models and the paired comparison is void.
# distillation loss. structure_diversity is the three-term objective from the paper.
# ablation variants live in imitate_episodes.py: structure_diversity_{nomag,nocos,bal}.
DISTILL_LOSS=${ACT_DISTILL_LOSS:-structure_diversity}
# overall weight on the distillation loss. 1.0 matches every existing experiment.
DISTILL_W=${ACT_DISTILL_WEIGHT:-1.0}

BACKBONE_TYPE=${BACKBONE_TYPE:-dinov2}
BACKBONE_NAME=${BACKBONE_NAME:-vit_base_patch14_dinov2}

SPEC=${1:?unfreeze spec required}
NUM_EPOCHS=${2:-4000}
GPU_ID=${3:-0}
SEED=${4:-0}
SAVE_FREQ=${5:-4000}
EXP=${6:-method_clean_v1}
DISTILL_AT=${7:-}     # block to mount supervision on; empty = final layer (post-norm)

TASK=${ACT_TASK:-beat_block_hammer}
CFG=${ACT_TASK_CFG:-demo_clean}
NUM=50

# the container has 20 cores but libraries size their pools from the host's 208 ->
# heavy oversubscription. training is single-process; give it 16, leave 4 to the system.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

ARM=$(python - "$SPEC" <<'PYARM'
import sys, re
s = sys.argv[1].strip().lower()
print(re.sub(r"[^a-z0-9]+", "-", s).strip("-"))
PYARM
)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# dir name carries the mount layer, or 'unfreeze b8 + distill@b8' and 'unfreeze b8 +
if [ -n "$DISTILL_AT" ]; then
  # with several mount points DISTILL_AT looks like "5 11": spaces become -b in the arm name,
  # and the flag must expand UNQUOTED or argparse receives one argv "5 11" and fails on type=int
  ARM="${ARM}_at-b${DISTILL_AT// /-b}"
  # shellcheck disable=SC2206
  DISTILL_AT_FLAG=(--distill_at_block $DISTILL_AT)
else
  ARM="${ARM}_at-final"
  DISTILL_AT_FLAG=()
fi
# loss variants get an arm-name suffix, or ablation arms collide on one ckpt dir.
# structure_diversity takes no suffix so existing arm paths stay put.
case "$DISTILL_LOSS" in
  structure_diversity)        ;;
  structure_diversity_nomag)  ARM="${ARM}_nomag" ;;
  structure_diversity_nocos)  ARM="${ARM}_nocos" ;;
  structure_diversity_bal)    ARM="${ARM}_bal" ;;
  structure_diversity_mseonly) ARM="${ARM}_mseonly" ;;
  structure_diversity_cosonly) ARM="${ARM}_cosonly" ;;
  *) echo "FATAL: unknown ACT_DISTILL_LOSS=$DISTILL_LOSS" >&2; exit 1 ;;
esac

# non-default weights get a suffix too, else arms at different weights share a ckpt dir
if [ "$DISTILL_W" != "1.0" ]; then ARM="${ARM}_w${DISTILL_W}"; fi

# spatial teacher-permutation control: with ACT_DISTILL_SHUFFLE=1 the teacher features
# are permuted by student patch-grid blocks (block contents and global stats unchanged).
if [ "${ACT_DISTILL_SHUFFLE:-0}" = "1" ]; then
  SCOPE=${ACT_DISTILL_SHUFFLE_SCOPE:-space}
  case "$SCOPE" in
    space) ARM="${ARM}_shuf" ;;          # permute blocks within the image
    batch) ARM="${ARM}_shufbatch" ;;     # permute across images, same camera -- tight control
    *) echo "FATAL: ACT_DISTILL_SHUFFLE_SCOPE must be space|batch, got $SCOPE" >&2; exit 1 ;;
  esac
  DISTILL_AT_FLAG+=(--distill_shuffle_teacher --distill_shuffle_scope "$SCOPE")
fi

CKPT_DIR=$ROOT/act_ckpt/$EXP/${BACKBONE_TYPE}_distill_${ARM}_seed${SEED}
mkdir -p "$CKPT_DIR"
LOG=$CKPT_DIR/train_log.txt

TEACHER=${ACT_TEACHER_CKPT:-$ROOT/../../Segmentation/checkpoints/best_unet_model.pth}
if [ ! -f "$TEACHER" ]; then
  echo "FATAL: teacher weights not found at $TEACHER" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONUNBUFFERED=1

{
echo "=========================================================="
echo " arm        : $ARM   unfreeze_spec=$SPEC   DISTILL=ON"
echo " backbone   : $BACKBONE_TYPE / $BACKBONE_NAME"
echo " distill_loss: $DISTILL_LOSS"
echo " distill_at : ${DISTILL_AT:-final(post-norm)}"
echo " ckpt_dir   : $CKPT_DIR"
echo " teacher    : $TEACHER"
echo " epochs     : $NUM_EPOCHS   save_freq: $SAVE_FREQ   seed: $SEED"
echo " threads    : OMP=$OMP_NUM_THREADS (20-core container)"
echo " started    : $(date '+%F %T')"
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
    --use_distillation true \
    --distill_loss_type "$DISTILL_LOSS" \
    --distill_weight "$DISTILL_W" \
    --teacher_vis_interval 1000 \
    --unfreeze_blocks "$SPEC" \
    "${DISTILL_AT_FLAG[@]}" 2>&1 | tee -a "$LOG"

echo "finished   : $(date '+%F %T')" | tee -a "$LOG"
