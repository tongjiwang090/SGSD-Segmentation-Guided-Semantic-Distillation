# Segmentation-Guided Semantic Distillation (SGSD)

This repository contains the segmentation teacher and the modified ACT and Diffusion
Policy implementations used by **Segmentation-Guided Semantic Distillation for Robot
Manipulation**. SGSD trains a task-specific U-Net on RoboTwin actor-segmentation labels
and distills its dense spatial features directly into the policy's visual backbone
during fine-tuning.

The code is designed to be used inside
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) as an overlay: every path here
mirrors a RoboTwin path, so the files can be copied over without editing imports. The
repository intentionally does not include RoboTwin, datasets, checkpoints, or generated
visualizations, and it ships only the files this work modified — unmodified upstream
files come from RoboTwin itself.

## Method

The training pipeline has two stages:

1. `Segmentation/` trains an attention-gated residual U-Net on RGB frames and
   actor-segmentation masks stored in RoboTwin HDF5 episodes. The intermediate `up4`
   feature map is the teacher representation.
2. `policy/ACT/` trains ACT with the usual action and KL objectives plus a dense feature
   distillation loss on the visual backbone.

Teacher and student both carry 768 channels, so alignment is spatial only: the student's
patch grid is bilinearly upsampled to the teacher's `224x224`. **There is no learned
projection on either side**, and the teacher is discarded at inference, leaving the
deployed architecture unchanged.

Three terms are optimised jointly with the policy objective:

```
L = L_policy + lambda_D * ( L_feat + 0.5 * L_dir + 0.5 * L_mag )

L_feat   mean squared error between corresponding feature vectors
L_dir    1 - cosine similarity, scale-invariant
L_mag    squared difference of feature norms
```

These coefficients are nominal. In the reported run the realised loss mass at the final
epoch was roughly 3% / 4% / 93%, so the magnitude term dominates the scalar objective —
which does not by itself establish its share of the gradient.

Supervision is applied to the **pre-LayerNorm** output of the final block while the
policy reads the **post-LayerNorm** features. These are distinct measurement points:
LayerNorm removes per-token scale, so a norm map at the supervised site is not preserved
at the policy input.

`imitate_episodes.py` also retains a number of earlier distillation objectives
(`focal_cosine`, `anti_artifact`, the `attention_mse_*` family). The paper uses
`structure_diversity`, which is the default throughout this repository.

## Repository layout

```text
sgsd/
  train_teacher.py     teacher training entry point
  train_policy.py      policy training entry point, one flag per arm
  evaluate.py          closed-loop evaluation of a checkpoint

Segmentation/
  dataset.py           HDF5 dataset, train/validation split, mask conventions
  unet.py              attention-gated residual U-Net, up4 emits 768 channels
  train.py             teacher training
  evaluate.py          single-image and episode inference
  extract_data.py      export RGB and actor masks from raw episodes

policy/ACT/
  imitate_episodes.py  distillation losses, teacher loading, training loop
  act_policy.py        ACT policy wrapper, auxiliary distillation projection removed
  constants.py         task definitions, read from SIM_TASK_CONFIGS.json
  utils.py             episode dataset, carries ground-truth segmentation
  deploy_policy.py     RoboTwin policy entry point: get_model / eval / reset_model
  detr/models/backbone.py  mount point and trainable-visual-set selection
  detr/main.py         adds --backbone_type and --backbone_image_size
  run_distill_arm.sh   distilled arm
  run_freeze_ablation.sh   frozen and action-only arms
  run_eval_arm.sh      evaluation

policy/DP/diffusion_policy/
  model/vision/dp_distill.py   the same objective for Diffusion Policy
  model/vision/dp_vit.py       vit visual head replacing the random resnet18
  workspace/robotworkspace.py  wires the loss into DP training
  config/robot_dp_vit_14.yaml  config for the vit-backbone DP runs
```

## Requirements

Use the RoboTwin environment. `requirements.txt` lists only what this method adds on top
of it; RoboTwin's own dependencies are not duplicated.

```bash
pip install -r requirements.txt
```

Verified on Python 3.10, CUDA 12.1, torch 2.4.1+cu121, RTX 4090 (sm_89).

Run all commands below from the RoboTwin root unless a command explicitly changes
directory.

## Prepare RoboTwin data

Collect demonstrations with the normal RoboTwin workflow:

```bash
bash collect_data.sh beat_block_hammer demo_clean 0
```

Convert the raw episodes into the ACT format. The converter also updates
`policy/ACT/SIM_TASK_CONFIGS.json`.

```bash
cd policy/ACT
bash process_data.sh beat_block_hammer demo_clean 50
cd ../..
```

The converted files are written to
`policy/ACT/processed_data/sim-beat_block_hammer/demo_clean-50/`. Each episode contains
RGB images under `observations/images` and segmentation masks under
`observations/segmentation`.

## Train the segmentation teacher

Train one teacher per task. Using a shared output directory overwrites the previous
teacher.

```bash
python -m sgsd.train_teacher \
    --data-dir policy/ACT/processed_data/sim-beat_block_hammer/demo_clean-50 \
    --save-dir Segmentation/checkpoints_beat_block_hammer
```

The best checkpoint is saved as `<save-dir>/best_unet_model.pth`. The default split uses
80% of the episodes for training and the remainder for validation.

By default the teacher's foreground is every non-background pixel. To supervise only the
manipulated objects, pass their simulator actor ids:

```bash
python -m sgsd.train_teacher --data-dir <...> --save-dir <...> \
    --mask-mode target --target-ids 178,184
```

Actor ids are scene-assembly ordinals, not object identities. They differ per task and
must be read from `observations/segmentation` rather than assumed.

For raw RoboTwin episodes, `Segmentation/extract_data.py` exports RGB and actor masks to
per-camera PNG folders.

## Run segmentation inference

Evaluate one image and save a visualization containing the original image, prediction,
and overlay:

```bash
cd Segmentation
python evaluate.py \
  --model checkpoints_beat_block_hammer/best_unet_model.pth \
  --image /path/to/frame.png \
  --save_path results/frame_prediction.png
```

To inspect an extracted episode, provide its directory instead of `--image`:

```bash
python evaluate.py \
  --model checkpoints_beat_block_hammer/best_unet_model.pth \
  --test_data /path/to/extracted_all_episodes/episode0
```

The teacher always resizes RGB inputs to `224x224` and applies ImageNet normalization.
The same preprocessing is applied inside the distillation loop.

## Train ACT with SGSD

```bash
python -m sgsd.train_policy --arm sgsd --task beat_block_hammer \
    --teacher-ckpt Segmentation/checkpoints_beat_block_hammer/best_unet_model.pth
```

Baselines use the same command with a different arm:

```bash
python -m sgsd.train_policy --arm frozen      --task beat_block_hammer
python -m sgsd.train_policy --arm action-only --task beat_block_hammer
```

| `--arm` | Trainable visual parameters | Teacher |
|---|---|---|
| `sgsd` | `b11`, CLS token, final LayerNorm | yes |
| `action-only` | same set | no |
| `frozen` | none | no |

The paper configuration is the default: 4,000 epochs, batch size 4, learning rate `1e-5`
for both backbone and policy, supervision mounted on `b11`. Add `--print-only` to resolve
and print the configuration without training.

Checkpoints and logs are written to `policy/ACT/act_ckpt/<task>_<arm>_seed<seed>/`.

Ablations:

```bash
--loss structure_diversity_nomag    # drop the magnitude term
--loss structure_diversity_nocos    # drop the direction term
--loss structure_diversity_mseonly  # feature MSE only
--mount-block 8 --unfreeze b8       # supervise b8 instead of b11
--mount-block -1                    # supervise the post-LayerNorm output
--shuffle-teacher space|batch       # teacher-correspondence controls
--backbone-type vit|clip            # supervised ViT-B/16 or CLIP ViT-B/16
```

Moving the mount point requires moving `--unfreeze` with it. The supervised block must
be trainable or no gradient reaches the loss, and the run completes without ever
optimizing the distillation objective.

`--mount-block` also takes several blocks (`--mount-block 5 11 --unfreeze b5,b11`). That
is the appendix placement ablation, not the method: two-point supervision matches
single-point success while degrading pretrained features considerably more, which is why
the method supervises `b11` alone.

If the teacher checkpoint is missing, the training code builds a randomly initialized
teacher and prints a warning. That path exists for smoke testing only; do not use it for
an actual experiment.

## ACT inference and RoboTwin deployment

```bash
python -m sgsd.evaluate --name bbh-sgsd \
    --ckpt-dir policy/ACT/act_ckpt/beat_block_hammer_sgsd_seed0
```

This wraps `policy/ACT/run_eval_arm.sh`, which changes to the RoboTwin root and invokes
`script/eval_policy.py`. The deployment adapter in `policy/ACT/deploy_policy.py` converts
RoboTwin observations into ACT tensors and denormalizes predicted actions.

`--task-config` must match the configuration the policy was trained on. Evaluating a
policy under a configuration it was not trained on measures something else.

## Checkpoints

Not included in this repository. Each released checkpoint directory holds
`policy_epoch_4000_seed_0.ckpt` and the `run_config.json` it was trained with; the
`task_name` field in that file records the training configuration.

```bash
# continue training from, or fine-tune, a released checkpoint
python -m sgsd.train_policy --arm sgsd --task beat_block_hammer \
    --teacher-ckpt <teacher.pth> \
    --resume-ckpt <dir>/policy_epoch_4000_seed_0.ckpt

# evaluate one directly
python -m sgsd.evaluate --name bbh-sgsd --ckpt-dir <dir> \
    --ckpt-name policy_epoch_4000_seed_0.ckpt
```

## Train Diffusion Policy with SGSD

```bash
cd policy/DP
bash process_data.sh beat_block_hammer demo_clean 50
bash train.sh beat_block_hammer demo_clean 50 0 14 0
```

Distillation is off by default and is enabled from the command line with
`distill.enabled=true`; see `config/robot_dp_vit_14.yaml` for the mount point, sample
count and weight. DP draws a random subset of images per step for the distillation loss
because the full batch does not fit in memory — a known difference from the ACT side.

## Embed SGSD into RoboTwin

Use a RoboTwin checkout with the same data and API version used to collect the
demonstrations.

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin && bash script/_install.sh

cp -r /path/to/SGSD/{Segmentation,policy,envs} .
cp -r /path/to/SGSD/sgsd .
```

Paths are identical to RoboTwin's, so no import needs editing. Files replaced by the
overlay, and what each one changes:

| Path | Change |
|---|---|
| `Segmentation/` | the teacher; written for this work |
| `policy/ACT/imitate_episodes.py` | the three distillation terms, teacher loading, the forward hook exposing the supervised block |
| `policy/ACT/detr/models/backbone.py` | trainable-visual-set selector, timm ViT backbones, optional readout pooling |
| `policy/ACT/detr/main.py` | adds `--backbone_type` and `--backbone_image_size` |
| `policy/ACT/act_policy.py` | removes the auxiliary learned distillation projection |
| `policy/ACT/deploy_policy.py`, `__init__.py` | RoboTwin's policy entry point; routes the backbone configuration into the evaluated model |
| `policy/ACT/constants.py`, `utils.py` | task definitions from JSON; segmentation carried through the episode pipeline |
| `policy/DP/diffusion_policy/` | the same objective for Diffusion Policy |
| `envs/camera/camera.py` | records per-actor and per-mesh segmentation alongside RGB |

Two environment switches matter and fail silently if lost:

- **`ACT_TASK_CFG`** (`run_distill_arm.sh`, `run_freeze_ablation.sh`).
  Without it the scripts fall back to `demo_clean`, so a non-clean experiment trains and
  evaluates on clean data and reports plausible but wrong numbers.
- **`TEACHER_SEG_MODE`** (`Segmentation/dataset.py`) selects the mask convention
  (`all` / `target`). Unset, behaviour is pixel-identical to the original.

`dataset.py` also fixes an ordering defect in the original: the mask was cast with
`astype(np.uint8)` before resizing, so actor ids above 255 wrapped (260 selected an
unrelated actor, 256 dropped out of the foreground). It now binarizes first.

## Citation

If you use this code, please cite the repository and the accompanying paper. The paper is
currently under double-blind review; author metadata is therefore intentionally omitted.

```bibtex
@article{sgsd2026,
  title   = {Segmentation-Guided Semantic Distillation for Robot Manipulation},
  author  = {Anonymous authors},
  journal = {Under review at ICLR 2027},
  year    = {2026},
  url     = {https://github.com/tongjiwang090/SGSD-Segmentation-Guided-Semantic-Distillation}
}
```

This project builds on RoboTwin:

```bibtex
@article{chen2025robotwin,
  title   = {RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author  = {Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal = {arXiv preprint arXiv:2506.18088},
  year    = {2025}
}
```

Please also cite the original ACT and Diffusion Policy implementations, and follow the
licenses of RoboTwin, ACT, DETR, and their dependencies when redistributing this code.

## License

Apache License 2.0; see `LICENSE`. `NOTICE` lists the upstream projects each modified
file derives from and the scope of every change. Files under `policy/ACT/detr/` retain
Facebook's original copyright notices.
