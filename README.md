# Segmentation-Guided Semantic Distillation (SGSD)

This repository contains the segmentation teacher and the modified ACT policy used by **Segmentation-Guided Semantic Distillation for Robot Manipulation**. SGSD trains a task-specific U-Net on RoboTwin actor-segmentation labels and distills its dense spatial features into the ACT visual backbone during policy fine-tuning.

The code is designed to be used inside [RoboTwin](https://github.com/TianxingChen/RoboTwin), with the files under `policy/ACT` replacing RoboTwin's ACT implementation. The repository intentionally does not include RoboTwin, datasets, checkpoints, or generated visualizations.

## Method

The training pipeline has two stages:

1. `segmentation/` trains an attention-gated residual U-Net on RGB frames and actor-segmentation masks stored in RoboTwin HDF5 episodes. The intermediate `up4` feature map is the teacher representation.
2. `policy/ACT/` trains ACT with the normal action and KL objectives plus a dense feature distillation loss. ACT features are projected to 64 channels, resized to the U-Net `up4` resolution, and compared with one teacher feature map per camera.

The implementation supports cosine, MSE, focal, spatially weighted, anti-artifact, and magnitude-aware distillation losses. The default loss is `focal_cosine`; choose another loss with `--distill_loss_type`.

## Repository layout

```text
segmentation/
  dataset.py       HDF5 dataset and train/validation split
  unet.py          Attention-gated residual U-Net
  train.py         Teacher training entry point
  evaluate.py      Single-image and episode inference
  extract_data.py  Extract RGB and actor masks from raw episodes

policy/ACT/
  imitate_episodes.py  ACT training and validation
  act_policy.py        ACT policy wrapper used by RoboTwin
  deploy_policy.py     RoboTwin deployment adapter
  process_data.py      Convert raw RoboTwin episodes to ACT HDF5
  detr/                ACT VAE, transformer, and visual backbones
  backbone_config.yml  Shared backbone defaults
  train.sh / eval.sh   RoboTwin-compatible shell entry points
```

## Requirements

Use the RoboTwin environment, or create an equivalent Python 3.9 environment from `policy/ACT/conda_env.yaml`. A CUDA-enabled PyTorch installation is recommended for both stages. The ACT backbone code additionally requires `timm` and the pretrained weights for the selected backbone.

```bash
conda env create -f policy/ACT/conda_env.yaml
conda activate aloha
pip install timm
```

Run all commands below from the repository root unless a command explicitly changes directory.

## Prepare RoboTwin data

First collect RoboTwin demonstrations with the normal RoboTwin workflow. For example:

```bash
bash collect_data.sh beat_block_hammer demo_clean 0
```

Convert the raw episodes into the ACT format. The converter also updates `policy/ACT/SIM_TASK_CONFIGS.json`.

```bash
cd policy/ACT
bash process_data.sh beat_block_hammer demo_clean 50
cd ../..
```

The converted files are written to `policy/ACT/processed_data/sim-beat_block_hammer/demo_clean-50/`. Each episode contains RGB images under `observations/images` and segmentation masks under `observations/segmentation`.

## Train the segmentation teacher

Train the U-Net using the converted ACT episodes. The default split uses 80% of the episodes for training and the remainder for validation.

```bash
cd segmentation
python train.py \
  --data_dir ../policy/ACT/processed_data/sim-beat_block_hammer/demo_clean-50 \
  --epochs 20 \
  --batch_size 12 \
  --lr 1e-3
```

The best checkpoint is saved as `segmentation/checkpoints/best_unet_model.pth`. To resume, pass `--resume path/to/checkpoint.pth`; pass an empty or nonexistent path when starting from scratch.

For raw RoboTwin episodes, `extract_data.py` exports RGB and actor masks to per-camera PNG folders:

```bash
cd segmentation
python extract_data.py
```

## Run segmentation inference

Evaluate one image and save a visualization containing the original image, prediction, and overlay:

```bash
cd segmentation
python evaluate.py \
  --model checkpoints/best_unet_model.pth \
  --image /path/to/frame.png \
  --save_path results/frame_prediction.png
```

To inspect an extracted episode, provide its directory instead of `--image`:

```bash
python evaluate.py \
  --model checkpoints/best_unet_model.pth \
  --test_data /path/to/extracted_all_episodes/episode0
```

The teacher always resizes RGB inputs to `224x224` and applies ImageNet normalization. The same preprocessing is applied inside the ACT distillation loop.

## Train ACT with SGSD

Make sure `segmentation/checkpoints/best_unet_model.pth` exists before enabling distillation. From `policy/ACT`, run:

```bash
cd policy/ACT
bash train.sh \
  beat_block_hammer demo_clean 50 0 0 \
  dinov2 vit_base_patch14_dinov2 true 224 true false
```

The positional arguments are:

```text
task_name task_config expert_data_num seed gpu_id
[backbone_type] [backbone] [backbone_pretrained] [backbone_image_size]
[use_distillation] [freeze_last_layer_and_cls] [resume_ckpt]
```

The command writes checkpoints and logs to `policy/ACT/act_ckpt/act-<task_name>/<task_config>-<expert_data_num>/`. The default backbone configuration is in `policy/ACT/backbone_config.yml`; command-line values override that file. Supported backbone families are `resnet`, `vit`, `dinov2`, and `clip`.

Useful distillation options accepted by `imitate_episodes.py` include:

```text
--distill_loss_type focal_cosine
--distill_focal_gamma 2.0
--distill_weight 10.0
--teacher_vis_interval 500   # set to 0 to disable teacher visualizations
```

If the teacher checkpoint is missing, the current code creates a randomly initialized teacher for smoke testing and prints a warning. Do not use that fallback for an actual experiment.

## ACT inference and RoboTwin deployment

Evaluate a trained policy through RoboTwin's standard evaluation entry point:

```bash
cd policy/ACT
bash eval.sh \
  beat_block_hammer demo_clean demo_clean 50 0 0 \
  dinov2 vit_base_patch14_dinov2 true 224 true true auto
```

`eval.sh` changes to the RoboTwin root and invokes `script/eval_policy.py`. The deployment adapter in `policy/ACT/deploy_policy.py` converts RoboTwin observations (`head_camera`, `left_camera`, `right_camera`, and joint state) into ACT tensors and denormalizes predicted actions.

## Embed SGSD into RoboTwin

Use a RoboTwin checkout with the same data and API version as the one used to collect demonstrations.

1. Copy this repository's `segmentation/` directory to the RoboTwin root.
2. Replace RoboTwin's `policy/ACT/` directory with this repository's `policy/ACT/` directory, keeping any locally generated `act_ckpt/` or `processed_data/` outside version control.
3. Install the dependencies and run `pip install -e policy/ACT/detr` if the environment does not already resolve the local `detr` package.
4. Train the teacher at `segmentation/checkpoints/best_unet_model.pth`.
5. Run `policy/ACT/process_data.sh` and then `policy/ACT/train.sh` with `use_distillation=true`.
6. Use `policy/ACT/eval.sh` or RoboTwin's normal deployment command for inference.

The modified ACT model creates `DETRVAE.distill_proj` only when `use_distillation` is enabled. It maps the selected visual backbone's channels to the 64-channel U-Net `up4` representation, so the replacement is self-contained and does not require changes to RoboTwin's evaluation script.

## Citation

If you use this code, please cite the repository and the accompanying paper. The paper is currently under double-blind review; author metadata is therefore intentionally omitted.

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

Please also cite the original ACT implementation and follow the licenses of RoboTwin, ACT, DETR, and their dependencies when redistributing this code.

## License

The repository follows the MIT license included in the upstream project. Third-party components under `policy/ACT/detr/` retain their original notices.
