# ACT Distillation Notes

The complete SGSD workflow is documented in the repository [README](../../README.md). This file records the invariants that must stay aligned when changing the teacher or host policy.

## Feature alignment

The U-Net teacher is trained at `224x224` with ImageNet normalization:

```python
images_resized = F.interpolate(
    images_reshaped, size=(224, 224), mode="bilinear", align_corners=False
)
mean = torch.tensor([0.485, 0.456, 0.406], device=images_resized.device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=images_resized.device).view(1, 3, 1, 1)
images_normalized = (images_resized - mean) / std
```

The student backbone receives the normal ACT image tensor. `DETRVAE.distill_proj` maps the student feature channels to the 64 channels emitted by `UNet.up4`; the spatial dimensions are then aligned with bilinear interpolation.

## Teacher checkpoint

Place the trained checkpoint at:

```text
segmentation/checkpoints/best_unet_model.pth
```

The checkpoint must contain `model_state_dict` and `num_classes`, as produced by `segmentation/train.py`. The ACT trainer freezes the teacher and uses a forward hook on `up4`; no teacher gradients are accumulated.

## Diagnostics

When distillation is enabled, the trainer writes `artifact_stats.csv` and optional teacher visualizations under the policy checkpoint directory. Set `--teacher_vis_interval 0` to disable image generation. Monitor the cosine similarity and the teacher/student norm columns for preprocessing or resolution mismatches.
