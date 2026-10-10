# ACT Backbone Training Guide

The shared defaults live in [`backbone_config.yml`](backbone_config.yml). Every command-line value passed to `train.sh` or `eval.sh` overrides the corresponding value in that file.

## Supported configurations

| Family | Example model | Typical input size |
| --- | --- | --- |
| `resnet` | `resnet18` | model default |
| `vit` | `vit_base_patch16_224` | `224` |
| `dinov2` | `vit_base_patch14_dinov2` | `518` |
| `clip` | `vit_base_patch16_clip_224` | `224` |

For a frozen pretrained backbone, keep `backbone_pretrained=true` and leave `freeze_last_layer_and_cls=false`. Set `freeze_last_layer_and_cls=true` when the last block and CLS token should remain fixed while the other backbone layers are trained.

## Examples

```bash
# Standard ACT baseline
bash train.sh beat_block_hammer demo_clean 50 0 0

# SGSD with DINOv2
bash train.sh beat_block_hammer demo_clean 50 0 0 \
  dinov2 vit_base_patch14_dinov2 true 224 true false

# SGSD with ViT
bash train.sh beat_block_hammer demo_clean 50 0 0 \
  vit vit_base_patch16_224 true 224 true false
```

The selected pretrained weights are downloaded by `timm` on first use. For offline runs, download them in advance with `download_pretrained_backbones.py` and set the corresponding cache environment variables.
