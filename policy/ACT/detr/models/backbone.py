# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Backbone modules."""
from collections import OrderedDict
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

try:
    import timm
except ImportError:
    timm = None

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                             unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


RESNET_NUM_CHANNELS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
}

DEFAULT_BACKBONES = {
    "resnet": "resnet18",
    "vit": "vit_base_patch16_224",
    "dinov2": "vit_base_patch14_dinov2",
    "clip": "vit_base_patch16_clip_224",
}


def _get_arg(args, key: str, default=None):
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


def _to_hw(size) -> Optional[Tuple[int, int]]:
    if size is None:
        return None
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (list, tuple)):
        if len(size) == 1:
            return (int(size[0]), int(size[0]))
        if len(size) == 2:
            return (int(size[0]), int(size[1]))
    if isinstance(size, str):
        parts = [part.strip() for part in size.split(",") if part.strip()]
        if len(parts) == 1:
            value = int(parts[0])
            return (value, value)
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    raise ValueError(f"Unsupported backbone_image_size: {size}")


def _normalize_backbone_type(backbone_type: Optional[str], backbone_name: Optional[str]) -> str:
    if backbone_type:
        return backbone_type.lower()

    name = (backbone_name or "").lower()
    if name.startswith("resnet"):
        return "resnet"
    if "dinov2" in name:
        return "dinov2"
    if "clip" in name:
        return "clip"
    if "vit" in name:
        return "vit"
    if name in DEFAULT_BACKBONES:
        return name
    return "resnet"


def _resolve_backbone_name(backbone_type: str, backbone_name: Optional[str]) -> str:
    if not backbone_name or backbone_name.lower() == backbone_type:
        return DEFAULT_BACKBONES[backbone_type]
    return backbone_name


def _freeze_backbone(backbone: nn.Module, train_backbone: bool):
    if train_backbone:
        return
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)


def _freeze_except_last_layer(backbone: nn.Module, backbone_type: str):
    # English
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    # English
    if backbone_type == "resnet":
        # English ResNet,English layer4
        if hasattr(backbone, "layer4"):
            for parameter in backbone.layer4.parameters():
                parameter.requires_grad_(True)
    else:
        # English Timm English(English ViT, DINOv2, CLIP)

        # 1. English CLS Token (English:English)
        if hasattr(backbone, "cls_token"):
            backbone.cls_token.requires_grad_(True)

        # 2. English Attention English (English CLS English Patch English) English
        if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
            # English attention English
            for block in backbone.blocks:
                if hasattr(block, "attn"):
                    for parameter in block.attn.parameters():
                        parameter.requires_grad_(True)
            # English (English MLP English Attention)
            for parameter in backbone.blocks[-1].parameters():
                parameter.requires_grad_(True)

        if hasattr(backbone, "norm"):
            for parameter in backbone.norm.parameters():
                parameter.requires_grad_(True)


def _freeze_last_layer_and_cls(backbone: nn.Module, backbone_type: str):
    # English
    for parameter in backbone.parameters():
        parameter.requires_grad_(True)

    # English CLS token
    if backbone_type == "resnet":
        # English ResNet,English layer4
        if hasattr(backbone, "layer4"):
            for parameter in backbone.layer4.parameters():
                parameter.requires_grad_(False)
    else:
        # English Timm English(English ViT, DINOv2, CLIP)

        # 1. English CLS Token
        if hasattr(backbone, "cls_token"):
            backbone.cls_token.requires_grad_(False)

        # 2. English
        if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
            for parameter in backbone.blocks[-1].parameters():
                parameter.requires_grad_(False)

        if hasattr(backbone, "norm"):
            for parameter in backbone.norm.parameters():
                parameter.requires_grad_(False)


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        self.train_backbone = train_backbone
        _freeze_backbone(backbone, train_backbone)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def train(self, mode: bool = True):
        super().train(mode)
        if not getattr(self, "train_backbone", False):
            self.body.eval()
        else:
            self.body.eval()
            if hasattr(self.body, "layer4"):
                self.body.layer4.train(mode)
        return self

    def forward(self, tensor):
        return self.body(tensor)


class ResNetBackbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
        pretrained: bool,
    ):
        if name not in RESNET_NUM_CHANNELS:
            raise ValueError(f"Unsupported ResNet backbone: {name}")
        weights = torchvision.models.get_model_weights(name).DEFAULT if pretrained else None
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights=weights,
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = RESNET_NUM_CHANNELS[name]
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class TimmVisionBackbone(nn.Module):

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        pretrained: bool,
        image_size: Optional[Tuple[int, int]],
        freeze_last_layer_and_cls: bool = False,
    ):
        super().__init__()
        self.train_backbone = train_backbone
        self.freeze_last_layer_and_cls = freeze_last_layer_and_cls
        if timm is None:
            raise ImportError("timm is required for ViT, DINOv2 and CLIP backbones")
        self.body = timm.create_model(name, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        self.num_channels = int(getattr(self.body, "num_features"))
        self.patch_size = self._normalize_patch_size(getattr(self.body.patch_embed, "patch_size", 16))
        default_size = getattr(self.body, "img_size", None)
        if default_size is None:
            default_size = getattr(self.body.patch_embed, "img_size", None)
        self.image_size = _to_hw(image_size) if image_size is not None else _to_hw(default_size)
        _freeze_backbone(self.body, train_backbone)

    def train(self, mode: bool = True):
        super().train(mode)
        if not getattr(self, "train_backbone", False):
            self.body.eval()
        else:
            self.body.eval()
            if self.freeze_last_layer_and_cls:
                # English CLS,English
                for i, block in enumerate(self.body.blocks):
                    if i != len(self.body.blocks) - 1:
                        block.train(mode)
            else:
                # English:English
                if hasattr(self.body, "blocks") and len(self.body.blocks) > 0:
                    self.body.blocks[-1].train(mode)
                    if hasattr(self.body, "norm"):
                        self.body.norm.train(mode)
        return self

    @staticmethod
    def _normalize_patch_size(patch_size) -> Tuple[int, int]:
        if isinstance(patch_size, tuple):
            return (int(patch_size[0]), int(patch_size[1]))
        return (int(patch_size), int(patch_size))

    def _prepare_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.image_size is None:
            target_h = int(math.ceil(tensor.shape[-2] / self.patch_size[0]) * self.patch_size[0])
            target_w = int(math.ceil(tensor.shape[-1] / self.patch_size[1]) * self.patch_size[1])
            target_size = (target_h, target_w)
        else:
            target_h = int(math.ceil(self.image_size[0] / self.patch_size[0]) * self.patch_size[0])
            target_w = int(math.ceil(self.image_size[1] / self.patch_size[1]) * self.patch_size[1])
            target_size = (target_h, target_w)

        if tensor.shape[-2:] == target_size:
            return tensor
        return F.interpolate(tensor, size=target_size, mode="bicubic", align_corners=False)

    def _to_feature_map(self, features: torch.Tensor, input_hw: Tuple[int, int]) -> torch.Tensor:
        if isinstance(features, (list, tuple)):
            features = features[-1]

        if features.ndim == 4:
            return features
        if features.ndim != 3:
            raise ValueError(f"Unsupported timm backbone output rank: {features.ndim}")

        num_prefix_tokens = int(getattr(self.body, "num_prefix_tokens", 0))
        patch_tokens = features[:, num_prefix_tokens:, :]
        grid_h = input_hw[0] // self.patch_size[0]
        grid_w = input_hw[1] // self.patch_size[1]
        if patch_tokens.shape[1] != grid_h * grid_w:
            raise ValueError(
                f"Token count mismatch for {self.body.__class__.__name__}: "
                f"expected {grid_h * grid_w}, got {patch_tokens.shape[1]}"
            )
        return patch_tokens.transpose(1, 2).reshape(features.shape[0], features.shape[2], grid_h, grid_w)

    def forward(self, tensor):
        tensor = self._prepare_input(tensor)
        features = self.body.forward_features(tensor)
        feature_map = self._to_feature_map(features, tensor.shape[-2:])
        return OrderedDict({"0": feature_map})


class Joiner(nn.Sequential):

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    lr_backbone = _get_arg(args, "lr_backbone", 0)
    train_backbone = lr_backbone > 0
    return_interm_layers = _get_arg(args, "masks", False)
    backbone_type = _normalize_backbone_type(
        _get_arg(args, "backbone_type", None),
        _get_arg(args, "backbone", None),
    )
    backbone_name = _resolve_backbone_name(backbone_type, _get_arg(args, "backbone", None))
    pretrained_value = _get_arg(args, "backbone_pretrained", None)
    pretrained = is_main_process() if pretrained_value is None else bool(pretrained_value)
    image_size = _to_hw(_get_arg(args, "backbone_image_size", None))
    freeze_last_layer_and_cls = _get_arg(args, "freeze_last_layer_and_cls", False)

    if backbone_type == "resnet":
        backbone = ResNetBackbone(
            backbone_name,
            train_backbone,
            return_interm_layers,
            _get_arg(args, "dilation", False),
            pretrained,
        )
        if train_backbone:
            if freeze_last_layer_and_cls:
                _freeze_last_layer_and_cls(backbone.body, backbone_type)
            else:
                _freeze_except_last_layer(backbone.body, backbone_type)
    else:
        if return_interm_layers:
            raise NotImplementedError(f"`masks=True` is not supported for `{backbone_type}` backbones")
        backbone = TimmVisionBackbone(
            backbone_name,
            train_backbone,
            pretrained,
            image_size,
            freeze_last_layer_and_cls=freeze_last_layer_and_cls,
        )
        if train_backbone:
            if freeze_last_layer_and_cls:
                _freeze_last_layer_and_cls(backbone.body, backbone_type)
            else:
                _freeze_except_last_layer(backbone.body, backbone_type)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
