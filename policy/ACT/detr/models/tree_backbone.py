# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import sys

import pathlib
import os
HOME_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(HOME_PATH+"/ACT/detr")
print(HOME_PATH)

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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

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


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


# class DINOv2BackBone(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.body = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
#         self.body.eval()
#         self.num_channels = 384

#     @torch.no_grad()
#     def forward(self, tensor):
#         xs = self.body.forward_features(tensor)["x_norm_patchtokens"]
#         od = OrderedDict()
#         od["0"] = xs.reshape(xs.shape[0], 22, 16, 384).permute(0, 3, 2, 1)
#         return od

class DINOv2BackBone(nn.Module):
    def __init__(self, return_interm_layers: bool = False) -> None:
        super().__init__()
        self.body = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        # self.body = torch.hub.load('/home/realman/james/unitree_h1_learn-main/dinov2',  # English
                                  # 'dinov2_vits14',  # English
                                 #  source='local',  # English,English
                                 #  pretrained=False)

        # EnglishDINOv2English
        for param in self.body.parameters():
            param.requires_grad = False

        # English block English norm English,English
        for param in self.body.blocks[-1].parameters():
            param.requires_grad = True
        for param in self.body.norm.parameters():
            param.requires_grad = True

        self.body.eval()
        # English norm English train English
        self.body.blocks[-1].train()
        self.body.norm.train()

        self.num_channels = 384
        self.return_interm_layers = return_interm_layers

    def get_last_selfattention(self, x):
        """
        English DINOv2 English block English
        """
        attentions = []
        def hook(module, input, output):
            # output of qkv is (B, N, 3 * C)
            B, N, C3 = output.shape
            num_heads = self.body.blocks[-1].attn.num_heads
            C = C3 // 3
            qkv = output.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # English q, k, v (shape: B, num_heads, N, head_dim)
            scale = (C // num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attentions.append(attn)

        # English forward hook English qkv
        handle = self.body.blocks[-1].attn.qkv.register_forward_hook(hook)
        with torch.no_grad():
            if isinstance(x, NestedTensor):
                self.body(x.tensors)
            else:
                self.body(x)
        handle.remove()
        return attentions[0]

    def visualize_attention(self, tensor, save_path="dinov2_attention.png"):
        """
        English CLS token English patch English
        tensor: English tensor,English [B, 3, H, W],English
        save_path: English
        """
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np

        # English tensor English
        if isinstance(tensor, NestedTensor):
            actual_tensor = tensor.tensors
        else:
            actual_tensor = tensor

        B, C, H, W = actual_tensor.shape
        patch_size = 14

        # English DINOv2 English,English pad English 14 English
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size

        if pad_h > 0 or pad_w > 0:
            padded_tensor = torch.nn.functional.pad(actual_tensor, (0, pad_w, 0, pad_h))
        else:
            padded_tensor = actual_tensor

        # 1. English (B, num_heads, N, N)
        attn = self.get_last_selfattention(padded_tensor)

        # 2. English CLS token English patch English (B, num_heads, N-1)
        # English 0 English CLS token,1: English patch tokens
        cls_attn = attn[:, :, 0, 1:]

        # 3. English head English (B, N-1)
        cls_attn = cls_attn.mean(dim=1)

        h_featmap = (H + pad_h) // patch_size
        w_featmap = (W + pad_w) // patch_size

        # English 2D English (B, h_featmap, w_featmap)
        num_patches = cls_attn.shape[1]
        if num_patches == h_featmap * w_featmap:
            cls_attn = cls_attn.reshape(B, h_featmap, w_featmap)
        else:
            # English,English fallback
            cls_attn = cls_attn.reshape(B, int(np.sqrt(num_patches)), -1)
            h_featmap, w_featmap = cls_attn.shape[1], cls_attn.shape[2]

        # 4. English(English batch English)
        attn_map = cls_attn[0].cpu().numpy()

        # English 0-255 English
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        attn_map = np.uint8(255 * attn_map)

        # English heatmap English padding English,English padding English
        attn_map_resized = cv2.resize(attn_map, (W + pad_w, H + pad_h))
        attn_map_cropped = attn_map_resized[:H, :W]
        heatmap = cv2.applyColorMap(attn_map_cropped, cv2.COLORMAP_JET)

        # English tensor English(English),English ImageNet English
        img = actual_tensor[0].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        img = np.uint8(255 * img)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # English
        result = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)

        # English
        cv2.imwrite(save_path, result)
        print(f"DINOv2 English: {save_path}")
        return result

    @torch.no_grad()
    def forward_frozen(self, tensor):
        pass # Not used directly

    def forward(self, tensor):
        if isinstance(tensor, NestedTensor):
            actual_tensor = tensor.tensors
        else:
            actual_tensor = tensor

        # We need to compute features with gradients for the last block
        # Use body.forward_features logic but keep gradients
        x = self.body.prepare_tokens_with_masks(actual_tensor)

        for i, blk in enumerate(self.body.blocks):
            if i < len(self.body.blocks) - 1:
                with torch.no_grad():
                    x = blk(x)
            else:
                x = blk(x)

        x_norm = self.body.norm(x)
        features = {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
        }

        if self.return_interm_layers:

            layer1 = features["x_norm_patchtokens"]
            layer2 = features["x_norm_patchtokens"]
            layer3 = features["x_norm_patchtokens"]
            layer4 = features["x_norm_patchtokens"]

            od = OrderedDict()
            od["0"] = layer1.reshape(layer1.shape[0], 22, 16, 384).permute(0, 3, 2, 1)
            od["1"] = layer2.reshape(layer2.shape[0], 22, 16, 384).permute(0, 3, 2, 1)
            od["2"] = layer3.reshape(layer3.shape[0], 22, 16, 384).permute(0, 3, 2, 1)
            od["3"] = layer4.reshape(layer4.shape[0], 22, 16, 384).permute(0, 3, 2, 1)
            return od
        else:
            xs = features["x_norm_patchtokens"]
            od = OrderedDict()
            od["0"] = xs.reshape(xs.shape[0], 22, 16, 384).permute(0, 3, 2, 1)
            return od

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
    position_embedding = build_position_encoding(args)  #English
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks

    if args.backbone == 'dino_v2':
        backbone = DINOv2BackBone()
    else:
        assert args.backbone in ['resnet18', 'resnet34']
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)

    model = Joiner(backbone, position_embedding) # Joiner English,English(English),English
    model.num_channels = backbone.num_channels
    return model
