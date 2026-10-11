# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Backbone modules."""
from collections import OrderedDict
import math
import re
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
    # freeze everything first
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
        
    # unfreeze the last stage
    if backbone_type == "resnet":
        # resnet: the last stage is layer4
        if hasattr(backbone, "layer4"):
            for parameter in backbone.layer4.parameters():
                parameter.requires_grad_(True)
    else:
        # timm vision models (vit, dinov2, clip)
        
        if hasattr(backbone, "cls_token"):
            backbone.cls_token.requires_grad_(True)
            
        # 2. unfreeze every attention module (opens the cls-patch routing) and the last block
        # if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
        #     for block in backbone.blocks:
        #         if hasattr(block, "attn"):
        #             for parameter in block.attn.parameters():
        #                 parameter.requires_grad_(True)
        for parameter in backbone.blocks[-1].parameters():
            parameter.requires_grad_(True)
                
        if hasattr(backbone, "norm"):
            for parameter in backbone.norm.parameters():
                parameter.requires_grad_(True)


def _freeze_last_layer_and_cls(backbone: nn.Module, backbone_type: str):
    # unfreeze everything first
    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
        
    # then freeze the last layer and the cls token
    if backbone_type == "resnet":
        # resnet: freeze layer4
        if hasattr(backbone, "layer4"):
            for parameter in backbone.layer4.parameters():
                parameter.requires_grad_(False)
    else:
        # timm vision models (vit, dinov2, clip)
        
        # 1. freeze the cls token
        if hasattr(backbone, "cls_token"):
            backbone.cls_token.requires_grad_(False)
            
        # 2. freeze the last block
        if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
            for parameter in backbone.blocks[-1].parameters():
                parameter.requires_grad_(False)
                
        if hasattr(backbone, "norm"):
            for parameter in backbone.norm.parameters():
                parameter.requires_grad_(False)


# ---------------------------------------------------------------------------
# general unfreeze spec (freeze-position / substructure ablation)
# ---------------------------------------------------------------------------
# spec syntax (comma-composable, case-insensitive):
#
#   none / ""                 -> freeze everything
#   all                       -> everything trainable
#   first / middle / last     -> b0 / b{n//2} / b{n-1}, whole blocks
#   b<i>        b<i>-<j>      -> whole transformer block, inclusive range, negatives ok
#   cls                       -> cls_token
#   pos                       -> pos_embed
#   patch_embed               -> patch embedding
#   norm                      -> the final LayerNorm, top level, after the blocks
#
#   substructures (optional :block range; omitted means all blocks):
#   attn        attn:b8       attn:b8-11   -> attention inside a block (qkv + proj)
#   mlp         mlp:b8        mlp:b0-2     -> mlp inside a block (fc1 + fc2)
#   ls          ls:b8         ls:b8-11     -> LayerScale（ls1.gamma + ls2.gamma）
#   blocknorm   blocknorm:b8               -> norm1 + norm2 inside a block
#
# examples:
#   "b8"            unfreeze block 8 only
#   "attn"          unfreeze attention in all 12 blocks
#   "mlp:b8"        unfreeze the mlp of block 8 only
#   "ls"            unfreeze every LayerScale (18,432 params)
#   "b11,cls,norm"  unfreeze b11 + cls_token + final LayerNorm
#
# semantics are strict: freeze ALL parameters, then unfreeze exactly the names the spec resolves to.
SUBMODULE_PREFIXES = {
    "attn": ("attn.",),
    "mlp": ("mlp.",),
    "ls": ("ls1.", "ls2."),
    "blocknorm": ("norm1.", "norm2."),
}
TOPLEVEL_PARAMS = {
    "cls": ("cls_token",),
    "pos": ("pos_embed",),
    "patch_embed": ("patch_embed.proj.weight", "patch_embed.proj.bias"),
    "norm": ("norm.weight", "norm.bias"),
}


def _block_range(text, num_blocks, raw):
    """parse 'b8' / 'b8-11' / 'b-1' into a list of indices."""
    m = re.fullmatch(r"b(-?\d+)(?:-(-?\d+))?", text)
    if not m:
        raise ValueError(f"cannot parse block range '{text}' (full spec: '{raw}')")

    def idx(tok):
        i = int(tok)
        if i < 0:
            i += num_blocks
        if not 0 <= i < num_blocks:
            raise ValueError(f"block index out of range '{text}': {i} (num_blocks={num_blocks}, spec='{raw}')")
        return i

    lo = idx(m.group(1))
    hi = idx(m.group(2)) if m.group(2) is not None else lo
    if hi < lo:
        lo, hi = hi, lo
    return list(range(lo, hi + 1))


def parse_unfreeze_spec(spec, num_blocks):
    """parse a spec into (whole_blocks, toplevel_keys, submodule_targets).

    whole_blocks       : set[int]                     block indices to unfreeze whole
    toplevel_keys      : set[str]                     keys of TOPLEVEL_PARAMS
    submodule_targets  : set[tuple[str, int]]         (substructure name, block index)
    """
    whole, top, sub = set(), set(), set()
    if spec is None:
        raise ValueError("unfreeze spec is None")
    text = str(spec).strip().lower()
    if text in ("", "none"):
        return whole, top, sub
    if text == "all":
        return set(range(num_blocks)), set(TOPLEVEL_PARAMS), {
            (k, i) for k in SUBMODULE_PREFIXES for i in range(num_blocks)
        }

    for part in (t.strip() for t in text.split(",") if t.strip()):
        if part == "first":
            whole.add(0)
        elif part == "last":
            whole.add(num_blocks - 1)
        elif part == "middle":
            whole.add(num_blocks // 2)
        elif part in TOPLEVEL_PARAMS:
            top.add(part)
        elif part in SUBMODULE_PREFIXES:
            sub.update((part, i) for i in range(num_blocks))
        elif ":" in part:
            name, _, rng = part.partition(":")
            if name not in SUBMODULE_PREFIXES:
                raise ValueError(f"unknown substructure '{name}' (full spec: '{spec}')")
            sub.update((name, i) for i in _block_range(rng, num_blocks, spec))
        else:
            whole.update(_block_range(part, num_blocks, spec))
    return whole, top, sub


def resolve_unfreeze_param_names(body, spec):
    """resolve a spec into the set of parameter NAMES to unfreeze on body.

    apply and verify share this function so the two can never drift apart.
    """
    # 'all' means literally every parameter; skip the grouped enumeration.
    # grouped enumeration relies on the hardcoded TOPLEVEL_PARAMS table and misses
    # backbone-specific modules -- clip's norm_pre was missed exactly this way
    # (149/151 tensors: 'full fine-tuning' was not full, and 99.998% rounded to 100.00%).
    if isinstance(spec, str) and spec.strip().lower() == "all":
        return {n for n, _ in body.named_parameters()}

    num_blocks = len(body.blocks)
    whole, top, sub = parse_unfreeze_spec(spec, num_blocks)
    all_names = {n for n, _ in body.named_parameters()}
    picked = set()

    for i in whole:
        pref = f"blocks.{i}."
        picked.update(n for n in all_names if n.startswith(pref))
    for key in top:
        for cand in TOPLEVEL_PARAMS[key]:
            if cand in all_names:
                picked.add(cand)
    for name, i in sub:
        for pref in SUBMODULE_PREFIXES[name]:
            full = f"blocks.{i}.{pref}"
            picked.update(n for n in all_names if n.startswith(full))
    return picked


class LoRALinear(nn.Module):
    """attach a low-rank adapter to an existing nn.Linear: y = W x + (B A x) * (alpha/r).

    lora_B starts at 0 -> at injection the output is bit-identical to the original model.
    this has to hold, or the lora baseline already departs from pretraining at step 0
    """

    def __init__(self, base, r, alpha=None):
        super().__init__()
        if r <= 0:
            raise ValueError(f"lora rank must be positive, got {r}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.lora_A = nn.Parameter(torch.zeros(self.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = (float(alpha) if alpha is not None else float(self.r)) / self.r

    def forward(self, x):
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return self.base(x) + delta * self.scaling

    @torch.no_grad()
    def merged_weight(self):
        """W + scaling * B @ A -- for folding."""
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


LORA_TARGETS = ("qkv", "proj")


def apply_lora_policy(body, r):
    """lora baseline: backbone fully frozen, adapters only on attn.qkv / attn.proj per block."""
    for prm in body.parameters():
        prm.requires_grad_(False)
    injected = []
    for i, blk in enumerate(body.blocks):
        for attr in LORA_TARGETS:
            lin = getattr(blk.attn, attr, None)
            if not isinstance(lin, nn.Linear):
                raise RuntimeError(
                    f"blocks.{i}.attn.{attr} is not nn.Linear (got {type(lin)}); "
                    f"the lora injection-point assumption does not hold, check the backbone")
            setattr(blk.attn, attr, LoRALinear(lin, r))
            injected.append(f"blocks.{i}.attn.{attr}")
    # train() uses this to decide which blocks go into train mode; lora touches all of them
    body._unfrozen_blocks = set(range(len(body.blocks)))
    body._unfrozen_param_names = {n for n, p in body.named_parameters()
                                  if p.requires_grad}
    trainable = sum(p.numel() for p in body.parameters() if p.requires_grad)
    total = sum(p.numel() for p in body.parameters())
    base_total = total - trainable
    summary = {
        "spec": f"lora{r}",
        "num_blocks": len(body.blocks),
        "num_param_tensors": len(body._unfrozen_param_names),
        "touched_blocks": sorted(body._unfrozen_blocks),
        "trainable_params": trainable,
        "total_params": base_total,
        "trainable_pct": 100.0 * trainable / base_total,
        "lora_rank": r,
        "lora_targets": injected[:2] + ["..."] if len(injected) > 2 else injected,
    }
    print(f"[unfreeze] lora r={r}: injected {len(injected)} adapters, "
          f"trainable {trainable:,} / backbone {base_total:,} = "
          f"{summary['trainable_pct']:.2f}%", flush=True)
    return summary


def merge_lora_into_base(body):
    """fold every LoRALinear back into a plain nn.Linear -- the state_dict then has standard shapes.

    y = Wx + s*BAx = (W + s*BA)x is exact, so evaluation needs no change at all,
    which is also what makes 'no extra cost at inference' literally true.
    """
    n = 0
    for blk in body.blocks:
        for attr in LORA_TARGETS:
            mod = getattr(blk.attn, attr, None)
            if isinstance(mod, LoRALinear):
                base = mod.base
                base.weight.data.copy_(mod.merged_weight())
                setattr(blk.attn, attr, base)
                n += 1
    return n


def apply_unfreeze_policy(body, backbone_type, spec):
    """apply an unfreeze spec to a timm vit body, returning a readable summary."""
    if backbone_type == "resnet":
        raise NotImplementedError("unfreeze_blocks currently supports only vit-family backbones (vit/dinov2/clip)")

    # lora takes a separate path: it INJECTS new modules rather than unfreezing existing
    # parameters, so the actual == trainable assertion below does not apply to it.
    _m = re.fullmatch(r"lora(\d+)", str(spec).strip().lower())
    if _m:
        return apply_lora_policy(body, int(_m.group(1)))

    names = resolve_unfreeze_param_names(body, spec)

    # 1) freeze everything  2) unfreeze exactly the resolved names
    for prm in body.parameters():
        prm.requires_grad_(False)
    for n, prm in body.named_parameters():
        if n in names:
            prm.requires_grad_(True)

    # train() needs to know which blocks hold trainable params (bn/dropout mode)
    touched_blocks = set()
    for n in names:
        m = re.match(r"blocks\.(\d+)\.", n)
        if m:
            touched_blocks.add(int(m.group(1)))
    body._unfrozen_blocks = touched_blocks
    body._unfrozen_param_names = names

    trainable = sum(p.numel() for n, p in body.named_parameters() if n in names)
    total = sum(p.numel() for p in body.parameters())
    actual = sum(p.numel() for p in body.parameters() if p.requires_grad)
    assert actual == trainable, f"unfreeze mismatch: expected {trainable}, got {actual}"

    summary = {
        "spec": str(spec),
        "num_blocks": len(body.blocks),
        "num_param_tensors": len(names),
        "touched_blocks": sorted(touched_blocks),
        "trainable_params": trainable,
        "total_params": total,
    }
    print(f"[backbone] unfreeze spec='{spec}' -> {len(names)} tensors, blocks touched="
          f"{sorted(touched_blocks)} | trainable {trainable:,} / {total:,} "
          f"({100.0 * trainable / max(total, 1):.2f}%)")
    return summary


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
        # torchvision resnet is pretrained on imagenet -- this is the fact, not a fallback
        self.norm_mean = (0.485, 0.456, 0.406)
        self.norm_std = (0.229, 0.224, 0.225)


class TimmVisionBackbone(nn.Module):

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        pretrained: bool,
        image_size: Optional[Tuple[int, int]],
        freeze_last_layer_and_cls: bool = False,
        unfreeze_spec: Optional[str] = None,
    ):
        super().__init__()
        self.train_backbone = train_backbone
        self.freeze_last_layer_and_cls = freeze_last_layer_and_cls
        self.unfreeze_spec = unfreeze_spec
        if timm is None:
            raise ImportError("timm is required for ViT, DINOv2 and CLIP backbones")
        self.body = timm.create_model(name, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        self.num_channels = int(getattr(self.body, "num_features"))
        # each backbone declares its own normalization; the three differ:
        #   dinov2 (0.485,0.456,0.406)/(0.229,0.224,0.225)  happens to equal imagenet
        #   vit    (0.5,0.5,0.5)/(0.5,0.5,0.5)
        #   clip   (0.4815,0.4578,0.4082)/(0.2686,0.2613,0.2758)
        # index directly, no .get(): a cfg without mean/std should fail here, not silently use imagenet
        _cfg = self.body.pretrained_cfg
        self.norm_mean = tuple(_cfg["mean"])
        self.norm_std = tuple(_cfg["std"])
        self.patch_size = self._normalize_patch_size(getattr(self.body.patch_embed, "patch_size", 16))
        default_size = getattr(self.body, "img_size", None)
        if default_size is None:
            default_size = getattr(self.body.patch_embed, "img_size", None)
        self.image_size = _to_hw(image_size) if image_size is not None else _to_hw(default_size)
        _freeze_backbone(self.body, train_backbone)

        # ---- inference-time artifact intervention (causal experiment) ----
        # lazy: with ACT_INTERVENE_SITE unset this does nothing and the existing path is untouched.
        # once set it must be valid (parse_env raises KeyError on missing args); no silent fallback.
        if os.environ.get("ACT_INTERVENE_SITE"):
            _act_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
            if _act_dir not in sys.path:
                sys.path.insert(0, _act_dir)
            from artifact_intervention import (LOCALIZED, install as _iv_install,
                                              parse_env as _iv_parse)
            _iv_cfg = _iv_parse()
            self._intervene_localized = _iv_cfg["mode"] in LOCALIZED
            if self._intervene_localized:
                # the teacher is trained on imagenet statistics. only when this backbone
                # uses the same can the tensor after _prepare_input go straight to it.
                # anything else must be de-normalized first -- fail here rather than
                _IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                _cur = (tuple(round(v, 4) for v in self.norm_mean),
                        tuple(round(v, 4) for v in self.norm_std))
                if _cur != _IMAGENET:
                    raise ValueError(
                        f"localized intervention needs teacher-predicted foreground, and the teacher "
                        f"is trained on imagenet statistics; this backbone uses {_cur}, so feeding it "
                        f"directly yields a wrong mask. de-normalize then re-normalize to imagenet.")
            self._intervene_handle, _iv_desc = _iv_install(self.body, _iv_cfg)
            print(f"[intervene] inference intervention attached: {_iv_desc}", flush=True)

    def train(self, mode: bool = True):
        super().train(mode)
        if not getattr(self, "train_backbone", False):
            self.body.eval()
        elif getattr(self, "unfreeze_spec", None) is not None:
            # freeze-position ablation: only unfrozen blocks enter train mode
            self.body.eval()
            for i in sorted(getattr(self.body, "_unfrozen_blocks", set())):
                self.body.blocks[i].train(mode)
            if any(n.startswith("norm.") for n in getattr(self.body, "_unfrozen_param_names", ())) \
                    and hasattr(self.body, "norm"):
                self.body.norm.train(mode)
        else:
            self.body.eval()
            if self.freeze_last_layer_and_cls:
                # with the last layer and cls frozen, train everything else
                for i, block in enumerate(self.body.blocks):
                    if i != len(self.body.blocks) - 1:
                        block.train(mode)
            else:
                # original behaviour: train the last layer only
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
        fm = patch_tokens.transpose(1, 2).reshape(
            features.shape[0], features.shape[2], grid_h, grid_w)
        # readout spatial-resolution switch: reduces only what the POLICY can see.
        # the distillation target goes through the blocks[k] forward hook and never passes
        # here, so the loss needs no change. with ACT_READOUT_POOL unset this is a no-op.
        _pool = os.environ.get("ACT_READOUT_POOL")
        if _pool:
            fm = F.adaptive_avg_pool2d(fm, int(_pool))
        return fm

    def forward(self, tensor):
        tensor = self._prepare_input(tensor)
        if getattr(self, "_intervene_localized", False):
            # localized intervention: teacher-predicted foreground, downsampled to the patch
            # grid, consumed by the blocks[k] hook. __init__ already asserted imagenet stats.
            from artifact_intervention import _MASK_SLOT, predict_fg
            _side = tensor.shape[-1] // self.patch_size[1]
            _MASK_SLOT["fg"] = predict_fg(tensor, _side)
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
    unfreeze_spec = _get_arg(args, "unfreeze_blocks", None)
    if isinstance(unfreeze_spec, str) and unfreeze_spec.strip() == "":
        unfreeze_spec = None

    if backbone_type == "resnet":
        if unfreeze_spec is not None:
            raise NotImplementedError("unfreeze_blocks currently supports only vit-family backbones (vit/dinov2/clip)")
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
            unfreeze_spec=unfreeze_spec,
        )
        if unfreeze_spec is not None:
            # an explicit unfreeze spec wins, independent of train_backbone (the spec decides)
            backbone.unfreeze_summary = apply_unfreeze_policy(backbone.body, backbone_type, unfreeze_spec)
            backbone.train_backbone = bool(backbone.unfreeze_summary["trainable_params"] > 0)
        elif train_backbone:
            if freeze_last_layer_and_cls:
                _freeze_last_layer_and_cls(backbone.body, backbone_type)
            else:
                _freeze_except_last_layer(backbone.body, backbone_type)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
