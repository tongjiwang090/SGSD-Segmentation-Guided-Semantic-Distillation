# -*- coding: utf-8 -*-
"""vit visual head for dp -- replaces dp's randomly initialized resnet18 with a

pretrained vit. why this file exists:
  dp's MultiImageObsEncoder expects rgb_model to be an image -> vector module
  (get_resnet returns a resnet with fc=Identity, still carrying avgpool).
  a vit natively emits a token grid, so it needs a wrapper: pooled vector outside,
  blocks[k] still reachable inside for the distillation hook.

kept matched with the act side, otherwise the two are not comparable:
  same timm model name
  cls/register dropped the same way, by num_prefix_tokens
  distillation mounted at the same place, blocks[k] output, pre-norm
  same unfreeze-spec syntax, e.g. b11,cls,norm
"""
from __future__ import annotations

import re

import timm
import torch
import torch.nn as nn


class ViTPooledBackbone(nn.Module):
    """timm vit -> globally pooled vector. dp's obs_encoder only needs image -> vector.

    Args:
        name: timm model name
        unfreeze: unfreeze spec, same syntax as act's run_freeze_ablation
                  none / b11 / b11,cls,norm / all
        pool: mean | cls -- dp uses mean, matching get_resnet's avgpool semantics
    """

    def __init__(self, name="vit_base_patch14_dinov2.lvd142m",
                 unfreeze="none", pool="mean", pretrained=True):
        super().__init__()
        self.body = timm.create_model(name, pretrained=pretrained,
                                      num_classes=0, dynamic_img_size=True)
        self.num_features = int(self.body.num_features)
        self.pool = pool
        self._apply_unfreeze(unfreeze)

    # ---------------- unfreeze spec ----------------
    def _apply_unfreeze(self, spec: str):
        """same convention as act: freeze everything, then unfreeze by spec.
        no silent fallback -- an unrecognized item raises."""
        for p in self.body.parameters():
            p.requires_grad_(False)
        spec = (spec or "none").strip().lower()
        if spec in ("none", ""):
            return
        if spec == "all":
            for p in self.body.parameters():
                p.requires_grad_(True)
            return
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            m = re.fullmatch(r"b(\d+)", item)
            if m:
                for p in self.body.blocks[int(m.group(1))].parameters():
                    p.requires_grad_(True)
            elif item == "cls":
                if getattr(self.body, "cls_token", None) is not None:
                    self.body.cls_token.requires_grad_(True)
            elif item == "norm":
                for p in self.body.norm.parameters():
                    p.requires_grad_(True)
            else:
                raise ValueError(f"unknown unfreeze item: {item} (spec: {spec})")

    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,3,H,W], already imagenet-normalized by dp's transform -> [B,D]"""
        feats = self.body.forward_features(x)          # [B, 1+N, C]
        npre = int(getattr(self.body, "num_prefix_tokens", 1))
        if self.pool == "cls":
            return feats[:, 0]
        return feats[:, npre:].mean(dim=1)             # global average pool


def get_vit(name="vit_base_patch14_dinov2.lvd142m", unfreeze="none",
            pool="mean", weights=None, **kwargs):
    """factory in the same style as model_getter.get_resnet, for hydra's _target_.

    the weights argument only matches get_resnet's signature; vit pretrained weights
    """
    return ViTPooledBackbone(name=name, unfreeze=unfreeze, pool=pool)
