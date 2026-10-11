# -*- coding: utf-8 -*-
"""distillation hook-in for dp -- the same loss as the act side, so the two policy

architectures stay comparable. matched conventions (any mismatch and dp/act are not):
  teacher  up4 output of the same attention-unet, H,W kept at 224
           (a downsampled teacher was tried and failed)
  student  blocks[k] output, pre-norm, cls/register dropped by num_prefix_tokens
  align    student token grid bilinearly upsampled to the teacher's 224x224

memory constraint, dp-specific, forces subsampling:
  dp runs batch 128 x n_obs_steps 3 = 384 images per step; the teacher emits
  768x224x224 = 154MB each -> 59GB, which does not fit. act runs 8 x 3 cameras
  = 24 images per step (3.7GB). each step therefore draws K images at random out
  of B*T for the distillation loss; the task loss still uses all of them. known
  dp/act difference, belongs in the method section. K defaults to 16, ~5GB.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F


class DPDistiller:
    """attaches to the vit visual head, captures the input image and blocks[k] features, computes the distillation loss on demand."""

    def __init__(self, vit_module, teacher, mount_block=11,
                 w_mse=1.0, w_cos=0.5, w_mag=0.5, n_sample=16, device="cuda"):
        self.vit = vit_module
        self.teacher = teacher
        self.mount = mount_block
        self.w_mse, self.w_cos, self.w_mag = w_mse, w_cos, w_mag
        self.n_sample = n_sample
        self.device = device
        self._img = None          # input image as the visual head sees it (imagenet-normalized)
        self._feat = None         # blocks[k] output
        self._t_buf = {}
        self._attach()

    # ---------------- hooks ----------------
    def _attach(self):
        def pre_hook(_m, inputs):
            # capture the tensor the visual head actually sees -- already through dp's
            # resize and imagenet norm, matching the teacher, so nothing is guessed
            self._img = inputs[0]

        def blk_hook(_m, _i, out):
            self._feat = out

        self.vit.register_forward_pre_hook(pre_hook)
        self.vit.body.blocks[self.mount].register_forward_hook(blk_hook)
        self.teacher.up4.register_forward_hook(
            lambda _m, _i, o: self._t_buf.__setitem__("up4", o))

    # ---------------- loss ----------------
    def loss(self):
        """call after one policy forward. returns (scalar loss, per-term dict)."""
        if self._img is None or self._feat is None:
            raise RuntimeError("no student features -- check the vit visual head was actually run")

        img, feat = self._img, self._feat
        self._img = self._feat = None            # clear now; holding the graph ooms

        n = img.shape[0]
        k = min(self.n_sample, n)
        idx = torch.randperm(n, device=img.device)[:k]
        img_s, feat_s = img[idx], feat[idx]

        # student: drop prefix -> [k, C, g, g] -> upsample to teacher resolution
        npre = int(getattr(self.vit.body, "num_prefix_tokens", 1))
        pt = feat_s[:, npre:, :]
        side = int(round(pt.shape[1] ** 0.5))
        if side * side != pt.shape[1]:
            raise RuntimeError(f"non-square patch grid: N={pt.shape[1]}")
        s = pt.transpose(1, 2).reshape(k, pt.shape[2], side, side)

        # teacher: no gradient
        with torch.no_grad():
            ti = img_s if img_s.shape[-2:] == (224, 224) else \
                F.interpolate(img_s, size=(224, 224), mode="bilinear", align_corners=False)
            self.teacher(ti)
            t = self._t_buf["up4"].detach()

        s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)

        mse = F.mse_loss(s, t)
        s_n, t_n = F.normalize(s, dim=1), F.normalize(t, dim=1)
        cos = 1 - (s_n * t_n).sum(dim=1).mean()
        mag = F.mse_loss(torch.norm(s, dim=1), torch.norm(t, dim=1))
        total = self.w_mse * mse + self.w_cos * cos + self.w_mag * mag
        return total, {"distill_mse": mse.item(), "distill_cos": cos.item(),
                       "distill_mag": mag.item(), "distill_n": k}


def build_teacher(device="cuda"):
    """load the same attention-unet teacher as the act side. fail loudly if absent."""
    import glob
    cands = glob.glob("/root/autodl-tmp/RoboTwin/**/best_unet_model.pth", recursive=True)
    if not cands:
        raise FileNotFoundError("best_unet_model.pth not found")
    ck_path = cands[0]
    seg_dir = os.path.dirname(os.path.dirname(ck_path))
    if seg_dir not in sys.path:
        sys.path.insert(0, seg_dir)
    from unet import UNet
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    ncls = ck.get("num_classes", 1) if isinstance(ck, dict) else 1
    t = UNet(n_channels=3, n_classes=ncls)
    t.load_state_dict(ck.get("model_state_dict", ck))
    t.eval().to(device).requires_grad_(False)
    return t
