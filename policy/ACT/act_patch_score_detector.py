"""
Patch-score detector for ACT-trained transformer backbones.

This module follows LaSt-ViT / LazyStrike's core definition:
    patch_score(i) = cosine_similarity(cls_token, patch_token_i)

Unlike `patch_score.py`, which is tied to torchvision's supervised ViT,
this detector works with the ACT project's timm-based ViT / DINOv2 / CLIP
backbones and can load either timm pretrained weights or ACT checkpoints.
"""

import argparse
import csv
import io
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from .artifact_detector import BackboneLoader, LoadedBackbone
except ImportError:
    from artifact_detector import BackboneLoader, LoadedBackbone


@dataclass
class PatchScoreResult:
    """Patch-score inference result for a single image."""

    patch_scores: torch.Tensor          # [B, N]
    patch_grid: Tuple[int, int]
    stats: Dict[str, Any] = field(default_factory=dict)

    def score_map(self) -> torch.Tensor:
        gh, gw = self.patch_grid
        return self.patch_scores.reshape(self.patch_scores.shape[0], gh, gw)

    def summary(self) -> str:
        s = self.stats
        return (
            f"[Patch Score] total_tokens={s.get('total_tokens', -1)}  "
            f"mean={s.get('score_mean', 0.0):.4f}  "
            f"std={s.get('score_std', 0.0):.4f}  "
            f"min={s.get('score_min', 0.0):.4f}  "
            f"max={s.get('score_max', 0.0):.4f}"
        )


class PatchScoreDetector:
    """End-to-end patch-score detector for ACT-compatible transformer backbones."""

    def __init__(self, loaded_backbone: LoadedBackbone):
        self.backbone = loaded_backbone

    @classmethod
    def from_pretrained_timm(
        cls,
        backbone_type: str,
        *,
        image_size: int = 224,
        device: Union[str, torch.device] = "cpu",
    ) -> "PatchScoreDetector":
        return cls(
            BackboneLoader.from_pretrained_timm(
                backbone_type=backbone_type,
                image_size=image_size,
                device=device,
            )
        )

    @classmethod
    def from_act_checkpoint(
        cls,
        ckpt_path: Union[str, os.PathLike],
        backbone_type: str,
        *,
        image_size: int = 224,
        device: Union[str, torch.device] = "cpu",
        backbone_idx: int = 0,
    ) -> "PatchScoreDetector":
        return cls(
            BackboneLoader.from_act_checkpoint(
                ckpt_path=ckpt_path,
                backbone_type=backbone_type,
                image_size=image_size,
                device=device,
                backbone_idx=backbone_idx,
            )
        )

    @torch.no_grad()
    def detect(self, image: Image.Image) -> PatchScoreResult:
        tfm = self.backbone.image_transform
        x = tfm(image.convert("RGB")).unsqueeze(0).to(self.backbone.device)
        cls_token, patch_tokens = self._extract_cls_and_patch_tokens(x)
        patch_scores = torch.cosine_similarity(
            patch_tokens,
            cls_token.unsqueeze(1).expand_as(patch_tokens),
            dim=-1,
        )
        stats = self._build_stats(patch_scores)
        return PatchScoreResult(
            patch_scores=patch_scores.cpu(),
            patch_grid=self.backbone.patch_grid,
            stats=stats,
        )

    @torch.no_grad()
    def _extract_cls_and_patch_tokens(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bb = self.backbone.timm_backbone
        bb.eval()
        feats = bb.forward_features(x.to(self.backbone.device))
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]

        num_prefix = int(
            getattr(bb, "num_prefix_tokens", 0)
            or (1 if hasattr(bb, "cls_token") else 0)
        )
        if feats.ndim != 3:
            raise ValueError(
                f"Expected token features with rank 3, got shape={tuple(feats.shape)}"
            )
        if num_prefix <= 0:
            raise RuntimeError(
                f"{bb.__class__.__name__} has no prefix token; patch score requires a CLS-like token."
            )

        cls_token = F.normalize(feats[:, 0, :].float(), dim=-1)
        patch_tokens = F.normalize(feats[:, num_prefix:, :].float(), dim=-1)
        gh, gw = self.backbone.patch_grid
        expected_n = gh * gw
        if patch_tokens.shape[1] != expected_n:
            raise ValueError(
                f"Patch token count mismatch: expected {expected_n}, got {patch_tokens.shape[1]}"
            )
        return cls_token, patch_tokens

    def _build_stats(self, patch_scores: torch.Tensor) -> Dict[str, Any]:
        flat = patch_scores.reshape(-1)
        top_idx = int(flat.argmax().item())
        gh, gw = self.backbone.patch_grid
        return {
            "total_tokens": int(flat.numel()),
            "score_min": float(flat.min().item()),
            "score_max": float(flat.max().item()),
            "score_mean": float(flat.mean().item()),
            "score_std": float(flat.std().item()),
            "top_patch_index": top_idx,
            "top_patch_row": int(top_idx // gw),
            "top_patch_col": int(top_idx % gw),
        }

    def visualize(
        self,
        image: Image.Image,
        result: PatchScoreResult,
        *,
        save_path: Optional[Union[str, os.PathLike]] = None,
        top_k: Optional[int] = 10,
        show: bool = False,
    ) -> Image.Image:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        sz = self.backbone.image_size
        img = image.convert("RGB").resize((sz, sz), Image.BICUBIC)
        arr = np.asarray(img)
        score_map = result.score_map()[0].numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
        axes[0].imshow(arr)
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")

        im1 = axes[1].imshow(score_map, cmap="hot", interpolation="nearest")
        axes[1].set_title(
            f"Patch Score\nmin={score_map.min():.3f}, max={score_map.max():.3f}",
            fontsize=12,
        )
        axes[1].set_xlabel("Patch X")
        axes[1].set_ylabel("Patch Y")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(arr, alpha=0.6)
        upsampled = np.array(
            Image.fromarray(score_map.astype(np.float32)).resize((sz, sz), Image.BILINEAR)
        )
        im2 = axes[2].imshow(upsampled, cmap="hot", alpha=0.4, interpolation="bilinear")

        if top_k is not None and top_k > 0:
            gh, gw = result.patch_grid
            ph = sz / gh
            pw = sz / gw
            flat = result.patch_scores[0].numpy()
            top_idx = np.argsort(flat)[-int(top_k):]
            for idx in top_idx:
                row = int(idx // gw)
                col = int(idx % gw)
                rect = mpatches.Rectangle(
                    (col * pw, row * ph),
                    pw,
                    ph,
                    linewidth=1.5,
                    edgecolor="cyan",
                    facecolor="none",
                )
                axes[2].add_patch(rect)

        axes[2].set_title("Overlay", fontsize=12)
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buf.seek(0)
        out_img = Image.open(buf).convert("RGB")
        if save_path is not None:
            os.makedirs(os.path.dirname(os.fspath(save_path)) or ".", exist_ok=True)
            out_img.save(save_path)
        if show:
            try:
                out_img.show()
            except Exception:
                pass
        return out_img


def run_patch_score_on_act_ckpt(
    ckpt_path: Union[str, os.PathLike],
    backbone_type: str,
    image: Union[str, os.PathLike, Image.Image],
    *,
    image_size: int = 224,
    device: Union[str, torch.device] = "cpu",
    save_plot_to: Optional[Union[str, os.PathLike]] = None,
    save_stats_csv: Optional[Union[str, os.PathLike]] = None,
) -> PatchScoreResult:
    detector = PatchScoreDetector.from_act_checkpoint(
        ckpt_path=ckpt_path,
        backbone_type=backbone_type,
        image_size=image_size,
        device=device,
    )
    image_pil = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
    result = detector.detect(image_pil)
    if save_plot_to is not None:
        detector.visualize(image_pil, result, save_path=save_plot_to)
    if save_stats_csv is not None:
        os.makedirs(os.path.dirname(os.fspath(save_stats_csv)) or ".", exist_ok=True)
        with open(save_stats_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in result.stats.items():
                writer.writerow([key, value])
    return result


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Patch-score detection for ACT-trained ViT / DINOv2 / CLIP backbones."
    )
    ap.add_argument("--image", required=True, help="Input image path.")
    ap.add_argument("--backbone-type", required=True, choices=["vit", "dinov2", "clip"])
    ap.add_argument("--ckpt", default=None, help="ACT checkpoint path. Omit to use timm pretrained weights.")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default=None, help="Optional output image path.")
    ap.add_argument("--stats-csv", default=None, help="Optional CSV path for patch-score stats.")
    return ap


def main() -> None:
    args = _build_argparser().parse_args()
    image = Image.open(args.image).convert("RGB")
    if args.ckpt:
        detector = PatchScoreDetector.from_act_checkpoint(
            ckpt_path=args.ckpt,
            backbone_type=args.backbone_type,
            image_size=args.image_size,
            device=args.device,
        )
    else:
        detector = PatchScoreDetector.from_pretrained_timm(
            backbone_type=args.backbone_type,
            image_size=args.image_size,
            device=args.device,
        )
    result = detector.detect(image)
    print(result.summary())
    if args.output:
        detector.visualize(image, result, save_path=args.output)
        print(f"saved visualization to: {args.output}")
    if args.stats_csv:
        os.makedirs(os.path.dirname(os.fspath(args.stats_csv)) or ".", exist_ok=True)
        with open(args.stats_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in result.stats.items():
                writer.writerow([key, value])
        print(f"saved stats to: {args.stats_csv}")


if __name__ == "__main__":
    main()
