import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image


BACKBONE_SPECS: Dict[str, str] = {
    "vit": "vit_base_patch16_224",
    "dinov2": "vit_base_patch14_dinov2",
    "clip": "vit_base_patch16_clip_224",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize heatmaps for ViT, DINOv2 and CLIP backbones."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["vit", "dinov2", "clip"],
        help="Backbone aliases to visualize: vit, dinov2, clip, or all.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path("policy") / "ACT" / "heatmap_outputs"),
        help="Directory to save output heatmaps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    return parser.parse_args()


def expand_model_aliases(model_args: Iterable[str]) -> List[str]:
    aliases: List[str] = []
    for item in model_args:
        name = item.strip().lower()
        if name == "all":
            aliases.extend(BACKBONE_SPECS.keys())
        elif name in BACKBONE_SPECS:
            aliases.append(name)
        else:
            raise ValueError(f"Unsupported model alias: {item}")
    return list(dict.fromkeys(aliases))


def build_transform(model) -> Tuple[T.Compose, Tuple[int, int]]:
    cfg = model.pretrained_cfg
    input_size = cfg.get("input_size", (3, 224, 224))
    image_size = (int(input_size[1]), int(input_size[2]))
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    transform = T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return transform, image_size


def compute_patch_heatmap(model, image_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        features = model.forward_features(image_tensor)

    if isinstance(features, (list, tuple)):
        features = features[-1]

    if features.ndim != 3:
        raise ValueError(f"Expected token features with rank 3, got shape {tuple(features.shape)}")

    num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 0))
    if num_prefix_tokens <= 0:
        raise RuntimeError(
            f"{model.__class__.__name__} has no CLS-like prefix token; cannot compute patch score."
        )

    cls_token = torch.nn.functional.normalize(features[:, 0, :].float(), dim=-1)
    patch_tokens = torch.nn.functional.normalize(features[:, num_prefix_tokens:, :].float(), dim=-1)
    patch_scores = torch.cosine_similarity(
        patch_tokens,
        cls_token.unsqueeze(1).expand_as(patch_tokens),
        dim=-1,
    )

    patch_size = getattr(model.patch_embed, "patch_size", 16)
    if isinstance(patch_size, tuple):
        patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    else:
        patch_h = patch_w = int(patch_size)

    grid_h = image_tensor.shape[-2] // patch_h
    grid_w = image_tensor.shape[-1] // patch_w
    heatmap = patch_scores.reshape(grid_h, grid_w).detach().cpu().numpy()
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.6, colored, 0.4, 0)
    return heatmap_resized, overlay


def save_visualization(image_rgb: np.ndarray, heatmap: np.ndarray, overlay: np.ndarray, title: str, save_path: Path):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"{title} Patch Score")
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def visualize_model(alias: str, image_path: Path, output_dir: Path, device: torch.device):
    model_name = BACKBONE_SPECS[alias]
    model = timm.create_model(model_name, pretrained=True, num_classes=0, dynamic_img_size=True)
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image)
    transform, image_size = build_transform(model)
    image_tensor = transform(image).unsqueeze(0).to(device)

    heatmap = compute_patch_heatmap(model, image_tensor)
    heatmap_resized, overlay = overlay_heatmap(image_rgb, heatmap)

    save_path = output_dir / f"{image_path.stem}_{alias}_heatmap.png"
    save_visualization(image_rgb, heatmap_resized, overlay, f"{alias} ({image_size[0]}x{image_size[1]})", save_path)
    print(f"[saved] {alias}: {save_path}")


def main():
    args = parse_args()
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    aliases = expand_model_aliases(args.models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[image] {image_path}")
    print(f"[device] {device}")
    print(f"[output_dir] {output_dir.resolve()}")

    for alias in aliases:
        visualize_model(alias, image_path, output_dir, device)

    print("[done] heatmap visualization finished.")


if __name__ == "__main__":
    main()
