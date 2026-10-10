import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Import the actual ACT backbone factory
try:
    from detr.models.backbone import build_backbone, TimmVisionBackbone
except ImportError:
    from policy.ACT.detr.models.backbone import build_backbone, TimmVisionBackbone


BACKBONE_SPECS: Dict[str, dict] = {
    "vit": {
        "backbone_type": "vit",
        "backbone": "vit_base_patch16_224",
        "image_size": (224, 224),
    },
    "dinov2": {
        "backbone_type": "dinov2",
        "backbone": "vit_base_patch14_dinov2",
        "image_size": (224, 224),
    },
    "clip": {
        "backbone_type": "clip",
        "backbone": "vit_base_patch16_clip_224",
        "image_size": (224, 224),
    },
    "resnet18": {
        "backbone_type": "resnet",
        "backbone": "resnet18",
        "image_size": (480, 640),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize heatmaps for ACT backbones from specifically trained checkpoints."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--ckpts",
        nargs="+",
        required=True,
        help="Checkpoints in format alias=ckpt_path (e.g. clip=/path/to/policy_best.ckpt vit=...)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path("heatmap_outputs")),
        help="Directory to save output heatmaps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--target_ckpt",
        type=str,
        default=None,
        help="Specific ckpt file name to look for in the directories (e.g., policy_epoch_4000_seed_0.ckpt).",
    )
    return parser.parse_args()


def parse_checkpoints(ckpt_args: List[str]) -> Dict[str, str]:
    ckpt_dict = {}
    for item in ckpt_args:
        if "=" not in item:
            raise ValueError(f"Checkpoint argument must be in format alias=path, got: {item}")
        alias, path = item.split("=", 1)
        alias = alias.strip().lower()
        if alias not in BACKBONE_SPECS:
            raise ValueError(f"Unknown alias '{alias}'. Available aliases: {list(BACKBONE_SPECS.keys())}")
        ckpt_dict[alias] = path.strip()
    return ckpt_dict


def get_transform(backbone_wrapper) -> Tuple[T.Compose, Tuple[int, int]]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image_size = (480, 640)

    if isinstance(backbone_wrapper, TimmVisionBackbone):
        cfg = backbone_wrapper.body.pretrained_cfg
        mean = cfg.get("mean", mean)
        std = cfg.get("std", std)
        input_size = cfg.get("input_size", (3, 224, 224))
        image_size = (int(input_size[1]), int(input_size[2]))

        if backbone_wrapper.image_size is not None:
            image_size = backbone_wrapper.image_size

    transform = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return transform, image_size


def compute_act_heatmap(model, image_tensor: torch.Tensor) -> np.ndarray:
    backbone_wrapper = model[0]

    # Transformer backbones: use LaSt-ViT patch score = cos(CLS, patch)
    if isinstance(backbone_wrapper, TimmVisionBackbone):
        with torch.no_grad():
            features = backbone_wrapper.body.forward_features(image_tensor)

        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.ndim != 3:
            raise ValueError(f"Expected token features with rank 3, got {features.ndim}")

        num_prefix_tokens = int(getattr(backbone_wrapper.body, "num_prefix_tokens", 0))
        if num_prefix_tokens <= 0:
            raise RuntimeError(
                f"{backbone_wrapper.body.__class__.__name__} has no CLS-like token; cannot compute patch score."
            )

        cls_token = torch.nn.functional.normalize(features[:, 0, :].float(), dim=-1)
        patch_tokens = torch.nn.functional.normalize(features[:, num_prefix_tokens:, :].float(), dim=-1)
        scores = torch.cosine_similarity(
            patch_tokens,
            cls_token.unsqueeze(1).expand_as(patch_tokens),
            dim=-1,
        )
        grid_h = image_tensor.shape[-2] // backbone_wrapper.patch_size[0]
        grid_w = image_tensor.shape[-1] // backbone_wrapper.patch_size[1]
        heatmap = scores.reshape(grid_h, grid_w).detach().cpu().numpy()
    else:
        # ResNet fallback: keep the old feature-norm visualization.
        with torch.no_grad():
            features, _ = model(image_tensor)
        feature_map = features[0]
        if feature_map.ndim != 4:
            raise ValueError(f"Expected feature map with rank 4, got {feature_map.ndim}")
        heatmap = feature_map.norm(dim=1)[0].detach().cpu().numpy()

    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()
    if heatmap_max > heatmap_min:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap = heatmap - heatmap_min

    return heatmap


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.6, colored, 0.4, 0)
    return heatmap_resized, overlay


def save_visualization(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    title: str,
    save_path: Path,
    map_label: str = "Patch Score",
):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"{title} {map_label}")
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def load_backbone_from_ckpt(model, ckpt_path: str, target_ckpt: str = None):
    print(f"Loading weights from {ckpt_path}...")

    # Check if the provided path is a directory
    if os.path.isdir(ckpt_path):
        if target_ckpt:
            specific_ckpt = os.path.join(ckpt_path, target_ckpt)
            if os.path.exists(specific_ckpt):
                ckpt_path = specific_ckpt
            else:
                raise FileNotFoundError(f"Cannot find specified checkpoint at {specific_ckpt}")
        else:
            best_ckpt = os.path.join(ckpt_path, "policy_best.ckpt")
            last_ckpt = os.path.join(ckpt_path, "policy_last.ckpt")
            if os.path.exists(best_ckpt):
                ckpt_path = best_ckpt
            elif os.path.exists(last_ckpt):
                ckpt_path = last_ckpt
            else:
                raise FileNotFoundError(f"Cannot find checkpoint at {ckpt_path} or policy_best.ckpt inside it.")
    elif not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Actually loading from file: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Check for multiple possible prefixes depending on how the model was wrapped during training
    # For Joiner: model -> backbones -> 0 -> body
    # But inside the ckpt, it might just be "backbones.0.body" or "model.backbones.0.body"
    backbone_state_dict = {}
    for k, v in state_dict.items():
        if "backbones.0." in k:
            # Extract the actual parameter name that belongs to the TimmVisionBackbone / ResNet wrapper
            # Find the position of "backbones.0." and take everything after it
            idx = k.find("backbones.0.")
            new_k = k[idx + len("backbones.0."):]
            backbone_state_dict[new_k] = v

    if not backbone_state_dict:
        print(f"Warning: No weights containing 'backbones.0.' found in {ckpt_path}. Model will use default weights.")
        print(f"First 10 keys in checkpoint: {list(state_dict.keys())[:10]}")
        return model

    missing, unexpected = model.load_state_dict(backbone_state_dict, strict=False)
    print(f"Loaded {len(backbone_state_dict)} backbone tensors.")
    return model


def visualize_model(alias: str, ckpt_path: str, target_ckpt: str, image_path: Path, output_dir: Path, device: torch.device):
    spec = BACKBONE_SPECS[alias]

    # Use standard args for building backbone
    args = SimpleNamespace(
        hidden_dim=512,  # typical for ACT
        position_embedding="sine",
        lr_backbone=0,
        masks=False,
        dilation=False,
        backbone_type=spec["backbone_type"],
        backbone=spec["backbone"],
        backbone_pretrained=False, # We will load our own weights!
        backbone_image_size=spec["image_size"],
    )

    print(f"\n[{alias}] Building ACT backbone '{spec['backbone']}'...")
    model = build_backbone(args)

    # Load trained weights
    try:
        model = load_backbone_from_ckpt(model, ckpt_path, target_ckpt)
    except Exception as e:
        print(f"Error loading checkpoint for {alias}: {e}")
        print("Skipping visualization for this model.")
        return

    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image)

    # model[0] is the backbone wrapper (ResNetBackbone or TimmVisionBackbone)
    transform, image_size = get_transform(model[0])
    image_tensor = transform(image).unsqueeze(0).to(device)

    print(f"[{alias}] Forwarding image tensor of shape {list(image_tensor.shape)}...")
    heatmap = compute_act_heatmap(model, image_tensor)

    print(f"[{alias}] Extracted feature map heatmap of shape {list(heatmap.shape)}")
    heatmap_resized, overlay = overlay_heatmap(image_rgb, heatmap)

    save_path = output_dir / f"{image_path.stem}_trained_{alias}_heatmap.png"
    title = f"Trained ACT {alias} ({image_size[0]}x{image_size[1]})"
    map_label = "Feature Norm" if alias == "resnet18" else "Patch Score"
    save_visualization(image_rgb, heatmap_resized, overlay, title, save_path, map_label=map_label)
    print(f"[{alias}] Saved: {save_path}")


def main():
    args = parse_args()
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ckpt_dict = parse_checkpoints(args.ckpts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[image] {image_path}")
    print(f"[device] {device}")
    print(f"[output_dir] {output_dir.resolve()}")

    for alias, ckpt_path in ckpt_dict.items():
        visualize_model(alias, ckpt_path, args.target_ckpt, image_path, output_dir, device)

    print("\n[done] Trained ACT backbone heatmap visualization finished.")


if __name__ == "__main__":
    main()
