import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Import the actual ACT backbone factory
from detr.models.backbone import build_backbone, TimmVisionBackbone


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


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()
    if heatmap_max > heatmap_min:
        return (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    return heatmap - heatmap_min


def compute_feature_heatmap(feature_map: torch.Tensor) -> np.ndarray:
    feature_map = feature_map.detach()

    if feature_map.ndim != 4:
        raise ValueError(f"Expected feature map with rank 4, got {feature_map.ndim}")

    # Use channel-wise L2 norm to summarize activation strength per spatial position.
    feature_norm = feature_map.norm(dim=1)
    return normalize_heatmap(feature_norm[0].cpu().numpy())


def extract_timm_feature_and_cls_heatmap(
    backbone_wrapper: TimmVisionBackbone,
    image_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    prepared_tensor = backbone_wrapper._prepare_input(image_tensor)
    timm_body = backbone_wrapper.body
    cls_heatmap = None
    attentions = []

    def hook(module, inputs, output):
        if output.ndim != 3:
            return

        batch_size, num_tokens, channels_times_three = output.shape
        if channels_times_three % 3 != 0:
            return

        num_heads = getattr(timm_body.blocks[-1].attn, "num_heads", None)
        if not num_heads:
            return

        embed_dim = channels_times_three // 3
        head_dim = embed_dim // num_heads
        qkv = output.reshape(batch_size, num_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attentions.append(attn.softmax(dim=-1))

    handle = None
    if hasattr(timm_body, "blocks") and len(timm_body.blocks) > 0 and hasattr(timm_body.blocks[-1].attn, "qkv"):
        handle = timm_body.blocks[-1].attn.qkv.register_forward_hook(hook)

    try:
        with torch.no_grad():
            features_raw = timm_body.forward_features(prepared_tensor)
    finally:
        if handle is not None:
            handle.remove()

    feature_map = backbone_wrapper._to_feature_map(features_raw, prepared_tensor.shape[-2:])

    if attentions:
        attn = attentions[-1]
        num_prefix_tokens = int(getattr(timm_body, "num_prefix_tokens", 1))
        cls_to_patch = attn[:, :, 0, num_prefix_tokens:]
        grid_h = prepared_tensor.shape[-2] // backbone_wrapper.patch_size[0]
        grid_w = prepared_tensor.shape[-1] // backbone_wrapper.patch_size[1]
        expected_tokens = grid_h * grid_w

        if cls_to_patch.shape[-1] == expected_tokens:
            cls_map = cls_to_patch.mean(dim=1)[0].reshape(grid_h, grid_w).detach().cpu().numpy()
            cls_heatmap = normalize_heatmap(cls_map)
        else:
            print(
                f"Warning: CLS attention token count mismatch, "
                f"expected {expected_tokens}, got {cls_to_patch.shape[-1]}"
            )

    return feature_map, cls_heatmap


def compute_act_heatmaps(model, image_tensor: torch.Tensor) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    backbone_wrapper = model[0]

    if isinstance(backbone_wrapper, TimmVisionBackbone):
        feature_map, cls_heatmap = extract_timm_feature_and_cls_heatmap(backbone_wrapper, image_tensor)
    else:
        with torch.no_grad():
            # model is Joiner, returns features (List[Tensor]) and positions (List[Tensor])
            features, _ = model(image_tensor)
        feature_map = features[0]
        cls_heatmap = None

    return compute_feature_heatmap(feature_map), cls_heatmap


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
    cls_heatmap: Optional[np.ndarray] = None,
    cls_overlay: Optional[np.ndarray] = None,
):
    panels = [
        ("Original", image_rgb, None),
        (f"{title} Heatmap", heatmap, "jet"),
        ("Backbone Overlay", overlay, None),
    ]
    if cls_heatmap is not None and cls_overlay is not None:
        panels.extend([
            ("CLS Attention Heatmap", cls_heatmap, "jet"),
            ("CLS Attention Overlay", cls_overlay, None),
        ])

    plt.figure(figsize=(5 * len(panels), 5))

    for idx, (panel_title, panel_image, cmap) in enumerate(panels, start=1):
        plt.subplot(1, len(panels), idx)
        if cmap is None:
            plt.imshow(panel_image)
        else:
            plt.imshow(panel_image, cmap=cmap)
        plt.title(panel_title)
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
    heatmap, cls_heatmap = compute_act_heatmaps(model, image_tensor)

    print(f"[{alias}] Extracted feature map heatmap of shape {list(heatmap.shape)}")
    heatmap_resized, overlay = overlay_heatmap(image_rgb, heatmap)
    cls_heatmap_resized = None
    cls_overlay = None
    if cls_heatmap is not None:
        print(f"[{alias}] Extracted CLS attention heatmap of shape {list(cls_heatmap.shape)}")
        cls_heatmap_resized, cls_overlay = overlay_heatmap(image_rgb, cls_heatmap)
    else:
        print(f"[{alias}] CLS attention heatmap is unavailable for this backbone.")

    save_path = output_dir / f"{image_path.stem}_trained_{alias}_heatmap.png"
    save_visualization(
        image_rgb,
        heatmap_resized,
        overlay,
        f"Trained ACT {alias} ({image_size[0]}x{image_size[1]})",
        save_path,
        cls_heatmap=cls_heatmap_resized,
        cls_overlay=cls_overlay,
    )
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
