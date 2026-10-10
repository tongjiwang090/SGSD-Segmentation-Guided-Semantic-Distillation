import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

# English
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from act_policy import ACTPolicy

def get_heatmap(feat_tensor, size=(224, 224)):
    """
    English.
    feat_tensor: English (1, C, H, W)
    """
    # English,English
    heatmap = torch.mean(torch.abs(feat_tensor), dim=1).squeeze().cpu().numpy()

    # English [0, 1] English
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # English,English
    heatmap = cv2.resize(heatmap, size, interpolation=cv2.INTER_CUBIC)

    # English (JET colormap)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return heatmap_color

def main():
    parser = argparse.ArgumentParser(description="Visualize Heatmaps for ACT Backbone and Projections")
    parser.add_argument("--image", type=str, default="./extracted_images/cam_high/frame_0025.png", help="Path to input image")
    parser.add_argument("--ckpt", type=str, default="./act_ckpt/act-beat_block_hammer/demo_clean-50/policy_epoch_2000_seed_0.ckpt", help="Path to model checkpoint")
    parser.add_argument("--save_path", type=str, default="heatmap_visualization.png", help="Path to save the output visualization")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # English DINOv2 English
    args_override = {
        "lr": 1e-5,
        "chunk_size": 50, # English:model.query_embed.weight English 50
        "num_queries": 50, # English:model.query_embed.weight English 50
        "kl_weight": 10.0,
        "hidden_dim": 512, # English:model.action_head.weight English 512
        "dim_feedforward": 3200, # English:model.encoder.layers.0.linear1.weight English 3200
        "lr_backbone": 1e-5,
        "backbone_type": "dinov2",
        "backbone": "vit_base_patch14_dinov2",
        "backbone_pretrained": False,
        "backbone_image_size": "224,224",
        "use_distillation": True, # English distill_proj
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"], # English (3 cams)
        "device": device
    }

    print("Initializing ACT Policy model...")
    # Mock sys.argv to bypass required arguments in detr/main.py parser
    original_argv = sys.argv.copy()
    sys.argv = [
        "dummy.py",
        "--policy_class", "ACT",
        "--task_name", "dummy",
        "--seed", "0",
        "--num_epochs", "0",
        "--ckpt_dir", "dummy"
    ]

    policy = ACTPolicy(args_override)
    sys.argv = original_argv
    policy.to(device)

    print(f"Loading checkpoint from: {args.ckpt}")
    if not os.path.exists(args.ckpt):
        print(f"Warning: Checkpoint {args.ckpt} not found. Ensure the path is correct.")
        print("We will still attempt to run with random weights for demonstration.")
    else:
        # weights_only=False English
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        missing, unexpected = policy.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"Missing keys (can be ignored if not related to backbone/proj): {missing[:5]}...")
    policy.eval()

    print(f"Loading image from: {args.image}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found at {args.image}")

    orig_img = cv2.imread(args.image)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # English,English cv2.INTER_CUBIC English 224x224
    img_resized = cv2.resize(orig_img, (224, 224), interpolation=cv2.INTER_CUBIC)

    # English (English policy English transforms.Normalize English)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    # normalize English mean English std English act_policy.py
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img_tensor = (img_tensor - mean) / std

    # English (B, num_cams, C, H, W)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)

    print("Extracting features...")

    # English forward hook English attention
    attentions = []
    def hook(module, input, output):
        B, N, C3 = output.shape
        # English num_heads
        num_heads = getattr(module, 'num_heads', None)
        if num_heads is None:
            num_heads = policy.model.backbones[0][0].body.blocks[-1].attn.num_heads
        C = C3 // 3
        qkv = output.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = (C // num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attentions.append(attn)

    timm_body = policy.model.backbones[0][0].body
    handle = timm_body.blocks[-1].attn.qkv.register_forward_hook(hook)

    with torch.inference_mode():
        # 1. English (DINOv2 English norm English blocks[-1])

        # English timm English forward_features English patch tokens
        # English TimmVisionBackbone English _prepare_input English forward English
        prepared_tensor = policy.model.backbones[0][0]._prepare_input(img_tensor[:, 0])
        features_raw = timm_body.forward_features(prepared_tensor)

        # English norm English (English timm English forward_features English norm)
        # English backbone_wrapper English spatial feature map
        backbone_feat = policy.model.backbones[0][0]._to_feature_map(features_raw, prepared_tensor.shape[-2:]) # (1, 768, H', W')

        # 2. Transformer English (English hidden_dim=512)
        proj_feat = policy.model.input_proj(backbone_feat) # (1, 512, H', W')

        # 3. English (English,English policy.distill_proj)
        distill_feat = None
        if hasattr(policy, 'distill_proj'):
            distill_feat = policy.distill_proj(backbone_feat) # (1, 64, H', W')
        elif hasattr(policy.model, 'distill_proj'):
            distill_feat = policy.model.distill_proj(backbone_feat)

    handle.remove()

    print("Generating heatmaps...")
    # English overlay
    alpha = 0.5

    attn_overlay = None
    if len(attentions) > 0:
        attn = attentions[0] # (1, num_heads, N, N)
        cls_attn = attn[:, :, 0, 1:] # CLS English patches English
        cls_attn = cls_attn.mean(dim=1) # English (1, N-1)

        # English
        H, W = prepared_tensor.shape[-2:]
        patch_size = policy.model.backbones[0][0].patch_size
        grid_h = H // patch_size[0]
        grid_w = W // patch_size[1]

        if cls_attn.shape[1] == grid_h * grid_w:
            cls_attn = cls_attn.reshape(1, 1, grid_h, grid_w)
            attn_hm = get_heatmap(cls_attn, size=(224, 224))
            attn_overlay = cv2.addWeighted(img_resized, 1-alpha, attn_hm, alpha, 0)

    backbone_hm = get_heatmap(backbone_feat)
    backbone_overlay = cv2.addWeighted(img_resized, 1-alpha, backbone_hm, alpha, 0)

    proj_hm = get_heatmap(proj_feat)
    proj_overlay = cv2.addWeighted(img_resized, 1-alpha, proj_hm, alpha, 0)

    if distill_feat is not None:
        distill_hm = get_heatmap(distill_feat)
        distill_overlay = cv2.addWeighted(img_resized, 1-alpha, distill_hm, alpha, 0)

    # English
    num_plots = 4 if distill_feat is not None else 3
    if attn_overlay is not None:
        num_plots += 1

    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4.5))

    axes[0].imshow(img_resized)
    axes[0].set_title("Original Image (224x224)", fontsize=10)
    axes[0].axis("off")

    curr_idx = 1
    if attn_overlay is not None:
        axes[curr_idx].imshow(attn_overlay)
        axes[curr_idx].set_title("CLS Attention Heatmap", fontsize=10)
        axes[curr_idx].axis("off")
        curr_idx += 1

    axes[curr_idx].imshow(backbone_overlay)
    axes[curr_idx].set_title(f"Backbone Last Layer Heatmap\n({backbone_feat.shape[1]} channels)", fontsize=10)
    axes[curr_idx].axis("off")
    curr_idx += 1

    axes[curr_idx].imshow(proj_overlay)
    axes[curr_idx].set_title(f"Input Proj Heatmap\n({proj_feat.shape[1]} channels)", fontsize=10)
    axes[curr_idx].axis("off")
    curr_idx += 1

    if distill_feat is not None:
        axes[curr_idx].imshow(distill_overlay)
        axes[curr_idx].set_title(f"Distill Proj Heatmap\n({distill_feat.shape[1]} channels)", fontsize=10)
        axes[curr_idx].axis("off")

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300)
    print(f"Visualization successfully saved to: {args.save_path}")

    # English
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot interactively: {e}")

if __name__ == "__main__":
    main()
