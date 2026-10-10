import os
import glob
import torch
import cv2
import numpy as np
from unet import UNet
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def _process_multi_channel_mask(mask_hwc):
    mask_color = np.zeros((mask_hwc.shape[0], mask_hwc.shape[1], 3), dtype=np.uint8)
    c_len = mask_hwc.shape[2]

    colors = plt.get_cmap('hsv', max(2, c_len))
    for c in range(c_len):
        color = np.array(colors(c)[:3]) * 255
        mask_color[mask_hwc[:, :, c] > 0] = color
    return mask_color


def _tensor_to_rgb_image(image_tensor):
    image_np = image_tensor.detach().cpu().float().clamp(0, 1).numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = (image_np * 255.0).round().astype(np.uint8)
    return image_np


def _normalize_heatmap(heatmap):
    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()
    if heatmap_max > heatmap_min:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap = heatmap - heatmap_min
    return heatmap


def evaluate_teacher_tensor(model, image_tensor, save_path, device=None):
    if device is None:
        device = next(model.parameters()).device

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    model.eval()

    raw_image = image_tensor[0]
    image_rgb = _tensor_to_rgb_image(raw_image)
    orig_h, orig_w = image_rgb.shape[:2]

    input_tensor = image_tensor.to(device=device, dtype=torch.float32)
    input_tensor = F.interpolate(
        input_tensor,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    # English,English ACT English image_tensor English 255.0,English ImageNet Normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    input_tensor = (input_tensor - mean) / std

    hook_features = {}

    def hook_fn(module, hook_input, hook_output):
        hook_features["up4"] = hook_output.detach()

    handle = model.up4.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            output = model(input_tensor)
    finally:
        handle.remove()

    pred_prob = torch.sigmoid(output).squeeze(0).cpu().numpy()
    pred_mask = (pred_prob > 0.5).astype(np.uint8)
    pred_mask = pred_mask.transpose(1, 2, 0)
    pred_mask_orig = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    if len(pred_mask_orig.shape) == 2:
        pred_mask_orig = np.expand_dims(pred_mask_orig, axis=-1)

    pred_color_rgb = _process_multi_channel_mask(pred_mask_orig)
    overlay = cv2.addWeighted(image_rgb, 0.7, pred_color_rgb, 0.3, 0)

    up4_feat = hook_features["up4"]
    up4_heatmap = up4_feat.norm(dim=1)[0].cpu().numpy()
    up4_heatmap = _normalize_heatmap(up4_heatmap)
    up4_heatmap_resized = cv2.resize(up4_heatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    up4_heatmap_uint8 = np.uint8(255 * up4_heatmap_resized)
    up4_colored = cv2.applyColorMap(up4_heatmap_uint8, cv2.COLORMAP_JET)
    up4_colored = cv2.cvtColor(up4_colored, cv2.COLOR_BGR2RGB)
    up4_overlay = cv2.addWeighted(image_rgb, 0.6, up4_colored, 0.4, 0)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original RGB")
    axes[0].axis('off')

    if pred_mask_orig.shape[-1] == 1:
        axes[1].imshow(pred_mask_orig.squeeze(-1), cmap='gray', vmin=0, vmax=1)
    else:
        axes[1].imshow(pred_color_rgb)
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title("Mask Overlay")
    axes[2].axis('off')

    axes[3].imshow(up4_heatmap_resized, cmap="jet")
    axes[3].set_title("UP4 Heatmap")
    axes[3].axis('off')

    axes[4].imshow(up4_overlay)
    axes[4].set_title("UP4 Overlay")
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Teacher evaluation saved to {save_path}")

def evaluate_single_image(model_path, image_path, save_path=None, device=None, show=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. English
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = checkpoint['num_classes']

    model = UNet(n_channels=3, n_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded. Number of classes: {num_classes}")

    # 2. English
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. English
    print(f"Reading image from {image_path}...")
    img_data = np.fromfile(image_path, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # English,English
    orig_h, orig_w = img_np.shape[:2]

    # English 224x224,English (dataset.py) English INTER_CUBIC
    img_resized = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_CUBIC)

    # English
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    # English up4 English hook
    hook_features = {}
    def hook_fn(module, hook_input, hook_output):
        hook_features["up4"] = hook_output.detach()

    handle = None
    if hasattr(model, 'up4'):
        handle = model.up4.register_forward_hook(hook_fn)

    # English
    try:
        with torch.no_grad():
            output = model(img_tensor)
            # English tuple English
            if isinstance(output, tuple):
                output = output[0]

            # output English (1, num_classes, H, W)
            pred_prob = torch.sigmoid(output).squeeze(0).cpu().numpy()
            pred_mask = (pred_prob > 0.5).astype(np.uint8) # (C, H, W)
            pred_mask = pred_mask.transpose(1, 2, 0) # English (H, W, C)
    finally:
        if handle is not None:
            handle.remove()

    # English Mask English
    pred_mask_orig = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    if len(pred_mask_orig.shape) == 2:
        pred_mask_orig = np.expand_dims(pred_mask_orig, axis=-1)

    pred_color_rgb = _process_multi_channel_mask(pred_mask_orig)

    # English (English | EnglishMask | English)
    overlay = cv2.addWeighted(img_np, 0.7, pred_color_rgb, 0.3, 0)

    if "up4" in hook_features:
        up4_feat = hook_features["up4"]
        up4_heatmap = up4_feat.norm(dim=1)[0].cpu().numpy()
        up4_heatmap = _normalize_heatmap(up4_heatmap)
        up4_heatmap_resized = cv2.resize(up4_heatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        up4_heatmap_uint8 = np.uint8(255 * up4_heatmap_resized)
        up4_colored = cv2.applyColorMap(up4_heatmap_uint8, cv2.COLORMAP_JET)
        up4_colored = cv2.cvtColor(up4_colored, cv2.COLOR_BGR2RGB)
        up4_overlay = cv2.addWeighted(img_np, 0.6, up4_colored, 0.4, 0)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Original RGB")
        axes[0].axis('off')

        if pred_mask_orig.shape[-1] == 1:
            axes[1].imshow(pred_mask_orig.squeeze(-1), cmap='gray', vmin=0, vmax=1)
        else:
            axes[1].imshow(pred_color_rgb)
        axes[1].set_title("Predicted Mask")
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title("Mask Overlay")
        axes[2].axis('off')

        axes[3].imshow(up4_heatmap_resized, cmap="jet")
        axes[3].set_title("UP4 Heatmap")
        axes[3].axis('off')

        axes[4].imshow(up4_overlay)
        axes[4].set_title("UP4 Overlay")
        axes[4].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Original RGB")
        axes[0].axis('off')

        if pred_mask_orig.shape[-1] == 1:
            axes[1].imshow(pred_mask_orig.squeeze(-1), cmap='gray', vmin=0, vmax=1)
        else:
            axes[1].imshow(pred_color_rgb)
        axes[1].set_title("Predicted Mask")
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Result saved to {save_path}")
    else:
        # English,English ./results English
        os.makedirs('./results', exist_ok=True)
        default_save_path = os.path.join('./results', os.path.basename(image_path))
        plt.savefig(default_save_path, dpi=200, bbox_inches='tight')
        print(f"Result saved to {default_save_path}")

    if show:
        plt.show()
    plt.close()

def evaluate_model(model_path, test_dir, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. English
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = checkpoint['num_classes']

    model = UNet(n_channels=3, n_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded. Number of classes: {num_classes}")

    # 2. English
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. English
    cams = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']

    for cam_name in cams:
        rgb_dir = os.path.join(test_dir, cam_name, 'rgb')
        seg_dir = os.path.join(test_dir, cam_name, 'seg')

        if not os.path.exists(rgb_dir):
            continue

        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.*')))

        print(f"Evaluating camera: {cam_name}, found {len(rgb_files)} images")

        for step, rgb_file in enumerate(rgb_files):
            if step % 10 != 0: # English10English
                continue

            base_name = os.path.splitext(os.path.basename(rgb_file))[0]
            seg_file_npy = os.path.join(seg_dir, f"{base_name}.npy")
            seg_file_png = os.path.join(seg_dir, f"{base_name}.png")

            if os.path.exists(seg_file_npy):
                gt_mask = np.load(seg_file_npy)
            elif os.path.exists(seg_file_png):
                gt_mask = cv2.imread(seg_file_png, cv2.IMREAD_UNCHANGED)
                if len(gt_mask.shape) == 2:
                    gt_mask = np.expand_dims(gt_mask, axis=-1)
            else:
                continue

            img_np = cv2.imread(rgb_file)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

            # English,English
            # English evaluate_model English resize,English Tensor
            # English resize,English 224x224 English
            img_resized = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_CUBIC)
            img_tensor = transform(img_resized).unsqueeze(0).to(device)

            # English
            with torch.no_grad():
                output = model(img_tensor)
                # output English (1, num_classes, H, W)
                # English (BCE),English sigmoid English
                pred_prob = torch.sigmoid(output).squeeze(0).cpu().numpy()

            pred_mask = (pred_prob > 0.5).astype(np.uint8) # (C, H, W)
            pred_mask = pred_mask.transpose(1, 2, 0) # English (H, W, C)

            # English Mask English,English resize
            orig_h, orig_w = img_np.shape[:2]
            pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            if len(pred_mask.shape) == 2:
                pred_mask = np.expand_dims(pred_mask, axis=-1)

            gt_color_rgb = _process_multi_channel_mask(gt_mask)
            pred_color_rgb = _process_multi_channel_mask(pred_mask)

            # English (English | EnglishMask | EnglishMask)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_np)
            axes[0].set_title(f"Original RGB ({cam_name})")
            axes[0].axis('off')

            if gt_mask.shape[-1] == 1:
                axes[1].imshow(gt_mask.squeeze(-1), cmap='gray', vmin=0, vmax=1)
            else:
                axes[1].imshow(gt_color_rgb)
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis('off')

            if pred_mask.shape[-1] == 1:
                axes[2].imshow(pred_mask.squeeze(-1), cmap='gray', vmin=0, vmax=1)
            else:
                axes[2].imshow(pred_color_rgb)
            axes[2].set_title("Predicted Mask")
            axes[2].axis('off')

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(2) # English2English
            plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate U-Net Segmentation Model")

    default_model_path = r"checkpoints/best_unet_model.pth"
    default_test_data = r"../data/extracted_all_episodes/episode0"

    parser.add_argument("--model", type=str, default=default_model_path, help="Path to trained model")
    parser.add_argument("--test_data", type=str, default=default_test_data, help="Path to test episode directory (optional)")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image to evaluate")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the single image evaluation result")
    parser.add_argument("--show", action="store_true", help="Show the visualization result window")

    args = parser.parse_args()

    if args.image:
        if os.path.exists(args.model) and os.path.exists(args.image):
            should_show = args.show or args.save_path is None
            evaluate_single_image(args.model, args.image, args.save_path, show=should_show)
        else:
            print(f"Error: Model ({args.model}) or Image ({args.image}) not found.")
    else:
        if os.path.exists(args.model) and os.path.exists(args.test_data):
            evaluate_model(args.model, args.test_data)
        else:
            print(f"Error: Model ({args.model}) or Test Data ({args.test_data}) not found.")
            print("Please train the model first by running train.py")
