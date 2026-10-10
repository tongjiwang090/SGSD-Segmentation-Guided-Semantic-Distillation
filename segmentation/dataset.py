import glob
import os

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class H1SegmentationDataset(Dataset):
    """Read RGB frames and actor masks from processed RoboTwin episodes."""

    CAMERAS = ("cam_high", "cam_right_wrist", "cam_left_wrist")

    def __init__(self, data_dir, split="train", train_ratio=0.8, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.h5_files = sorted(glob.glob(os.path.join(data_dir, "episode*.hdf5")))
        if not self.h5_files:
            raise ValueError(f"No .hdf5 episodes found in {data_dir}")

        num_episodes = len(self.h5_files)
        num_train = max(1, int(num_episodes * train_ratio))
        if split == "train":
            self.h5_files = self.h5_files[:num_train]
        elif split == "val":
            self.h5_files = self.h5_files[num_train:] if num_train < num_episodes else self.h5_files
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.samples = []
        print(f"Scanning {split} split...")
        for h5_file in self.h5_files:
            try:
                with h5py.File(h5_file, "r") as file_handle:
                    observations = file_handle.get("observations")
                    if observations is None or "images" not in observations or "segmentation" not in observations:
                        continue
                    for camera in self.CAMERAS:
                        if camera not in observations["images"] or camera not in observations["segmentation"]:
                            continue
                        frame_count = min(
                            len(observations["images"][camera]),
                            len(observations["segmentation"][camera]),
                        )
                        for frame_idx in range(frame_count):
                            self.samples.append(
                                {"h5_path": h5_file, "cam": camera, "frame_idx": frame_idx}
                            )
            except Exception as exc:
                print(f"Failed to read {h5_file}: {exc}")

        print(f"{split} split contains {len(self.samples)} samples")
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )

        if self.samples:
            _, sample_mask, _ = self[0]
            self.num_classes = sample_mask.shape[0]
        else:
            self.num_classes = 1
        print(f"[{split}] mask channels: {self.num_classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with h5py.File(sample["h5_path"], "r") as file_handle:
            observations = file_handle["observations"]
            image = observations["images"][sample["cam"]][sample["frame_idx"]].copy()
            mask = observations["segmentation"][sample["cam"]][sample["frame_idx"]]

        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.dtype != np.uint8:
            if np.issubdtype(mask.dtype, np.integer):
                mask = mask.astype(np.uint8)
            else:
                mask_min, mask_max = mask.min(), mask.max()
                scale = max(float(mask_max - mask_min), 1e-8)
                mask = ((mask - mask_min) / scale * 255).astype(np.uint8)
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        if mask.ndim == 2:
            mask = mask[..., None]

        if self.split == "train" and np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            if mask.ndim == 2:
                mask = mask[..., None]

        image_tensor = transforms.ToTensor()(image)
        if self.split == "train":
            image_tensor = self.color_jitter(image_tensor)
        original_image = image_tensor.clone()
        image_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(image_tensor)

        mask_tensor = torch.from_numpy(mask).float().permute(2, 0, 1)
        mask_tensor = (mask_tensor > 0).float()
        return image_tensor, mask_tensor, original_image
