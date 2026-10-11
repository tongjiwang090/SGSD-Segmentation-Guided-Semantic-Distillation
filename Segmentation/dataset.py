import os
import glob
import numpy as np
import cv2
import torch
import h5py
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class H1SegmentationDataset(Dataset):
    def __init__(self, data_dir, split='train', train_ratio=0.8, transform=None):
        """
        args:
            data_dir (str): directory holding the hdf5 files
            split (str): 'train' or 'val'
            train_ratio (float): train fraction
            transform: optional augmentation, unused, kept for interface compatibility
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # all hdf5 files
        self.h5_files = sorted(glob.glob(os.path.join(data_dir, 'episode*.hdf5')))
        if not self.h5_files:
            raise ValueError(f"no .hdf5 files found in {data_dir}")
            
        # train / val split
        num_episodes = len(self.h5_files)
        num_train = max(1, int(num_episodes * train_ratio))
        
        if split == 'train':
            self.h5_files = self.h5_files[:num_train]
        elif split == 'val':
            self.h5_files = self.h5_files[num_train:] if num_train < num_episodes else self.h5_files
        else:
            raise ValueError("split must be 'train' or 'val'")
            
        # sample index
        self.samples = []
        cams = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']
        
        print(f"scanning {split} split...")
        for h5_file in self.h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'observations' not in f:
                        continue
                    obs_group = f['observations']
                    if 'images' not in obs_group or 'segmentation' not in obs_group:
                        continue
                        
                    for cam in cams:
                        if cam not in obs_group['images'] or cam not in obs_group['segmentation']:
                            continue
                            
                        num_frames = len(obs_group['images'][cam])
                        seg_frames = len(obs_group['segmentation'][cam])
                        num_frames = min(num_frames, seg_frames)
                        
                        # one sample per frame
                        for frame_idx in range(num_frames):
                            self.samples.append({
                                'h5_path': h5_file,
                                'cam': cam,
                                'frame_idx': frame_idx
                            })
            except Exception as e:
                print(f"failed to read {h5_file}: {e}")
                
        print(f"{split} split: {len(self.samples)} samples")
        
        # color jitter, defined before __getitem__ is first called
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        
        if len(self.samples) > 0:
            # read sample 0 to learn the real channel count, matching __getitem__
            # __getitem__ returns 3 values: image_tensor, mask_tensor, orig_image_tensor
            _, sample_mask, _ = self[0]
            self.num_classes = sample_mask.shape[0]
        else:
            self.num_classes = 1
            
        print(f"[{split}] mask channels (num_classes): {self.num_classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        h5_path = sample_info['h5_path']
        cam = sample_info['cam']
        frame_idx = sample_info['frame_idx']
        
        with h5py.File(h5_path, 'r') as f:
            obs_group = f['observations']
            
            # ---- 1. rgb ----
            # images are already decoded in the hdf5 as uint8 (H, W, 3), natively rgb
            img_array = obs_group['images'][cam][frame_idx]
            image = img_array.copy()
            
            # bicubic to 224x224, smooth for inputs
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
            
            # ---- 2. segmentation mask ----
            seg_data = obs_group['segmentation'][cam][frame_idx]
            _ch = int(os.environ.get('TEACHER_SEG_CH', '0'))
            raw = seg_data[..., _ch] if seg_data.ndim == 3 else seg_data

            # binarize before casting to uint8: the original cast first, so actor ids
            # above 255 wrapped (260 -> 4 selects an unrelated actor, 256 -> 0 drops out
            # of the foreground). INTER_NEAREST does not interpolate, so for ids <= 255
            # this stays pixel-identical to the original ordering.
            _mode = os.environ.get('TEACHER_SEG_MODE', 'all')
            if _mode == 'target':
                _ids = [int(x) for x in os.environ.get('TEACHER_SEG_IDS', '').split(',') if x.strip()]
                if not _ids:
                    raise ValueError('TEACHER_SEG_MODE=target but TEACHER_SEG_IDS is empty')
                fg = np.isin(raw, _ids)
            elif _mode == 'all':
                fg = raw > 0
            else:
                raise ValueError('unknown TEACHER_SEG_MODE: %r' % _mode)

            # announce once: the training log has to prove which mask convention ran,
            # otherwise m-all and m-target arms cannot be told apart afterwards.
            if not getattr(H1SegmentationDataset, '_seg_announced', False):
                print('[teacher] mask mode = %s%s, channel = %d'
                      % (_mode,
                         (', target ids ' + os.environ.get('TEACHER_SEG_IDS', '')) if _mode == 'target' else '',
                         _ch), flush=True)
                H1SegmentationDataset._seg_announced = True

            mask = fg.astype(np.uint8) * 255
            mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
            
        # mask must end up (H, W, C)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
            
        # ---- 2.5 augmentation, train split only ----
        if self.split == 'train':
            # random horizontal flip
            if np.random.rand() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
                # cv2.flip drops the channel dim on single-channel input
                if len(mask.shape) == 2:
                    mask = np.expand_dims(mask, axis=-1)
        
        # ---- 3. to torch tensors ----
        image_tensor = transforms.ToTensor()(image)
        if self.split == 'train':
            image_tensor = self.color_jitter(image_tensor)
        
        # keep the un-normalized tensor (0~1) for visual comparison
        orig_image_tensor = image_tensor.clone()
        
        image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
        
        # mask: to tensor, reordered to (C, H, W)
        mask_tensor = torch.from_numpy(mask).float().permute(2, 0, 1)
        # binary task: threshold at 0
        mask_tensor = (mask_tensor > 0).float()
        
        # return orig_image_tensor too, so train.py need not un-normalize by hand
        return image_tensor, mask_tensor, orig_image_tensor