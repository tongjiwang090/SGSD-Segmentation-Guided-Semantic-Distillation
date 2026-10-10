import glob
import os

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def save_rgb_and_segmentation_for_episode(hdf5_path, output_dir, cameras=None):
    """Export RGB frames and actor masks from one raw RoboTwin episode."""
    if not os.path.exists(hdf5_path):
        print(f"File not found, skipping: {hdf5_path}")
        return False

    with h5py.File(hdf5_path, "r") as file_handle:
        observations = file_handle["observation"]
        if cameras is None:
            cameras = [
                name
                for name in observations.keys()
                if "rgb" in observations[name] and "actor_segmentation" in observations[name]
            ]
        if not cameras:
            print(f"No camera with RGB and actor segmentation data: {hdf5_path}")
            return False

        os.makedirs(output_dir, exist_ok=True)
        for camera in cameras:
            camera_group = observations[camera]
            if "rgb" not in camera_group or "actor_segmentation" not in camera_group:
                print(f"Skipping {camera}: RGB or actor segmentation data is missing")
                continue

            rgb_streams = camera_group["rgb"][:]
            segmentation = camera_group["actor_segmentation"][:]
            frame_count = min(len(rgb_streams), segmentation.shape[0])
            if len(rgb_streams) != segmentation.shape[0]:
                print(
                    f"Warning: {camera} has {len(rgb_streams)} RGB frames and "
                    f"{segmentation.shape[0]} masks; using {frame_count} frames"
                )

            camera_dir = os.path.join(output_dir, camera)
            rgb_dir = os.path.join(camera_dir, "rgb")
            mask_dir = os.path.join(camera_dir, "seg")
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            print(f"Processing {camera}: {frame_count} frames")

            for frame_idx in tqdm(range(frame_count), desc=f"  {camera}", unit="frame", leave=False):
                encoded = rgb_streams[frame_idx]
                encoded = encoded if isinstance(encoded, bytes) else encoded.tobytes()
                rgb_image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if rgb_image is None:
                    print(f"Warning: could not decode RGB frame {frame_idx} from {camera}")
                else:
                    cv2.imwrite(
                        os.path.join(rgb_dir, f"frame_{frame_idx:04d}.png"), rgb_image
                    )

                mask = segmentation[frame_idx]
                if mask.ndim == 3:
                    mask = mask[..., 0]
                if mask.dtype != np.uint8:
                    if np.issubdtype(mask.dtype, np.integer):
                        mask = mask.astype(np.uint8)
                    else:
                        mask_min, mask_max = mask.min(), mask.max()
                        scale = max(float(mask_max - mask_min), 1e-8)
                        mask = ((mask - mask_min) / scale * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(mask_dir, f"frame_{frame_idx:04d}.png"), mask)

            print(f"Finished {camera}: RGB={rgb_dir}, masks={mask_dir}")
    return True


def batch_process_all_episodes(data_dir_pattern, output_root="extracted_all_episodes", cameras=None):
    """Export every episode matching ``data_dir_pattern`` into separate folders."""
    file_list = sorted(glob.glob(data_dir_pattern))
    if not file_list:
        print(f"No files matched: {data_dir_pattern}")
        return
    print(f"Found {len(file_list)} episode files")

    for file_path in file_list:
        basename = os.path.basename(file_path)
        episode_name = os.path.splitext(basename)[0]
        episode_dir = os.path.join(output_root, episode_name)
        print(f"\nProcessing {basename} -> {episode_dir}")
        success = save_rgb_and_segmentation_for_episode(file_path, episode_dir, cameras)
        print(f"{'Finished' if success else 'Failed'}: {basename}")


if __name__ == "__main__":
    data_pattern = "../data/beat_block_hammer/demo_clean/data/episode*.hdf5"
    output_root_dir = "../data/extracted_all_episodes"
    batch_process_all_episodes(data_pattern, output_root_dir)
