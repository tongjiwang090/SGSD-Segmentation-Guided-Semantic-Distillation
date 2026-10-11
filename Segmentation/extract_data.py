import h5py
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

def save_rgb_and_segmentation_for_episode(hdf5_path, output_dir, cameras=None):
    """
    split one hdf5 file into rgb and segmentation images under the output dir.
    
    args:
        hdf5_path (str): path to a single hdf5 file
        output_dir (str): output root for this episode; camera subdirs are created here
        cameras (list): camera names to process; default is every camera with rgb and actor_segmentation
    """
    if not os.path.exists(hdf5_path):
        print(f"file does not exist, skipping: {hdf5_path}")
        return False

    with h5py.File(hdf5_path, 'r') as f:
        obs_group = f['observation']
        
        if cameras is None:
            cameras = []
            for cam in obs_group.keys():
                if 'rgb' in obs_group[cam] and 'actor_segmentation' in obs_group[cam]:
                    cameras.append(cam)
                else:
                    print(f"warning: camera {cam} has no rgb or segmentation, skipping")
        
        if len(cameras) == 0:
            print(f"no usable camera data, skipping: {hdf5_path}")
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        
        for cam in cameras:
            cam_group = obs_group[cam]
            if 'rgb' not in cam_group:
                print(f"warning: {cam} has no rgb, skipping")
                continue
            if 'actor_segmentation' not in cam_group:
                print(f"warning: {cam} has no actor_segmentation, skipping")
                continue
            
            rgb_streams = cam_group['rgb'][:]          # shape (T,)
            seg_data = cam_group['actor_segmentation'][:]  # shape (T, H, W) or (T, H, W, C)
            
            num_frames = len(rgb_streams)
            if seg_data.shape[0] != num_frames:
                print(f"warning: {cam} rgb frames ({num_frames}) != segmentation frames ({seg_data.shape[0]}), taking the smaller")
                num_frames = min(num_frames, seg_data.shape[0])
            
            # rgb and seg subdirs for this camera
            cam_dir = os.path.join(output_dir, cam)
            rgb_dir = os.path.join(cam_dir, "rgb")
            seg_dir = os.path.join(cam_dir, "seg")
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(seg_dir, exist_ok=True)
            
            print(f"  processing {cam}: {num_frames} frames")
            for i in tqdm(range(num_frames), desc=f"  {cam}", unit="frame", leave=False):
                # ----- rgb -----
                rgb_bytes = rgb_streams[i]
                if isinstance(rgb_bytes, bytes):
                    img_array = np.frombuffer(rgb_bytes, dtype=np.uint8)
                else:
                    img_array = np.frombuffer(rgb_bytes.tobytes(), dtype=np.uint8)
                rgb_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if rgb_img is None:
                    print(f"    warning: {cam} frame {i} rgb decode failed, skipping")
                else:
                    rgb_filename = f"frame_{i:04d}.png"
                    rgb_path = os.path.join(rgb_dir, rgb_filename)
                    cv2.imwrite(rgb_path, rgb_img)
                
                # ----- segmentation -----
                seg_frame = seg_data[i]
                if seg_frame.ndim == 3:
                    seg_img = seg_frame[..., 0]   # first channel
                else:
                    seg_img = seg_frame
                # to uint8
                if seg_img.dtype != np.uint8:
                    if np.issubdtype(seg_img.dtype, np.integer):
                        seg_img = seg_img.astype(np.uint8)
                    else:
                        seg_img = ((seg_img - seg_img.min()) / (seg_img.max() - seg_img.min()) * 255).astype(np.uint8)
                seg_filename = f"frame_{i:04d}.png"
                seg_path = os.path.join(seg_dir, seg_filename)
                cv2.imwrite(seg_path, seg_img)
            
            print(f"  done {cam}: rgb in {rgb_dir}, segmentation in {seg_dir}")
        return True

def batch_process_all_episodes(data_dir_pattern, output_root="extracted_all_episodes", cameras=None):
    """
    process every matching file, one episode per output_root/episodeX/ subdir.
    
    args:
        data_dir_pattern (str): glob pattern, e.g. "/path/to/episode*.hdf5"
        output_root (str): output root; one subdir per episode
        cameras (list): camera names to process
    """
    # all matching hdf5 files
    file_list = sorted(glob.glob(data_dir_pattern))
    if not file_list:
        print(f"no files matched: {data_dir_pattern}")
        return
    
    print(f"found {len(file_list)} files")
    
    for file_path in file_list:
        # episode name from the filename, e.g. episode0.hdf5 -> episode0
        basename = os.path.basename(file_path)
        episode_name = os.path.splitext(basename)[0]  # drop .hdf5
        # or use the full filename without extension as the subdir name
        episode_output_dir = os.path.join(output_root, episode_name)
        
        print(f"\nprocessing: {basename} -> {episode_output_dir}")
        success = save_rgb_and_segmentation_for_episode(file_path, episode_output_dir, cameras)
        if success:
            print(f"done: {basename}")
        else:
            print(f"failed: {basename}")

if __name__ == "__main__":
    # path pattern, adjust as needed
    data_pattern = "../data/beat_block_hammer/demo_clean/data/episode*.hdf5"
    output_root_dir = "../data/extracted_all_episodes"
    
    # optional: restrict cameras; default detects every one with rgb and segmentation
    # cameras = ['front_camera', 'head_camera', 'left_camera', 'right_camera']
    cameras = None   # auto-detect
    
    batch_process_all_episodes(data_pattern, output_root_dir, cameras)