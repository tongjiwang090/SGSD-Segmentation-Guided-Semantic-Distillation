import os

# Set rendering backend for MuJoCo
os.environ["MUJOCO_GL"] = "egl"

import torch
import torch.nn.functional as F
import numpy as np
import pickle
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data  # data functions
from utils import sample_box_pose, sample_insertion_pose  # robot functions
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from act_policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

try:
    from backbone_config import apply_backbone_config
except ImportError:
    from .backbone_config import apply_backbone_config

import IPython

e = IPython.embed


# ===== diagnostic helpers =====
import numpy as np

def compute_gram_matrix(feat, target_size=16):
    """
    adaptively average-pool feat to target_size x target_size, then compute the gram matrix.
    feat: [B, C, H, W]
    """
    B, C, H, W = feat.shape
    # pool down
    if H > target_size or W > target_size:
        feat = F.adaptive_avg_pool2d(feat, (target_size, target_size))
    _, _, Hp, Wp = feat.shape
    f = feat.view(B, C, -1)  # [B, C, N]，N=Hp*Wp
    f_norm = F.normalize(f, dim=1)
    gram = torch.bmm(f_norm.transpose(1, 2), f_norm)  # [B, N, N]
    return gram

def heuristic_artifact_mask(s_feat, t_feat, threshold_s=1.5, threshold_t=0.8):
    """
    norm-based heuristic for artifact candidates:
    - student norm > student mean * threshold_s
    - and teacher norm < teacher mean * threshold_t
    returns mask [B, H, W] bool
    """
    s_norm = torch.norm(s_feat, dim=1)      # [B, H, W]
    t_norm = torch.norm(t_feat, dim=1)      # [B, H, W]
    s_mean = s_norm.mean(dim=(1,2), keepdim=True)
    t_mean = t_norm.mean(dim=(1,2), keepdim=True)
    mask = (s_norm > s_mean * threshold_s) & (t_norm < t_mean * threshold_t)
    return mask

def group_mse_by_mask(s_feat, t_feat, mask):
    """
    split pixels by mask (bool, [B,H,W]) and average mse within each group.
    returns (mse_mask, mse_unmask, ratio)
    """
    mse_per_pixel = F.mse_loss(s_feat, t_feat, reduction='none').mean(dim=1)  # [B,H,W]
    B, H, W = mse_per_pixel.shape
    mask_flat = mask.view(B, -1)      # [B, N]
    mse_flat = mse_per_pixel.view(B, -1)
    mse_mask_vals = []
    mse_unmask_vals = []
    for b in range(B):
        m = mask_flat[b]
        if m.sum() > 0:
            mse_mask_vals.append(mse_flat[b][m].mean().item())
        else:
            mse_mask_vals.append(float('nan'))
        if (~m).sum() > 0:
            mse_unmask_vals.append(mse_flat[b][~m].mean().item())
        else:
            mse_unmask_vals.append(float('nan'))
    return np.nanmean(mse_mask_vals), np.nanmean(mse_unmask_vals), mask_flat.float().mean().item()


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {value}")


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]

    # get task parameters
    is_sim = task_name[:4] == "sim-"
    if is_sim:
        from constants import SIM_TASK_CONFIGS

        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS

        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # fixed parameters
    state_dim = 14  # yiheng
    backbone_overrides = apply_backbone_config(
        {
            "backbone_type": args.get("backbone_type"),
            "backbone": args.get("backbone"),
            "backbone_pretrained": args.get("backbone_pretrained"),
            "backbone_image_size": args.get("backbone_image_size"),
            "use_distillation": args.get("use_distillation"),
            "freeze_last_layer_and_cls": args.get("freeze_last_layer_and_cls", False),
            "unfreeze_blocks": args.get("unfreeze_blocks"),
        }
    )
    lr_backbone = args["lr_backbone"]
    distill_weight = args.get("distill_weight", 10.0)
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone_type": backbone_overrides["backbone_type"],
            "backbone": backbone_overrides["backbone"],
            "backbone_pretrained": backbone_overrides["backbone_pretrained"],
            "backbone_image_size": backbone_overrides["backbone_image_size"],
            "use_distillation": backbone_overrides.get("use_distillation", False),
            "freeze_last_layer_and_cls": backbone_overrides.get("freeze_last_layer_and_cls", False),
            "unfreeze_blocks": backbone_overrides.get("unfreeze_blocks"),
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
            "distill_weight": distill_weight,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone_type": backbone_overrides["backbone_type"],
            "backbone": backbone_overrides["backbone"],
            "backbone_pretrained": backbone_overrides["backbone_pretrained"],
            "backbone_image_size": backbone_overrides["backbone_image_size"],
            "num_queries": 1,
            "camera_names": camera_names,
            "distill_weight": distill_weight,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "teacher_vis_interval": args["teacher_vis_interval"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "save_freq": args['save_freq'],
        "resume_ckpt": args.get("resume_ckpt", None),
        "distill_loss_type": args.get("distill_loss_type", "focal_cosine"),
        "distill_at_block": args.get("distill_at_block"),
        "distill_focal_gamma": args.get("distill_focal_gamma", 2.0),
        "distill_shuffle_teacher": args.get("distill_shuffle_teacher", False),
        "distill_shuffle_scope": args.get("distill_shuffle_scope", "space"),
    }

    if is_eval:
        ckpt_names = [f"policy_best.ckpt"]
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train,
                                                           batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    # dump the full run config so the experiment stays reproducible
    import json as _json, subprocess as _sp, datetime as _dt
    try:
        _git = _sp.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                stderr=_sp.DEVNULL).decode().strip()
    except Exception:
        _git = None
    _run_cfg = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git,
        "cli_args": {k: (str(v) if not isinstance(v, (int, float, bool, type(None), list)) else v)
                     for k, v in args.items()},
        "policy_config": {k: (str(v) if not isinstance(v, (int, float, bool, type(None), list)) else v)
                          for k, v in policy_config.items()},
        "train_config": {k: v for k, v in config.items() if k != "policy_config"},
        "dataset_dir": dataset_dir,
        "num_episodes": num_episodes,
    }
    with open(os.path.join(ckpt_dir, "run_config.json"), "w", encoding="utf-8") as f:
        _json.dump(_run_cfg, f, indent=2, ensure_ascii=False)
    print(f"[run_config] saved to {os.path.join(ckpt_dir, 'run_config.json')}")
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers  # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha

        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env

        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if "sim_transfer_cube" in task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif "sim_insertion" in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if "images" in obs:
                    image_list.append(obs["images"])
                else:
                    image_list.append({"main": obs["image"]})
                qpos_numpy = np.array(obs["qpos"])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config["policy_class"] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1))
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers(
                [env.puppet_bot_left, env.puppet_bot_right],
                [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                move_time=0.5,
            )  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )

        if save_episode:
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(ckpt_dir, f"video{rollout_id}.mp4"),
            )

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    # save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None


def compute_distillation_loss(s_feat_proj_resized, t_feat, loss_type="focal_cosine", focal_gamma=2.0):
    """
    distillation losses
    
    Args:
        s_feat_proj_resized: student features (B*num_cams, C, H, W)
        t_feat: teacher features (B*num_cams, C, H, W)
        loss_type: which objective
        focal_gamma: gamma for focal weighting
    
    Returns:
        distill_loss: the scalar loss
        loss_dict: per-term breakdown
    """
    # s_norm = F.normalize(s_feat_proj_resized, dim=1)
    # t_norm = F.normalize(t_feat, dim=1)
    
    cos_sim = F.cosine_similarity(s_feat_proj_resized, t_feat, dim=1)  # (B*num_cams, H, W)
    base_dist = 1.0 - cos_sim  # (B*num_cams, H, W)
    
    # spatial activation magnitude, used by several variants below
    s_mag = torch.norm(s_feat_proj_resized, dim=1, keepdim=False)  # (B*num_cams, H, W)
    t_mag = torch.norm(t_feat, dim=1, keepdim=False)  # (B*num_cams, H, W)
    
    loss_dict = {}
    
    if loss_type == "cosine":
        # cosine distance only
        distill_loss = base_dist.mean()
        loss_dict["distill_cosine"] = distill_loss
        
    elif loss_type == "mse":
        # mse only
        distill_loss = F.mse_loss(s_norm, t_norm)
        loss_dict["distill_mse"] = distill_loss
        
    elif loss_type == "attention_mse":
        # original variant: attention_weight * mse_loss
        attention_weight = 1.0 - cos_sim.mean(dim=[1, 2], keepdim=True)  # (B*num_cams, 1, 1)
        mse_loss = F.mse_loss(s_norm, t_norm, reduction='none').mean(dim=1)  # (B*num_cams, H, W)
        distill_loss = (attention_weight * mse_loss).mean()
        loss_dict["distill_attn_mse"] = distill_loss
        
    elif loss_type == "attention_mse_v2":
        # variant 1: spatially aware attention_weight
        # weight from per-position cos_sim instead of the global mean
        spatial_attention_weight = 1.0 - cos_sim  # (B*num_cams, H, W)
        mse_loss = F.mse_loss(s_norm, t_norm, reduction='none').mean(dim=1)  # (B*num_cams, H, W)
        distill_loss = (spatial_attention_weight * mse_loss).mean()
        loss_dict["distill_attn_mse_v2"] = distill_loss
        
    elif loss_type == "attention_mse_v3":
        # variant 2: focal combined with attention_mse
        # hard-example weighting and spatial attention together
        s_norm = F.normalize(s_feat_proj_resized, dim=1)
        t_norm = F.normalize(t_feat, dim=1)
        spatial_attention_weight = 1.0 - cos_sim
        mse_loss = F.mse_loss(s_norm, t_norm, reduction='none').mean(dim=1)
        
        # focal weighting
        focal_weight = torch.pow(spatial_attention_weight.detach(), focal_gamma)
        
        distill_loss = (focal_weight * mse_loss).mean()
        loss_dict["distill_attn_mse_v3"] = distill_loss
        
    elif loss_type == "attention_mse_v4":
        # cosine + mse mix, weighted by spatial attention
        cosine_loss = base_dist
        mse_loss = F.mse_loss(s_norm, t_norm, reduction='none').mean(dim=1)
        
        spatial_attention_weight = 1.0 - cos_sim
        
        # mixed loss
        hybrid_loss = 0.7 * cosine_loss + 0.3 * mse_loss
        distill_loss = (spatial_attention_weight * hybrid_loss).mean()
        loss_dict["distill_attn_mse_v4"] = distill_loss
        
    elif loss_type == "attention_mse_v5":
        # refinement of attention_mse:
        # key changes:
        # 1. spatially aware weights instead of the global mean
        # 2. mild focal weighting (gamma 1.5)
        spatial_attention_weight = 1.0 - cos_sim  # spatially aware, [B, H, W]
        
        # mse term
        mse_loss = F.mse_loss(s_norm, t_norm, reduction='none').mean(dim=1)  # [B, H, W]
        
        # mild focal weighting (gamma=1.5)
        focal_weight = torch.pow(spatial_attention_weight.detach(), 1.5)
        
        # weighted loss
        distill_loss = (focal_weight * mse_loss).mean()
        loss_dict["distill_attn_mse_v5"] = distill_loss
        
    elif loss_type == "attention_mse_v6":
        # spatially aware, slightly stronger but still mild
        # same as the original but with the global-mean weight replaced by a spatial one
        spatial_attention_weight = 1.0 - cos_sim
        mse_loss = F.mse_loss(s_norm, t_norm, reduction='none').mean(dim=1)
        distill_loss = (spatial_attention_weight * mse_loss).mean()
        loss_dict["distill_attn_mse_v6"] = distill_loss
        
    elif loss_type == "attention_mse_v7":
        # conservative blend: 50% global weight + 50% spatial weight
        # keeps the original behaviour with a little spatial awareness added
        global_attention_weight = (1.0 - cos_sim.mean(dim=[1, 2], keepdim=True))  # [B,1,1]
        spatial_attention_weight = 1.0 - cos_sim  # [B, H, W]
        
        # blended weight
        final_weight = 0.5 * global_attention_weight + 0.5 * spatial_attention_weight
        
        mse_loss = F.mse_loss(s_norm, t_norm, reduction='none').mean(dim=1)
        distill_loss = (final_weight * mse_loss).mean()
        loss_dict["distill_attn_mse_v7"] = distill_loss
        
    elif loss_type == "attention_mse_v8":
        # targets ah/gp artifacts: v7 plus very mild suppression of candidate regions
        # suppression applies only where teacher is very weak and student unusually strong
        global_attention_weight = (1.0 - cos_sim.mean(dim=[1, 2], keepdim=True))
        spatial_attention_weight = 1.0 - cos_sim
        final_weight = 0.5 * global_attention_weight + 0.5 * spatial_attention_weight
        mse_loss = F.mse_loss(s_norm, t_norm, reduction='none').mean(dim=1)
        
        # ---- mild suppression of artifact candidates ----
        # normalize teacher and student spatial activations
        s_mag = torch.norm(s_feat_proj_resized, dim=1)
        t_mag = torch.norm(t_feat, dim=1)
        s_mag_norm = s_mag / (s_mag.mean(dim=[1, 2], keepdim=True) + 1e-8)
        t_mag_norm = t_mag / (t_mag.mean(dim=[1, 2], keepdim=True) + 1e-8)
        
        # teacher < 0.2 (very weak) and student > 1.2 (unusually strong) = candidate
        artifact_mask = (t_mag_norm < 0.2).float() * (s_mag_norm > 1.2).float()
        mild_suppression = (artifact_mask * s_mag).mean()
        
        # mostly v7 distillation plus a 0.1-weight mild suppression term
        distill_loss = (final_weight * mse_loss).mean() + 0.1 * mild_suppression
        loss_dict["distill_attn_mse_v8"] = distill_loss
        loss_dict["distill_suppression"] = mild_suppression
        
    elif loss_type == "focal_cosine":
        # focal cosine loss: weight hard (far) examples
        # focal_weight = (base_dist)^gamma
        focal_weight = torch.pow(base_dist.detach(), focal_gamma)
        distill_loss = (focal_weight * base_dist).mean()
        loss_dict["distill_focal"] = distill_loss
        
    elif loss_type == "focal_mse":
        # focal + mse
        s_norm = F.normalize(s_feat_proj_resized, dim=1)
        t_norm = F.normalize(t_feat, dim=1)
        mse_loss = F.mse_loss(s_norm, t_norm, reduction='none').mean(dim=1)  # (B*num_cams, H, W)
        focal_weight = torch.pow(base_dist.detach(), focal_gamma)
        distill_loss = (focal_weight * mse_loss).mean()
        loss_dict["distill_focal_mse"] = distill_loss
        
    elif loss_type == "hybrid":
        # focal cosine plus a small-weight mse
        focal_weight = torch.pow(base_dist.detach(), focal_gamma)
        focal_loss = (focal_weight * base_dist).mean()
        mse_loss = F.mse_loss(s_norm, t_norm)
        distill_loss = focal_loss + 0.1 * mse_loss
        loss_dict["distill_focal"] = focal_loss
        loss_dict["distill_mse"] = mse_loss
        loss_dict["distill_hybrid"] = distill_loss
    
    # === artifact-targeted objectives ===
    elif loss_type == "anti_artifact":
        # anti-artifact loss:
        # 1. where teacher activation is weak, damp student over-activation
        # 2. where teacher activation is strong, align with focal weighting
        t_mag_normalized = t_mag / (t_mag.mean(dim=[1, 2], keepdim=True) + 1e-8)
        weak_teacher_mask = (t_mag_normalized < 0.3).float()  # teacher weak-activation region
        strong_teacher_mask = 1.0 - weak_teacher_mask
        
        # weak regions: damp only the excess, do not zero the student out
        # find unusually high student activations (possible artifacts)
        s_mag_normalized = s_mag / (s_mag.mean(dim=[1, 2], keepdim=True) + 1e-8)
        # suppress only where teacher is weak and student unusually strong
        artifact_candidate_mask = weak_teacher_mask * (s_mag_normalized > 1.5).float()
        
        suppression_loss = (artifact_candidate_mask * s_mag).mean()
        
        # strong regions: focal alignment
        focal_weight = torch.pow(base_dist.detach() * strong_teacher_mask, focal_gamma)
        alignment_loss = (focal_weight * base_dist).mean()
        
        distill_loss = alignment_loss + 0.5 * suppression_loss
        loss_dict["distill_alignment"] = alignment_loss
        loss_dict["distill_suppression"] = suppression_loss
        loss_dict["distill_anti_artifact"] = distill_loss
        
    elif loss_type == "anti_artifact_v2":
        # gentler anti-artifact variant
        # idea: do not suppress student activation, make its spatial distribution match the teacher
        t_mag_normalized = t_mag / (t_mag.sum(dim=[1, 2], keepdim=True) + 1e-8)
        s_mag_normalized = s_mag / (s_mag.sum(dim=[1, 2], keepdim=True) + 1e-8)
        
        # spatial distribution alignment (kl)
        spatial_distribution_loss = F.kl_div(
            s_mag_normalized.log(), t_mag_normalized, 
            reduction='batchmean', log_target=False
        )
        
        # feature alignment
        focal_weight = torch.pow(base_dist.detach(), focal_gamma)
        alignment_loss = (focal_weight * base_dist).mean()
        
        distill_loss = alignment_loss + 0.3 * spatial_distribution_loss
        loss_dict["distill_alignment"] = alignment_loss
        loss_dict["distill_spatial"] = spatial_distribution_loss
        loss_dict["distill_anti_artifact_v2"] = distill_loss
        
    elif loss_type == "spatial_focal":
        # spatially aware focal loss:
        # use the teacher's spatial gradient; edge/texture regions get a higher gamma
        # teacher spatial gradient as a proxy for texture complexity
        t_gradient_h = torch.abs(t_feat[:, :, 1:, :] - t_feat[:, :, :-1, :])
        t_gradient_w = torch.abs(t_feat[:, :, :, 1:] - t_feat[:, :, :, :-1])
        t_gradient = F.pad(t_gradient_h.mean(dim=1), (0, 0, 1, 0)) + \
                     F.pad(t_gradient_w.mean(dim=1), (1, 0, 0, 0))
        
        # gamma from gradient: large gradient (textured) -> small gamma (treat evenly);
        t_gradient_normalized = t_gradient / (t_gradient.mean(dim=[1, 2], keepdim=True) + 1e-8)
        dynamic_gamma = focal_gamma * (1.0 - t_gradient_normalized * 0.5)
        
        spatial_focal_weight = torch.pow(base_dist.detach(), dynamic_gamma)
        distill_loss = (spatial_focal_weight * base_dist).mean()
        loss_dict["distill_spatial_focal"] = distill_loss
    elif loss_type == "structure_diversity":
        # 1. alignment base: mse, unnormalized
        mse_loss = F.mse_loss(s_feat_proj_resized, t_feat)
        
        # 2. direction: cosine, as a complement
        s_norm = F.normalize(s_feat_proj_resized, dim=1)
        t_norm = F.normalize(t_feat, dim=1)
        cos_loss = (1 - (s_norm * t_norm).sum(dim=1).mean())
        
        # 3. magnitude
        s_mag = torch.norm(s_feat_proj_resized, dim=1)
        t_mag = torch.norm(t_feat, dim=1)
        mag_loss = F.mse_loss(s_mag, t_mag)
        
        # combine (weights tunable)
        distill_loss = mse_loss + 0.5 * cos_loss + 0.5 * mag_loss
        loss_dict["distill_mse"] = mse_loss
        loss_dict["distill_cos"] = cos_loss
        loss_dict["distill_mag"] = mag_loss
    
    elif loss_type in ("structure_diversity_nomag", "structure_diversity_nocos",
                       "structure_diversity_bal"):
        # ablation variants of the three-term loss. terms are defined identically to
        # structure_diversity; only which terms enter the total and their weights change.
        mse_loss = F.mse_loss(s_feat_proj_resized, t_feat)
        s_n = F.normalize(s_feat_proj_resized, dim=1)
        t_n = F.normalize(t_feat, dim=1)
        cos_loss = (1 - (s_n * t_n).sum(dim=1).mean())
        s_m = torch.norm(s_feat_proj_resized, dim=1)
        t_m = torch.norm(t_feat, dim=1)
        mag_loss = F.mse_loss(s_m, t_m)

        if loss_type == "structure_diversity_nomag":
            distill_loss = mse_loss + 0.5 * cos_loss
        elif loss_type == "structure_diversity_nocos":
            distill_loss = mse_loss + 0.5 * mag_loss
        else:                                    # structure_diversity_bal
            # weights from measured mid-run magnitudes mse .072 / cos .233 / mag 4.295,
            # chosen so each term contributes roughly 1/3
            distill_loss = 1.0 * mse_loss + 0.31 * cos_loss + 0.017 * mag_loss

        # always log every term, even one excluded from the total, or ablation arms cannot be read against each other
        loss_dict["distill_mse"] = mse_loss
        loss_dict["distill_cos"] = cos_loss
        loss_dict["distill_mag"] = mag_loss

    elif loss_type in ("structure_diversity_mseonly", "structure_diversity_cosonly"):
        # naive baselines: keep one of the three terms, everything else identical to
        # structure_diversity, so 'new loss vs naive' carries no implementation difference.
        mse_loss = F.mse_loss(s_feat_proj_resized, t_feat)
        s_n = F.normalize(s_feat_proj_resized, dim=1)
        t_n = F.normalize(t_feat, dim=1)
        cos_loss = (1 - (s_n * t_n).sum(dim=1).mean())
        if loss_type == "structure_diversity_mseonly":
            distill_loss = mse_loss
        else:
            distill_loss = cos_loss
        loss_dict["distill_mse"] = mse_loss
        loss_dict["distill_cos"] = cos_loss

    elif loss_type == "magnitude_aware":
        # magnitude-aware loss: align direction (cosine) and magnitude
        cosine_loss = base_dist.mean()
        
        # magnitude alignment
        magnitude_loss = F.mse_loss(s_mag, t_mag)
        
        # combine
        distill_loss = cosine_loss + 0.3 * magnitude_loss
        loss_dict["distill_cosine"] = cosine_loss
        loss_dict["distill_magnitude"] = magnitude_loss
        loss_dict["distill_magnitude_aware"] = distill_loss
        
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return distill_loss, loss_dict


def collect_block_feats(feats, num_cams, nprefix):
    """stack the num_cams block outputs the hook collected into [B*num_cams, C, gh, gw] and clear the list.

    the backbone is called per camera, so feats holds num_cams tensors of [B, 1+N, C].
    stack(dim=1) to [B, num_cams, C, gh, gw] then flatten, which is the only order that
    matches t_feat (built from image_data.view(B*num_cams, ...)), batch-major.
    consume and clear: otherwise every iteration accumulates and holds graph references -> oom.
    """
    assert len(feats) == num_cams, f"hook fired {len(feats)} times, expected num_cams={num_cams}"
    per_cam = []
    for f in feats:
        pt = f[:, nprefix:, :]                     # [B, N, C]  N = gh*gw
        n_tok = pt.shape[1]
        side = int(round(n_tok ** 0.5))
        assert side * side == n_tok, f"non-square patch grid: N={n_tok}"
        per_cam.append(pt.transpose(1, 2).reshape(pt.shape[0], pt.shape[2], side, side))
    out = torch.stack(per_cam, dim=1)              # [B, num_cams, C, side, side]
    feats.clear()
    return out.reshape(-1, out.shape[2], out.shape[3], out.shape[4])


def spatial_block_shuffle(t_feat, gh, gw):
    """permute block positions within the student patch grid, independently per sample.

    t_feat [N,C,H,W] is the teacher at 224x224; gh/gw is the student grid side (16x16).
    returns the same shape: each 14x14 block keeps its contents, only its position moves.

    per-sample and re-randomized every step -- a fixed permutation would let the student
    """
    n, c, h, w = t_feat.shape
    if h % gh or w % gw:
        raise ValueError(f"teacher features {h}x{w} are not divisible by the student grid {gh}x{gw}; "
                         f"block permutation needs an exact division")
    ph, pw = h // gh, w // gw
    x = t_feat.view(n, c, gh, ph, gw, pw).permute(0, 1, 2, 4, 3, 5)
    x = x.reshape(n, c, gh * gw, ph * pw)
    idx = torch.argsort(torch.rand(n, gh * gw, device=t_feat.device), dim=1)
    x = torch.gather(x, 2, idx[:, None, :, None].expand(-1, c, -1, ph * pw))
    x = x.view(n, c, gh, gw, ph, pw).permute(0, 1, 2, 4, 3, 5)
    return x.reshape(n, c, h, w)


def batch_shuffle_teacher(t_feat, num_cams, gen=None):
    """take the teacher features of a DIFFERENT image: spatial structure intact, only the

    t_feat [B*num_cams, C, H, W], row index = b*num_cams + cam.
    a non-zero roll along the batch dim guarantees NO sample receives its own target,
    whereas a random permutation self-matches about 1/B of the time and those samples

    permute within the same camera only: crossing cameras would hand cam_high's target
    to a wrist camera and change the camera statistics too, adding a second variable.
    """
    n, c, h, w = t_feat.shape
    if num_cams <= 0 or n % num_cams:
        raise ValueError(f"t_feat has {n} rows, not a multiple of num_cams={num_cams}")
    b = n // num_cams
    if b < 2:
        # the last batch of an epoch can hold a single sample. that is a normal edge of the
        # data pipeline, not a bug -- raising would kill a 9-hour arm at the end of epoch 0
        # (it did once). report infeasible and let the caller fall back.
        return None
    x = t_feat.view(b, num_cams, c, h, w)
    k = int(torch.randint(1, b, (1,), generator=gen))
    idx = (torch.arange(b, device=t_feat.device) + k) % b
    return x[idx].reshape(n, c, h, w)


def _sum_mount_losses(mounts, t_feat, loss_type, focal_gamma, shuffle_teacher=False,
                      shuffle_scope="space", num_cams=None):
    """compute the distillation loss at each mount point and sum.

    each point carries w=1.0, faithful to the single-mount setting that produced the
    reported result. the cost is that total distillation strength grows linearly with the
    number of mounts (about 2x for two) -- a known confound that must be reported.
    """
    if shuffle_teacher:
        # permute on the GRADIENT PATH only. the cos_sim / t_norm monitors outside this
        # function still use the aligned teacher, so cos_sim in a permuted arm's log reads
        gh, gw = mounts[0][1].shape[-2:]
        if not getattr(_sum_mount_losses, "_shuf_announced", False):
            _desc = ("block permutation within the image" if shuffle_scope == "space"
                     else "across images, non-zero roll within a camera")
            # announce once: the cos_sim in the log uses the UNPERMUTED teacher and proves
            # nothing about whether this branch ran. without this line a silently inactive
            print(f"[distill] permuted teacher active: scope={shuffle_scope} -- {_desc}"
                  f" (grid {gh}x{gw}, re-randomized every step)", flush=True)
            _sum_mount_losses._shuf_announced = True
        if shuffle_scope == "space":
            t_feat = spatial_block_shuffle(t_feat, gh, gw)
        elif shuffle_scope == "batch":
            if num_cams is None:
                raise ValueError("cross-image permutation needs num_cams; the caller did not pass it")
            _bs = batch_shuffle_teacher(t_feat, num_cams)
            if _bs is None:
                # only one sample in this batch, so cross-image permutation is undefined.
                # fall back to WITHIN-image permutation, never to the aligned teacher.
                _sum_mount_losses._b1_fallback = getattr(
                    _sum_mount_losses, "_b1_fallback", 0) + 1
                if _sum_mount_losses._b1_fallback in (1, 100, 1000, 10000):
                    print(f"[distill] cross-image permutation hit a batch=1 tail, fell back to within-image"
                          f" ({_sum_mount_losses._b1_fallback} times so far)", flush=True)
                t_feat = spatial_block_shuffle(t_feat, gh, gw)
            else:
                t_feat = _bs
        else:
            raise ValueError(f"unknown shuffle_scope: {shuffle_scope!r}")
    total = None
    out = {}
    multi = len(mounts) > 1
    for k, sf in mounts:
        up = F.interpolate(sf, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
        dl, dd = compute_distillation_loss(up, t_feat, loss_type=loss_type,
                                          focal_gamma=focal_gamma)
        total = dl if total is None else total + dl
        sfx = f"_b{k}" if multi else ""
        for name, val in dd.items():
            out[name + sfx] = val
    return total, out


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)
    
    use_distillation = policy_config.get("use_distillation", False)
    distill_at_block = config.get("distill_at_block", None)
    if isinstance(distill_at_block, int):
        distill_at_block = [distill_at_block]        # backward compatible: older configs passed a single int
    # one consume-and-clear buffer per mount point
    student_block_feats = {k: [] for k in (distill_at_block or [])}
    distill_loss_type = config.get("distill_loss_type", "focal_cosine")
    distill_focal_gamma = config.get("distill_focal_gamma", 2.0)
    distill_shuffle_teacher = config.get("distill_shuffle_teacher", False)
    distill_shuffle_scope = config.get("distill_shuffle_scope", "space")
    teacher_model = None
    teacher_features = {}
    teacher_vis_dir = os.path.join(ckpt_dir, "teacher_visualizations")
    teacher_vis_interval = max(int(config.get("teacher_vis_interval", 500)), 0)
    
    # init the stats csv
    artifact_stats_path = os.path.join(ckpt_dir, "artifact_stats.csv")
    if use_distillation:
        with open(artifact_stats_path, "w", newline="") as f:
            f.write("epoch,step,s_max,s_mean,t_max,t_mean,cos_sim_mean,cos_sim_std\n")
            
    if use_distillation:
        import sys
        segmentation_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Segmentation"))
        sys.path.append(segmentation_dir)
        from unet import UNet
        from evaluate import evaluate_teacher_tensor
        
        # one teacher per task. with ACT_TEACHER_CKPT unset this matches the original path.
        ckpt_path = os.environ.get("ACT_TEACHER_CKPT") or os.path.join(
            segmentation_dir, "checkpoints", "best_unet_model.pth")
        print(f"[distill] teacher weights: {ckpt_path}", flush=True)
        if not os.path.exists(ckpt_path):
            print(f"Warning: UNet checkpoint not found at {ckpt_path}. Using dummy weights for testing.")
            num_classes = 1
            teacher_model = UNet(n_channels=3, n_classes=num_classes)
        else:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            num_classes = checkpoint.get('num_classes', 1)
            teacher_model = UNet(n_channels=3, n_classes=num_classes)
            teacher_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            
        teacher_model.cuda()
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        os.makedirs(teacher_vis_dir, exist_ok=True)
        
        def teacher_hook(model, input, output):
            teacher_features['up4'] = output
        teacher_model.up4.register_forward_hook(teacher_hook)

        # mount the distillation target on a given block (pre-norm) rather than the final layer (post-norm)
        if distill_at_block:
            _body = policy.model.backbones[0][0].body
            student_nprefix = int(getattr(_body, "num_prefix_tokens", 1))
            for _k in distill_at_block:
                # capture _k via a default argument: a direct closure reference would point every hook at the loop's last value
                _body.blocks[_k].register_forward_hook(
                    lambda m, i, o, kk=_k: student_block_feats[kk].append(o))
            print(f"[distill] target = blocks{distill_at_block} output (pre-norm), "
                  f"num_prefix_tokens={student_nprefix}")
            if len(distill_at_block) > 1:
                print(f"[distill] multi-mount: {len(distill_at_block)} points at w=1.0 summed "
                      f"-> total strength about {len(distill_at_block)}x (known confound)")
        else:
            student_nprefix = None
            print("[distill] target = forward_features output (post-norm, final layer)")

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    
    resume_ckpt = config.get("resume_ckpt")
    start_epoch = 0
    if resume_ckpt and os.path.isfile(resume_ckpt):
        print(f"Resuming weights from {resume_ckpt}")
        checkpoint = torch.load(resume_ckpt, map_location='cuda')
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["model_state_dict"])
        else:
            policy.load_state_dict(checkpoint)
        
        # Try to infer starting epoch from filename
        import re
        match = re.search(r"epoch_(\d+)", resume_ckpt)
        if match:
            start_epoch = int(match.group(1))
            print(f"Inferred starting epoch: {start_epoch}")
            
    best_ckpt_info = (start_epoch, min_val_loss, deepcopy(policy.state_dict()))

    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f"\nEpoch {epoch}")
        # validation
        with torch.inference_mode():
            policy.eval()
            if use_distillation:
                teacher_model.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                
                # --- Distillation for validation ---
                if use_distillation:
                    s_feat = forward_dict["s_feat"]
                    
                    # Get teacher feature via hook for all cameras
                    image_data = data[0].cuda()
                    B, num_cams, C, H, W = image_data.shape
                    
                    with torch.no_grad():
                        images_reshaped = image_data.view(B * num_cams, C, H, W)
                        
                        # --- Pre-processing Interceptor (Resize + Normalize) ---
                        images_resized = F.interpolate(images_reshaped, size=(224, 224), mode='bilinear', align_corners=False)
                        mean = torch.tensor([0.485, 0.456, 0.406], device=images_resized.device).view(1, 3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225], device=images_resized.device).view(1, 3, 1, 1)
                        images_normalized = (images_resized - mean) / std
                        
                        _ = teacher_model(images_normalized)
                        t_feat = teacher_features['up4'] # (B * num_cams, 768, 224, 224); up4 channels fixed to 768 by unet.py's C4

                    if teacher_vis_interval > 0 and batch_idx == 0 and epoch % teacher_vis_interval == 0:
                        vis_path = os.path.join(teacher_vis_dir, f"val_epoch_{epoch:04d}.png")
                        evaluate_teacher_tensor(
                            teacher_model,
                            image_data[:1, 0, ...],
                            vis_path,
                            device=image_data.device,
                        )
                        
                    # Flatten student feature along batch dimension
                    if not distill_at_block:
                        _, _, C_s, H_s, W_s = s_feat.shape
                        mounts = [(None, s_feat.reshape(B * num_cams, C_s, H_s, W_s))]
                    else:
                        mounts = [(k, collect_block_feats(
                            student_block_feats[k], num_cams, student_nprefix))
                            for k in distill_at_block]
                    s_feat_reshaped = mounts[0][1]
                    
                    # Map student feature dimension to out_channels (UNet up4 dimension)
                    # distill_proj = policy.model.distill_proj if hasattr(policy, 'model') and hasattr(policy.model, 'distill_proj') else policy.distill_proj
                    # s_feat_proj = distill_proj(s_feat_reshaped)
                    
                    # Resize student feature to align with teacher feature spatial resolution
                    # s_feat_proj_resized = F.interpolate(
                    #     s_feat_proj, size=t_feat.shape[-2:], mode='bilinear', align_corners=False
                    # )
                    s_feat_proj_resized = F.interpolate(
                        s_feat_reshaped, size=t_feat.shape[-2:], mode='bilinear', align_corners=False
                    )
                    
                    # Calculate cosine similarity and distillation loss
                    cos_sim = F.cosine_similarity(s_feat_proj_resized, t_feat, dim=1)
                    distill_loss = 1.0 - cos_sim.mean()
                    # distill_loss = 1.0 - torch.sigmoid(cos_sim).mean()
                    
                    # read distill_weight off the policy (default 10.0) so the weight takes effect
                    actual_distill_weight = getattr(policy, 'distill_weight', 1.0)
                    forward_dict["loss"] = forward_dict["loss"] + distill_loss * actual_distill_weight
                    forward_dict["distill_loss"] = distill_loss
                    
                # Remove s_feat to avoid memory leak during dict accumulation
                if "s_feat" in forward_dict:
                    del forward_dict["s_feat"]
                epoch_dicts.append(detach_dict(forward_dict))
                if use_distillation:
                    del images_reshaped, images_resized, images_normalized, t_feat

            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            val = v.item() if hasattr(v, 'item') else v
            summary_string += f"{k}: {val:.3f} "
        print(f"Val details: {summary_string}")

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict["loss"]
            
            # --- Distillation ---
            if use_distillation:
                s_feat = forward_dict["s_feat"]
                # every camera view takes part in distillation: num_cams folds into the batch dim below
                # (an earlier comment claimed only index 0 / head_cam was used; that was wrong)
                # ACT image_data shape: (B, num_cams, C, H, W)
                image_data = data[0].cuda()
                B, num_cams, C, H, W = image_data.shape
                
                with torch.no_grad():
                    # fold all views into the batch dim so one teacher pass covers them all
                    # (B, num_cams, C, H, W) -> (B * num_cams, C, H, W)
                    images_reshaped = image_data.view(B * num_cams, C, H, W)
                    
                    # --- Pre-processing Interceptor (Resize + Normalize) ---
                    images_resized = F.interpolate(images_reshaped, size=(224, 224), mode='bilinear', align_corners=False)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=images_resized.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=images_resized.device).view(1, 3, 1, 1)
                    images_normalized = (images_resized - mean) / std
                    
                    # Forward through teacher UNet and capture the hooked up4 feature
                    _ = teacher_model(images_normalized)
                    t_feat = teacher_features['up4'] # (B * num_cams, 768, 224, 224); matches vit-b embed_dim so no projection is needed
                    
                    if teacher_vis_interval > 0 and batch_idx == 0 and epoch % teacher_vis_interval == 0:
                        vis_path = os.path.join(teacher_vis_dir, f"train_epoch_{epoch:04d}.png")
                        # visualize one image from the first view (cam_high) only, to keep the figure small
                        evaluate_teacher_tensor(
                            teacher_model,
                            image_data[:1, 0, ...],
                            vis_path,
                            device=image_data.device,
                        )
                
                # Flatten student feature along batch dimension
                if not distill_at_block:
                    _, _, C_s, H_s, W_s = s_feat.shape
                    mounts = [(None, s_feat.reshape(B * num_cams, C_s, H_s, W_s))]
                else:
                    mounts = [(k, collect_block_feats(
                        student_block_feats[k], num_cams, student_nprefix))
                        for k in distill_at_block]
                s_feat_reshaped = mounts[0][1]
                
                # Map student feature dimension to out_channels (UNet up4 dimension)
                # distill_proj = policy.model.distill_proj if hasattr(policy, 'model') and hasattr(policy.model, 'distill_proj') else policy.distill_proj
                # s_feat_proj = distill_proj(s_feat_reshaped)                
                # Interpolate student feature map to match the spatial resolution of the teacher's backbone feature
                # s_feat_proj_resized = F.interpolate(
                #     s_feat_proj, size=t_feat.shape[-2:], mode='bilinear', align_corners=False
                # )

                s_feat_proj_resized = F.interpolate(
                    s_feat_reshaped, size=t_feat.shape[-2:], mode='bilinear', align_corners=False
                )

                
                # distillation loss
                distill_loss, distill_loss_dict = _sum_mount_losses(
                    mounts, t_feat, distill_loss_type, distill_focal_gamma,
                    shuffle_teacher=distill_shuffle_teacher,
                    shuffle_scope=distill_shuffle_scope, num_cams=num_cams)
                
                # stats
                # must use s_feat_reshaped, not s_feat: s_feat is (B, num_cams, C, H, W) and
                # taking a norm over dim=1 gives a norm ACROSS THE 3 CAMERAS, not the 768-dim
                # feature norm, which is not the quantity t_feat's channel norm measures.
                s_norm = torch.norm(s_feat_reshaped, dim=1, keepdim=True)
                t_norm = torch.norm(t_feat, dim=1, keepdim=True)
                
                s_max = s_norm.max().item()
                s_mean = s_norm.mean().item()
                t_max = t_norm.max().item()
                t_mean = t_norm.mean().item()
                
                # cosine similarity stats, for monitoring
                s_norm_feat = F.normalize(s_feat_proj_resized, dim=1)
                t_norm_feat = F.normalize(t_feat, dim=1)
                cos_sim = F.cosine_similarity(s_norm_feat, t_norm_feat, dim=1)
                cos_sim_mean = cos_sim.mean().item()
                cos_sim_std = cos_sim.std().item()
                
                # print stats
                if batch_idx % 50 == 0:
                    print(f"[Student-ViT] Max Norm: {s_max:.2f}, Mean Norm: {s_mean:.2f}")
                    print(f"[Teacher-CNN] Max Norm: {t_max:.2f}, Mean Norm: {t_mean:.2f}")
                    print(f"[Distillation] Cos Sim Mean: {cos_sim_mean:.4f}, Std: {cos_sim_std:.4f}")
                
                # append to csv
                with open(artifact_stats_path, "a", newline="") as f:
                    f.write(f"{epoch},{batch_idx},{s_max},{s_mean},{t_max},{t_mean},{cos_sim_mean},{cos_sim_std}\n")
                    
                # into forward_dict
                forward_dict["s_max_norm"] = torch.tensor(s_max, device=loss.device)
                forward_dict["s_mean_norm"] = torch.tensor(s_mean, device=loss.device)
                forward_dict["t_max_norm"] = torch.tensor(t_max, device=loss.device)
                forward_dict["t_mean_norm"] = torch.tensor(t_mean, device=loss.device)
                forward_dict["cos_sim_mean"] = torch.tensor(cos_sim_mean, device=loss.device)
                
                # merge losses
                actual_distill_weight = getattr(policy, 'distill_weight', 1.0)
                loss = loss + distill_loss * actual_distill_weight
                forward_dict["loss"] = loss
                forward_dict.update(distill_loss_dict)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Clear s_feat before appending to history to prevent memory leak
            if "s_feat" in forward_dict:
                del forward_dict["s_feat"]
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[- (batch_idx + 1):])
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            val = v.item() if hasattr(v, 'item') else v
            summary_string += f"{k}: {val:.3f} "
        print(f"Train details: {summary_string}")

        if (epoch + 1) % config['save_freq'] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch + 1}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            # plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}")

    # save training curves
    # plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() if hasattr(summary[key], 'item') else summary[key] for summary in train_history]
        val_values = [summary[key].item() if hasattr(summary[key], 'item') else summary[key] for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--batch_size", action="store", type=int, help="batch_size", required=True)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--num_epochs", action="store", type=int, help="num_epochs", required=True)
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)
    parser.add_argument("--lr_backbone", action="store", type=float, help="lr_backbone", required=False, default=1e-5)

    # for ACT
    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--hidden_dim", action="store", type=int, help="hidden_dim", required=False)
    parser.add_argument("--state_dim", action="store", type=int, help="state dim", required=True)
    parser.add_argument("--save_freq", action="store", type=int, help="save ckpt frequency", required=False, default=6000)
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--backbone_type", action="store", type=str, help="backbone type", required=False, default=None)
    parser.add_argument("--backbone", action="store", type=str, help="backbone name", required=False, default=None)
    parser.add_argument(
        "--backbone_pretrained",
        action="store",
        type=str2bool,
        help="use pretrained backbone",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--backbone_image_size",
        action="store",
        type=str,
        help="backbone input size, e.g. 224 or 224,224",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--use_distillation",
        action="store",
        type=str2bool,
        help="use UNet feature distillation",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--teacher_vis_interval",
        action="store",
        type=int,
        help="save teacher segmentation and up4 heatmap every N epochs during distillation; 0 disables",
        required=False,
        default=500,
    )
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--resume_ckpt", action="store", type=str, help="path to checkpoint to resume from", required=False, default=None)
    parser.add_argument(
        "--distill_loss_type",
        action="store",
        type=str,
        help="distillation loss type: cosine|mse|attention_mse|attention_mse_v2|attention_mse_v3|attention_mse_v4|attention_mse_v5|attention_mse_v6|attention_mse_v7|attention_mse_v8|final_ultimate|focal_cosine|focal_mse|hybrid|anti_artifact|anti_artifact_v2|spatial_focal|magnitude_aware",
        required=False,
        default="structure_diversity",
    )
    parser.add_argument(
        "--distill_focal_gamma",
        action="store",
        type=float,
        help="gamma parameter for focal loss",
        required=False,
        default=2.0,
    )
    parser.add_argument(
        "--distill_weight",
        action="store",
        type=float,
        help="weight for distillation loss",
        required=False,
        default=1.0,
    )
    parser.add_argument(
        "--distill_shuffle_scope",
        action="store",
        type=str,
        choices=["space", "batch"],
        default="space",
        help="scope of teacher permutation. space = block permutation within the image "
             "(measured: collapses the student toward spatially uniform features, "
             "max/mean 1.53 vs 4.3-4.5 for the aligned arm, so attribution is ambiguous); "
             "batch = across images, non-zero roll within a camera -- spatial structure "
    )
    parser.add_argument(
        "--distill_shuffle_teacher",
        action="store_true",
        help="spatial teacher-permutation control: permute the teacher features by student "
             "patch-grid blocks; block contents and global statistics are unchanged. isolates "
             "spatial alignment from the mere presence of an auxiliary regularizer -- the only "
    )
    parser.add_argument(
        "--distill_at_block",
        action="store",
        nargs="+",
        type=int,
        help="mount the distillation target on blocks[k] output (pre-norm) instead of the "
             "final layer (post-norm). unset -> final layer. the same block must also be "
             "unfrozen (--unfreeze_blocks b<k>) or no gradient reaches it.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--unfreeze_blocks",
        action="store",
        type=str,
        help="freeze-position ablation spec: none|all|first|middle|last|b<i>|b<i>-<j>|cls|norm|pos|patch_embed "
             "(comma-combinable, e.g. 'b8' / 'b9-11' / 'b11,cls'). "
             "Set -> freeze everything then unfreeze ONLY this. Unset -> legacy behaviour.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--freeze_last_layer_and_cls",
        action="store",
        type=str2bool,
        help="freeze last layer and CLS token of backbone",
        required=False,
        default=False,
    )

    main(vars(parser.parse_args()))
