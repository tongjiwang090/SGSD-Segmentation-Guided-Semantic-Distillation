# -*- coding: utf-8 -*-
"""SGSD policy training — single entry point for the three arms.

This is a thin wrapper: it builds the argument dict and calls
``imitate_episodes.main`` unchanged. No function in the underlying
implementation is modified or re-signed.

Arms
----
sgsd         segmentation-guided distillation at the final block (ours)
action-only  same trainable visual parameters, no teacher
frozen       frozen visual backbone

Examples
--------
# ours, anchor task
python -m sgsd.train_policy --arm sgsd --task beat_block_hammer \
    --teacher-ckpt Segmentation/checkpoints_beat_block_hammer/best_unet_model.pth

# frozen baseline
python -m sgsd.train_policy --arm frozen --task beat_block_hammer

# resume / fine-tune from a released checkpoint
python -m sgsd.train_policy --arm sgsd --task beat_block_hammer \
    --teacher-ckpt <...> --resume-ckpt checkpoints/beat_block_hammer/sgsd/policy_epoch_4000_seed_0.ckpt
"""
import argparse
import os
import sys

# Defaults of the configuration reported in the paper.
RECIPE = dict(
    policy_class="ACT",
    kl_weight=10,
    chunk_size=50,
    hidden_dim=512,
    batch_size=4,
    dim_feedforward=3200,
    num_epochs=4000,
    lr=1e-5,
    lr_backbone=1e-5,
    save_freq=4000,
    state_dim=14,
    backbone_type="dinov2",
    backbone="vit_base_patch14_dinov2",
    # booleans must stay python bools. imitate_episodes is normally reached through
    # argparse (type=str2bool) and tests them with `if flag:`, so a "false" string
    # would read as true and silently switch distillation on in the baseline arms.
    backbone_pretrained=True,
    backbone_image_size=224,
    distill_loss_type="structure_diversity",
    distill_weight=1.0,
    distill_at_block=[11],
    unfreeze_blocks="b11,cls,norm",
)

ARMS = {
    "sgsd":        dict(use_distillation=True,  unfreeze_blocks="b11,cls,norm"),
    "action-only": dict(use_distillation=False, unfreeze_blocks="b11,cls,norm"),
    "frozen":      dict(use_distillation=False, unfreeze_blocks="none"),
}


def build_parser():
    p = argparse.ArgumentParser(
        prog="sgsd.train_policy",
        description="Train an ACT policy with segmentation-guided semantic distillation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--arm", choices=sorted(ARMS), default="sgsd",
                   help="which arm to train")
    p.add_argument("--task", default="beat_block_hammer", help="RoboTwin task name")
    p.add_argument("--task-config", default="demo_clean",
                   help="task config under task_config/ (e.g. demo_clean, demo_distractor)")
    p.add_argument("--num-demos", type=int, default=50, help="demonstrations in the collection")
    p.add_argument("--ckpt-dir", default=None,
                   help="output directory (default: act_ckpt/<task>_<arm>_seed<seed>)")
    p.add_argument("--teacher-ckpt", default=None,
                   help="segmentation teacher .pth; required for --arm sgsd")
    p.add_argument("--resume-ckpt", default=None, help="policy checkpoint to resume from")

    p.add_argument("--epochs", type=int, default=RECIPE["num_epochs"])
    p.add_argument("--batch-size", type=int, default=RECIPE["batch_size"])
    p.add_argument("--lr", type=float, default=RECIPE["lr"])
    p.add_argument("--lr-backbone", type=float, default=RECIPE["lr_backbone"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    p.add_argument("--save-freq", type=int, default=RECIPE["save_freq"])

    p.add_argument("--backbone-type", default=RECIPE["backbone_type"],
                   choices=["dinov2", "vit", "clip"])
    p.add_argument("--backbone", default=RECIPE["backbone"])
    p.add_argument("--unfreeze", default=None,
                   help="override the trainable visual set, e.g. 'b11,cls,norm' or 'none'")

    p.add_argument("--mount-block", type=int, nargs="+", default=[11],
                   help="block(s) whose pre-LN output is supervised; several mount points "
                        "sum at w=1.0 each. -1 = post-LN final output")
    p.add_argument("--loss", default=RECIPE["distill_loss_type"],
                   choices=["structure_diversity", "structure_diversity_nomag",
                            "structure_diversity_nocos", "structure_diversity_bal",
                            "structure_diversity_mseonly", "structure_diversity_cosonly"],
                   help="distillation objective; the ablation variants drop one term")
    p.add_argument("--distill-weight", type=float, default=RECIPE["distill_weight"])
    p.add_argument("--shuffle-teacher", choices=["space", "batch"], default=None,
                   help="teacher-correspondence control: permute within image or across images")

    p.add_argument("--print-only", action="store_true",
                   help="print the resolved configuration and exit")
    return p


def build_config(a):
    cfg = dict(RECIPE)
    cfg.update(ARMS[a.arm])

    cfg["task_name"] = "sim-%s-%s-%d" % (a.task, a.task_config, a.num_demos)
    # posix separator: the training scripts and logs are read on Linux
    cfg["ckpt_dir"] = a.ckpt_dir or "act_ckpt/%s_%s_seed%d" % (a.task, a.arm, a.seed)
    cfg["num_epochs"] = a.epochs
    cfg["batch_size"] = a.batch_size
    cfg["lr"] = a.lr
    cfg["lr_backbone"] = a.lr_backbone
    cfg["seed"] = a.seed
    cfg["save_freq"] = a.save_freq
    cfg["backbone_type"] = a.backbone_type
    cfg["backbone"] = a.backbone
    cfg["distill_weight"] = a.distill_weight
    cfg["distill_loss_type"] = a.loss
    cfg["resume_ckpt"] = a.resume_ckpt

    if a.unfreeze is not None:
        cfg["unfreeze_blocks"] = a.unfreeze
    # post-LN supervision is expressed by passing no mount block.
    # no teacher -> no mount point, matching the baseline run_config of the
    # released checkpoints.
    if not cfg["use_distillation"] or any(k < 0 for k in a.mount_block):
        cfg["distill_at_block"] = None
    else:
        cfg["distill_at_block"] = list(a.mount_block)

    if a.shuffle_teacher:
        cfg["distill_shuffle_teacher"] = True
        cfg["distill_shuffle_scope"] = a.shuffle_teacher

    # keys imitate_episodes.main reads but the recipe does not set
    cfg.setdefault("eval", False)
    cfg.setdefault("onscreen_render", False)
    cfg.setdefault("temporal_agg", False)
    cfg.setdefault("distill_focal_gamma", 2.0)
    cfg.setdefault("freeze_last_layer_and_cls", False)
    cfg.setdefault("teacher_vis_interval", 1000)
    return cfg


def main(argv=None):
    a = build_parser().parse_args(argv)

    if a.arm == "sgsd" and not a.teacher_ckpt and not a.print_only:
        sys.exit("--teacher-ckpt is required for --arm sgsd "
                 "(train one with: python -m sgsd.train_teacher ...)")

    cfg = build_config(a)

    if a.print_only:
        for k in sorted(cfg):
            print("%-28s %s" % (k, cfg[k]))
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    if a.teacher_ckpt:
        # consumed by imitate_episodes when loading the frozen teacher
        os.environ["ACT_TEACHER_CKPT"] = os.path.abspath(a.teacher_ckpt)

    act_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "policy", "ACT")
    sys.path.insert(0, act_dir)
    os.chdir(act_dir)          # imitate_episodes resolves constants/utils by cwd

    import imitate_episodes
    return imitate_episodes.main(cfg)


if __name__ == "__main__":
    sys.exit(main())
