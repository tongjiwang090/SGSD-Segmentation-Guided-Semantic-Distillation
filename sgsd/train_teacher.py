# -*- coding: utf-8 -*-
"""Segmentation teacher training.

Thin wrapper over ``Segmentation/train.py``; the mask mode is passed through
the environment variables that ``Segmentation/dataset.py`` already reads, so no
function there is modified.

Mask modes
----------
all      foreground = every non-background pixel (default, matches the paper)
target   foreground = only the listed actor ids, i.e. the manipulated objects

Examples
--------
python -m sgsd.train_teacher --data-dir policy/ACT/processed_data/sim-beat_block_hammer-demo_clean-50 \
    --save-dir Segmentation/checkpoints_beat_block_hammer

python -m sgsd.train_teacher --data-dir <...> --save-dir <...> \
    --mask-mode target --target-ids 178,184
"""
import argparse
import os
import runpy
import sys


def build_parser():
    p = argparse.ArgumentParser(
        prog="sgsd.train_teacher",
        description="Train the Attention U-Net segmentation teacher.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True,
                   help="processed episode directory containing .hdf5 files")
    p.add_argument("--save-dir", required=True,
                   help="output directory; use one per task or the anchor teacher is overwritten")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=8e-3)
    p.add_argument("--resume", default="", help="checkpoint to resume from; empty = from scratch")
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")

    p.add_argument("--mask-mode", choices=["all", "target"], default="all",
                   help="foreground definition")
    p.add_argument("--target-ids", default="",
                   help="comma-separated actor ids, required when --mask-mode target")
    p.add_argument("--seg-channel", type=int, default=0,
                   help="channel of the segmentation array to read")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.mask_mode == "target" and not a.target_ids.strip():
        sys.exit("--target-ids is required when --mask-mode target")

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ["TEACHER_SEG_MODE"] = a.mask_mode
    os.environ["TEACHER_SEG_IDS"] = a.target_ids
    os.environ["TEACHER_SEG_CH"] = str(a.seg_channel)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seg_dir = os.path.join(root, "Segmentation")
    data_dir = os.path.abspath(a.data_dir)
    save_dir = os.path.abspath(a.save_dir)

    sys.path.insert(0, seg_dir)
    os.chdir(seg_dir)

    sys.argv = ["train.py",
                "--data_dir", data_dir,
                "--save_dir", save_dir,
                "--epochs", str(a.epochs),
                "--batch_size", str(a.batch_size),
                "--lr", str(a.lr),
                "--resume", a.resume]
    runpy.run_path(os.path.join(seg_dir, "train.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
