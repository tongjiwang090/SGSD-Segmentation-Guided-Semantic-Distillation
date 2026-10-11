# -*- coding: utf-8 -*-
"""Closed-loop evaluation of a trained policy checkpoint.

Thin wrapper over ``policy/ACT/run_eval_arm.sh``, which in turn calls
RoboTwin's ``script/eval_policy.py``. Nothing in either is modified.

Because the simulator lives in RoboTwin, this must run from the RoboTwin root
after the overlay has been applied (see README, step 1).

Examples
--------
python -m sgsd.evaluate --name bbh-sgsd \
    --ckpt-dir policy/ACT/act_ckpt/beat_block_hammer_sgsd_seed0

python -m sgsd.evaluate --name bbh-sgsd \
    --ckpt-dir policy/ACT/act_ckpt/beat_block_hammer_sgsd_seed0 \
    --ckpt-name policy_epoch_4000_seed_0.ckpt --trials 100 --task-config demo_distractor
"""
import argparse
import os
import subprocess
import sys


def build_parser():
    p = argparse.ArgumentParser(
        prog="sgsd.evaluate",
        description="Evaluate a policy checkpoint in closed loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--name", required=True,
                   help="arm name; also names the evaluation log")
    p.add_argument("--ckpt-dir", required=True,
                   help="checkpoint directory, relative to the RoboTwin root")
    p.add_argument("--ckpt-name", default="policy_epoch_4000_seed_0.ckpt",
                   help="checkpoint file inside --ckpt-dir")
    p.add_argument("--trials", type=int, default=100, help="completed rollouts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default="0")
    p.add_argument("--task-config", default="demo_clean",
                   help="must match the config the policy was trained on")
    p.add_argument("--print-only", action="store_true",
                   help="print the command and exit")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "policy", "ACT", "run_eval_arm.sh")
    if not os.path.isfile(script):
        sys.exit("missing %s" % script)

    cmd = ["bash", script, a.name, a.ckpt_dir, str(a.trials),
           str(a.seed), a.gpu, a.ckpt_name, a.task_config]

    if a.print_only:
        print(" ".join(cmd))
        return 0

    if not os.path.isdir("envs") or not os.path.isdir("script"):
        sys.exit("run this from the RoboTwin root (the simulator is not part of "
                 "this repository); see README step 1")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
