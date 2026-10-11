"""SGSD — Segmentation-Guided Semantic Distillation.

Command-line entry points. These are thin wrappers: they build arguments and
call the underlying implementations unchanged.

    python -m sgsd.train_teacher --help
    python -m sgsd.train_policy  --help
    python -m sgsd.evaluate      --help
"""
__all__ = ["train_teacher", "train_policy", "evaluate"]
