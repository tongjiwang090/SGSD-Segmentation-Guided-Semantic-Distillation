import os
from typing import Any, Dict

import yaml


BACKBONE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backbone_config.yml")
BACKBONE_CONFIG_KEYS = (
    "backbone_type",
    "backbone",
    "backbone_pretrained",
    "backbone_image_size",
)


def load_backbone_config() -> Dict[str, Any]:
    if not os.path.exists(BACKBONE_CONFIG_PATH):
        return {}
    with open(BACKBONE_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid backbone config in {BACKBONE_CONFIG_PATH}")
    return config


def apply_backbone_config(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(config)
    defaults = load_backbone_config()
    for key in BACKBONE_CONFIG_KEYS:
        if merged.get(key) is None and defaults.get(key) is not None:
            merged[key] = defaults[key]
    return merged
