import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List

import timm
import torch
import torchvision
from huggingface_hub import snapshot_download


BACKBONE_SPECS: Dict[str, Dict[str, str]] = {
    "resnet18": {
        "kind": "torchvision",
        "name": "resnet18",
    },
    "vit": {
        "kind": "timm",
        "name": "vit_base_patch16_224",
    },
    "dinov2": {
        "kind": "timm",
        "name": "vit_base_patch14_dinov2",
    },
    "clip": {
        "kind": "timm",
        "name": "vit_base_patch16_clip_224",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and cache ACT pretrained vision backbones."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to cache: all, resnet18, vit, dinov2, clip",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Instantiate each model with pretrained weights after caching",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of cached files when supported",
    )
    return parser.parse_args()


def expand_model_aliases(model_args: Iterable[str]) -> List[str]:
    expanded: List[str] = []
    for item in model_args:
        name = item.strip().lower()
        if name == "all":
            expanded.extend(BACKBONE_SPECS.keys())
        elif name in BACKBONE_SPECS:
            expanded.append(name)
        else:
            raise ValueError(f"Unsupported model alias: {item}")
    return list(dict.fromkeys(expanded))


def cache_torchvision_model(model_name: str):
    weights_enum = torchvision.models.get_model_weights(model_name).DEFAULT
    checkpoint_name = os.path.basename(weights_enum.url)
    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    checkpoint_path = checkpoint_dir / checkpoint_name

    print(f"[download] torchvision:{model_name}")
    state_dict = weights_enum.get_state_dict(progress=True, check_hash=True)
    print(f"[cached] {checkpoint_path}")
    return checkpoint_path, state_dict


def cache_timm_model(model_name: str, force: bool):
    model = timm.create_model(model_name, pretrained=False, num_classes=0, dynamic_img_size=True)
    pretrained_cfg = model.pretrained_cfg
    repo_id = pretrained_cfg.get("hf_hub_id")
    if not repo_id:
        raise RuntimeError(f"Missing hf_hub_id for timm model: {model_name}")

    print(f"[download] timm:{model_name} from {repo_id}")
    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=["*.json", "*.safetensors", "*.bin"],
        force_download=force,
    )
    print(f"[cached] {local_dir}")
    return Path(local_dir), None


def verify_model(alias: str):
    spec = BACKBONE_SPECS[alias]
    model_name = spec["name"]
    kind = spec["kind"]
    print(f"[verify] {alias} -> {model_name}")

    if kind == "torchvision":
        weights = torchvision.models.get_model_weights(model_name).DEFAULT
        model = getattr(torchvision.models, model_name)(weights=weights)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=0, dynamic_img_size=True)

    model.eval()
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"[ok] {alias} parameters={num_parameters}")


def main():
    args = parse_args()
    aliases = expand_model_aliases(args.models)

    print(f"[torch_cache] {Path(torch.hub.get_dir()) / 'checkpoints'}")
    print(f"[hf_cache] {Path.home() / '.cache' / 'huggingface' / 'hub'}")

    failures = []
    for alias in aliases:
        spec = BACKBONE_SPECS[alias]
        model_name = spec["name"]
        kind = spec["kind"]
        try:
            if kind == "torchvision":
                cache_torchvision_model(model_name)
            else:
                cache_timm_model(model_name, force=args.force)

            if args.verify:
                verify_model(alias)
        except Exception as exc:
            failures.append((alias, str(exc)))
            print(f"[failed] {alias}: {exc}")

    if failures:
        print("\nSummary: some downloads failed.")
        for alias, message in failures:
            print(f" - {alias}: {message}")
        raise SystemExit(1)

    print("\nSummary: all requested backbone weights are cached.")


if __name__ == "__main__":
    main()
