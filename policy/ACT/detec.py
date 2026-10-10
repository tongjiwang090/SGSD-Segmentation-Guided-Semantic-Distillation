#!/usr/bin/env python
"""
English `bash train.sh beat_block_hammer demo_clean 50 0 0 dinov2 vit_base_patch14_dinov2 true 224 true false`
English ACT English**English**.

===========================================================================
 train.sh English(English)
---------------------------------------------------------------------------
  $1 task_name         = beat_block_hammer
  $2 task_config       = demo_clean
  $3 expert_data_num   = 50
  $4 seed              = 0
  $5 gpu_id            = 0
  $6 backbone_type     = dinov2           <-- English
  $7 backbone          = vit_base_patch14_dinov2   <-- timm English
  $8 backbone_pretrained  = true          (English,English)
  $9 backbone_image_size = 224            <-- English(English)
  $10 use_distillation = true            (English,English)
  $11 freeze_last_layer_and_cls = false   (English,English)

=> imitate_episodes.py English checkpoint English:
     ckpt_dir  = ./act_ckpt/act-beat_block_hammer/demo_clean-50
=> English:
     policy_last.ckpt  (English,always)
     policy_best.ckpt  (English)
===========================================================================

English(English:English):
    # English ACT English,English train.sh English
    cd policy/ACT
    python detec.py

English(English ckpt English,English):
    python detec.py \
        --ckpt-dir  ./act_ckpt/act-beat_block_hammer/demo_clean-50 \
        --backbone  dinov2 \
        --image-size 224 \
        --device    cuda

English(English:English timm English DINOv2 English):
    python detec.py --compare-pretrained
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# English
# ---------------------------------------------------------------------------
_POLICY_ACT_DIR = Path(__file__).resolve().parent
if str(_POLICY_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_ACT_DIR))
_PROJECT_ROOT = _POLICY_ACT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from artifact_detector import (  # noqa: E402
    ACT_IMAGE_MEAN,
    ACT_IMAGE_STD,
    ArtifactDetector,
    ArtifactThresholds,
    BackboneLoader,
    DetectionResult,
    run_artifact_detection_on_act_ckpt,
)

# ---------------------------------------------------------------------------
# English(English train.sh beat_block_hammer demo_clean 50 ... dinov2 ... 224 )
# ---------------------------------------------------------------------------

DEFAULT_TASK_NAME = "beat_block_hammer"
DEFAULT_TASK_CONFIG = "demo_clean"
DEFAULT_EXPERT_DATA_NUM = 50
DEFAULT_BACKBONE_TYPE = "dinov2"
DEFAULT_BACKBONE_IMAGE_SIZE = 224
# English train.sh line 22:  ./act_ckpt/act-${task_name}/${task_config}-${expert_data_num}
DEFAULT_CKPT_DIR_RELATIVE = (
    f"./act_ckpt/act-{DEFAULT_TASK_NAME}"
    f"/{DEFAULT_TASK_CONFIG}-{DEFAULT_EXPERT_DATA_NUM}"
)

# English §4.2 + English UniRefiner schema.py English
DEFAULT_THRESHOLDS = ArtifactThresholds(
    fp=0.6,           # method.fp_gp_thres ≈ 0.5, English ACT 224 English 0.6 English
    gp=0.6,           # English fp English
    gp_sep_min=0.45,  # GP Eq.(2) English(b)English:English fp,English GP English
    ah_coeff=-0.3,    # English -0.25 English:English"English inflow"English AH
)

# English(English UniRefiner English fixed_reference.png,
#  English;English)
PREFERRED_REF_IMAGE = _POLICY_ACT_DIR / "UniRefiner-main" / "assets" / "backgrounds" / "fixed_reference.png"


# ============================================================================
# English / English
# ============================================================================

def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("detec")
    logger.setLevel(logging.INFO)
    while logger.handlers:  # English
        logger.handlers.pop()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


# ============================================================================
# English / English
# ============================================================================

def _ensure_reference_image() -> Path:
    """English UniRefiner English fixed_reference.png,English."""
    if PREFERRED_REF_IMAGE.exists():
        return PREFERRED_REF_IMAGE
    fallback = _POLICY_ACT_DIR / "artifact_outputs" / "_fallback_reference.png"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    if not fallback.exists():
        rng = np.random.default_rng(20260519)  # CVPR 2026
        arr = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr, "RGB").save(fallback)
    return fallback


def _synthetic_source_image_batch(n: int = 3, seed: int = 7) -> List[Path]:
    """English"English"English:English n English.

    English FP English 100%(English FP English),
    English + English"English",English CC3M English.
    """
    out_dir = _POLICY_ACT_DIR / "artifact_outputs" / "_synth_src"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    rng = np.random.default_rng(seed)
    for i in range(n):
        h = w = 224
        # 2D English + English => English
        y, x = np.mgrid[0:h, 0:w] / float(h)
        freq = rng.uniform(1.2, 4.2, size=6)
        phase = rng.uniform(0, 2 * np.pi, size=18)
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3):
            v = np.zeros_like(x)
            for k in range(3):
                v += rng.uniform(0.2, 0.9) * np.sin(2 * np.pi * (freq[2*k]*x + freq[2*k+1]*y) + phase[2*k+c])
            rgb[..., c] = (v - v.min()) / (v.max() - v.min() + 1e-9)
        # English
        rgb += rng.uniform(0, 0.08, size=(h, w, 3)).astype(np.float32)
        arr = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        p = out_dir / f"synth_src_{i:02d}.png"
        Image.fromarray(arr, "RGB").save(p)
        paths.append(p)
    return paths


# ============================================================================
# English & CSV / JSON English
# ============================================================================

@dataclass
class PerImageReport:
    ckpt: str
    src_image: str
    ref_image: str
    backbone_type: str
    image_size: int
    patch_grid: List[int]
    total_tokens: int
    fp_cnt: int;   fp_rate: float
    gp_cnt: int;   gp_rate: float
    ah_cnt: int;   ah_rate: float
    union_cnt: int; union_rate: float
    ah_z_mean: float; ah_z_std: float
    latency_ms: float


def _aggregate_and_write_report(
    reports: List[PerImageReport],
    out_csv: Path,
    out_json: Path,
    logger: logging.Logger,
) -> PerImageReport:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(PerImageReport.__dataclass_fields__.keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in reports:
            w.writerow(asdict(r))

    # English SUMMARY
    keys = ["fp_rate", "gp_rate", "ah_rate", "union_rate", "ah_z_mean", "ah_z_std", "latency_ms"]
    agg = {k: float(np.mean([getattr(r, k) for r in reports])) for k in keys}
    agg_std = {k + "_std": float(np.std([getattr(r, k) for r in reports])) for k in keys}
    summary = {
        "n_images": len(reports),
        "meta": {
            "ckpt": reports[0].ckpt if reports else "",
            "backbone_type": reports[0].backbone_type if reports else "",
            "image_size": reports[0].image_size if reports else 0,
            "patch_grid": reports[0].patch_grid if reports else [0, 0],
            "total_tokens": reports[0].total_tokens if reports else 0,
        },
        "mean": agg,
        "std": agg_std,
        "per_image": [asdict(r) for r in reports],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info("=" * 78)
    logger.info("PSEUDO-TOKEN DETECTION REPORT  (UniRefiner CVPR 2026 §3.1 taxonomy)")
    logger.info("=" * 78)
    logger.info(f"  ckpt           : {reports[0].ckpt if reports else 'N/A'}")
    logger.info(f"  backbone_type  : {reports[0].backbone_type if reports else 'N/A'}   "
                f"(patch_grid={reports[0].patch_grid if reports else '?'}, "
                f"total_tokens={reports[0].total_tokens if reports else '?'})")
    logger.info(f"  n_images eval  : {len(reports)}")
    logger.info("-" * 78)
    logger.info("  Artifact type  |  mean rate  ±   std      |  (FP/GP Eq.1/2,  AH z-score)")
    logger.info("-" * 78)
    def _row(name: str, k: str):
        m = agg[k] * 100; s = agg_std[k + "_std"] * 100
        logger.info(f"  {name:<14s} |  {m:6.2f}%   ± {s:6.2f}%")
    _row("FP (Fixed Pat.)", "fp_rate")
    _row("GP (Global Prox)", "gp_rate")
    _row("AH (Attn Hijack)", "ah_rate")
    _row("UNION (any)",     "union_rate")
    logger.info("-" * 78)
    logger.info(f"  mean latency   : {agg['latency_ms']:.1f} ± {agg_std['latency_ms_std']:.1f} ms  "
                f"(CPU single-batch baseline)")
    logger.info(f"  ah_z_mean      : {agg['ah_z_mean']:+.3f}   ah_z_std : {agg['ah_z_std']:.3f}")
    logger.info(f"  CSV report     : {out_csv}")
    logger.info(f"  JSON summary   : {out_json}")
    logger.info("=" * 78)

    # English"English"English
    avg = deepcopy(reports[0]) if reports else PerImageReport("", "", "", "", 0, [0,0], 0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)
    for k in ("fp_rate","gp_rate","ah_rate","union_rate","ah_z_mean","ah_z_std","latency_ms"):
        setattr(avg, k, agg[k])
    return avg


# ============================================================================
# English:timm English(English baseline)
# ============================================================================

def run_pretrained_baseline(
    backbone_type: str,
    src_images: List[Path],
    ref_image: Path,
    out_dir: Path,
    *,
    image_size: int = 224,
    device: str = "cpu",
    thresholds: Optional[ArtifactThresholds] = None,
    logger: Optional[logging.Logger] = None,
) -> PerImageReport:
    log = logger or logging.getLogger("detec")
    log.info(f"[BASELINE] timm pretrained {backbone_type} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0_setup = time.perf_counter()
    det = ArtifactDetector.from_pretrained_timm(
        backbone_type, image_size=image_size, device=device,
        thresholds=thresholds or DEFAULT_THRESHOLDS,
    )
    log.info(f"[BASELINE] setup {time.perf_counter()-t0_setup:.1f}s  "
             f"patch_grid={det.backbone.patch_grid}  num_channels={det.backbone.num_channels}")
    ref_pil = Image.open(ref_image).convert("RGB")
    reports: List[PerImageReport] = []
    for idx, src_p in enumerate(src_images):
        src_pil = Image.open(src_p).convert("RGB")
        t0 = time.perf_counter()
        res: DetectionResult = det.detect(src_pil, ref_pil, split_axis="horizontal")
        dt_ms = 1000.0 * (time.perf_counter() - t0)
        # English:English 0 English + English 1 English,English
        if idx in (0, len(src_images) - 1):
            save_to = out_dir / f"baseline_{backbone_type}_src{idx:02d}_artifacts.png"
            det.visualize(src_pil, res, save_path=save_to, dot_style=True)
            log.info(f"[BASELINE] vis saved -> {save_to}")
        reports.append(PerImageReport(
            ckpt=f"timm://{backbone_type}",
            src_image=str(src_p), ref_image=str(ref_image),
            backbone_type=backbone_type, image_size=image_size,
            patch_grid=list(res.patch_grid),
            total_tokens=int(res.stats.get("total", 0)),
            fp_cnt=int(res.stats["fp_cnt"]),   fp_rate=float(res.stats["fp_rate"]),
            gp_cnt=int(res.stats["gp_cnt"]),   gp_rate=float(res.stats["gp_rate"]),
            ah_cnt=int(res.stats["ah_cnt"]),   ah_rate=float(res.stats["ah_rate"]),
            union_cnt=int(res.stats["union_cnt"]), union_rate=float(res.stats["union_rate"]),
            ah_z_mean=float(res.stats.get("ah_z_mean", 0.0)),
            ah_z_std=float(res.stats.get("ah_z_std", 0.0)),
            latency_ms=float(dt_ms),
        ))
        log.info(
            f"[BASELINE][{idx+1:02d}/{len(src_images):02d}] "
            f"FP={reports[-1].fp_cnt}/{reports[-1].total_tokens} ({reports[-1].fp_rate*100:5.1f}%)   "
            f"GP={reports[-1].gp_cnt} ({reports[-1].gp_rate*100:5.1f}%)   "
            f"AH={reports[-1].ah_cnt} ({reports[-1].ah_rate*100:5.1f}%)   "
            f"UNION={reports[-1].union_cnt} ({reports[-1].union_rate*100:5.1f}%)   "
            f"dt={reports[-1].latency_ms:6.1f} ms"
        )
    csv_path = out_dir / f"baseline_{backbone_type}_report.csv"
    json_path = out_dir / f"baseline_{backbone_type}_summary.json"
    logger_out = logger or logging.getLogger("detec")
    return _aggregate_and_write_report(reports, csv_path, json_path, logger_out)


# ============================================================================
# English:English ACT .ckpt English
# ============================================================================

def run_act_ckpt_detection(
    ckpt_path: Path,
    backbone_type: str,
    src_images: List[Path],
    ref_image: Path,
    out_dir: Path,
    *,
    image_size: int = 224,
    device: str = "cpu",
    thresholds: Optional[ArtifactThresholds] = None,
    backbone_idx: int = 0,
    logger: Optional[logging.Logger] = None,
) -> PerImageReport:
    log = logger or logging.getLogger("detec")
    log.info(f"[ACT-CKPT] loading backbone from {ckpt_path} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0_setup = time.perf_counter()
    det = ArtifactDetector.from_act_checkpoint(
        str(ckpt_path), backbone_type, image_size=image_size, device=device,
        thresholds=thresholds or DEFAULT_THRESHOLDS,
        backbone_idx=backbone_idx,
    )
    log.info(f"[ACT-CKPT] setup {time.perf_counter()-t0_setup:.1f}s   "
             f"patch_grid={det.backbone.patch_grid}   num_channels={det.backbone.num_channels}")
    ref_pil = Image.open(ref_image).convert("RGB")
    reports: List[PerImageReport] = []
    for idx, src_p in enumerate(src_images):
        src_pil = Image.open(src_p).convert("RGB")
        t0 = time.perf_counter()
        res: DetectionResult = det.detect(src_pil, ref_pil, split_axis="horizontal")
        dt_ms = 1000.0 * (time.perf_counter() - t0)
        if idx in (0, len(src_images) - 1):
            save_to = out_dir / f"act_{ckpt_path.stem}_src{idx:02d}_artifacts.png"
            det.visualize(src_pil, res, save_path=save_to, dot_style=True)
            log.info(f"[ACT-CKPT] vis saved -> {save_to}")
        reports.append(PerImageReport(
            ckpt=str(ckpt_path),
            src_image=str(src_p), ref_image=str(ref_image),
            backbone_type=backbone_type, image_size=image_size,
            patch_grid=list(res.patch_grid),
            total_tokens=int(res.stats.get("total", 0)),
            fp_cnt=int(res.stats["fp_cnt"]),   fp_rate=float(res.stats["fp_rate"]),
            gp_cnt=int(res.stats["gp_cnt"]),   gp_rate=float(res.stats["gp_rate"]),
            ah_cnt=int(res.stats["ah_cnt"]),   ah_rate=float(res.stats["ah_rate"]),
            union_cnt=int(res.stats["union_cnt"]), union_rate=float(res.stats["union_rate"]),
            ah_z_mean=float(res.stats.get("ah_z_mean", 0.0)),
            ah_z_std=float(res.stats.get("ah_z_std", 0.0)),
            latency_ms=float(dt_ms),
        ))
        log.info(
            f"[ACT-CKPT][{idx+1:02d}/{len(src_images):02d}] "
            f"FP={reports[-1].fp_cnt}/{reports[-1].total_tokens} ({reports[-1].fp_rate*100:5.1f}%)   "
            f"GP={reports[-1].gp_cnt} ({reports[-1].gp_rate*100:5.1f}%)   "
            f"AH={reports[-1].ah_cnt} ({reports[-1].ah_rate*100:5.1f}%)   "
            f"UNION={reports[-1].union_cnt} ({reports[-1].union_rate*100:5.1f}%)   "
            f"dt={reports[-1].latency_ms:6.1f} ms"
        )
    csv_path = out_dir / f"act_{ckpt_path.stem}_report.csv"
    json_path = out_dir / f"act_{ckpt_path.stem}_summary.json"
    return _aggregate_and_write_report(reports, csv_path, json_path, log)


# ============================================================================
# CLI English
# ============================================================================

def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="ACT English(English train.sh dinov2 224 English)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt-dir", default=None,
                    help="ACT .ckpt English.English train.sh English " +
                         f"./act_ckpt/act-{DEFAULT_TASK_NAME}/{DEFAULT_TASK_CONFIG}-{DEFAULT_EXPERT_DATA_NUM}")
    ap.add_argument("--ckpt", default=None,
                    help="English ckpt English(English,English policy_epoch_500_seed_0.ckpt)."
                         "English:English policy_best.ckpt English,English policy_last.ckpt.")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE_TYPE, choices=["vit", "dinov2", "clip"])
    ap.add_argument("--image-size", type=int, default=DEFAULT_BACKBONE_IMAGE_SIZE)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--src-images", nargs="*", default=None,
                    help="English.English n-src English.")
    ap.add_argument("--n-src", type=int, default=3,
                    help="English(English --src-images English)")
    ap.add_argument("--ref-image", default=None,
                    help="English;English UniRefiner English fixed_reference.png")
    ap.add_argument("--compare-pretrained", action="store_true",
                    help="English timm English,English ACT ckpt English")
    ap.add_argument("--output-dir", default=None,
                    help="English.English:<ckpt_dir>/../artifact_reports/ English project artifact_outputs/")
    ap.add_argument("--thr-fp",    type=float, default=DEFAULT_THRESHOLDS.fp)
    ap.add_argument("--thr-gp",    type=float, default=DEFAULT_THRESHOLDS.gp)
    ap.add_argument("--thr-gpsep", type=float, default=DEFAULT_THRESHOLDS.gp_sep_min)
    ap.add_argument("--thr-ah",    type=float, default=DEFAULT_THRESHOLDS.ah_coeff,
                    help="AH z-score English(English:English)")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    # --- 1. English ckpt English --------------------------------------------------
    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
    else:
        ckpt_dir = _POLICY_ACT_DIR / DEFAULT_CKPT_DIR_RELATIVE
    # English best English last
    if args.ckpt:
        ckpt_path = ckpt_dir / args.ckpt
    else:
        best = ckpt_dir / "policy_best.ckpt"
        last = ckpt_dir / "policy_last.ckpt"
        ckpt_path = best if best.exists() else last
    # --- 2. English --------------------------------------------------------
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = _POLICY_ACT_DIR / "artifact_outputs" / (
            ckpt_dir.relative_to(_POLICY_ACT_DIR) if ckpt_dir.is_absolute() else
            Path(f"ckpt_{ckpt_dir.as_posix().replace('/', '_')}")
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(out_dir / "detec_run.log")

    if not ckpt_path.exists():
        logger.warning(f"[WARN] ckpt not found at {ckpt_path}  ——  "
                       "English timm English + English;English ACT English,English.")
        if not args.compare_pretrained:
            args.compare_pretrained = True

    # --- 3. English ------------------------------------------------------------
    thresholds = ArtifactThresholds(
        fp=args.thr_fp, gp=args.thr_gp,
        gp_sep_min=args.thr_gpsep, ah_coeff=args.thr_ah,
    )

    # --- 4. English / English ---------------------------------------------------
    ref_image = Path(args.ref_image) if args.ref_image else _ensure_reference_image()
    if args.src_images:
        src_images = [Path(p) for p in args.src_images]
        for p in src_images:
            if not p.exists():
                raise FileNotFoundError(p)
    else:
        src_images = _synthetic_source_image_batch(n=args.n_src)

    logger.info("=" * 78)
    logger.info("[CONFIG]")
    logger.info(f"  ckpt_dir        = {ckpt_dir}")
    logger.info(f"  ckpt_to_run     = {ckpt_path}  (exists={ckpt_path.exists()})")
    logger.info(f"  backbone_type   = {args.backbone}   image_size={args.image_size}")
    logger.info(f"  device          = {args.device}")
    logger.info(f"  thresholds      = fp≥{thresholds.fp:.2f}   gp≥{thresholds.gp:.2f}  "
                f"gp_sep<{thresholds.gp_sep_min:.2f}   ah_coeff={thresholds.ah_coeff:+.2f}")
    logger.info(f"  ref_image       = {ref_image}   (exists={ref_image.exists()})")
    logger.info(f"  src_images (#{len(src_images)}) = {[str(s) for s in src_images]}")
    logger.info(f"  output_dir      = {out_dir}")
    logger.info("=" * 78)

    # --- 5. English ------------------------------------------------------------
    if args.compare_pretrained:
        base_dir = out_dir / "baseline_timm"
        logger.info("")
        _ = run_pretrained_baseline(
            args.backbone, src_images, ref_image, base_dir,
            image_size=args.image_size, device=args.device,
            thresholds=thresholds, logger=logger,
        )
    if ckpt_path.exists():
        logger.info("")
        try:
            _ = run_act_ckpt_detection(
                ckpt_path, args.backbone, src_images, ref_image, out_dir,
                image_size=args.image_size, device=args.device,
                thresholds=thresholds, logger=logger,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("=" * 78)
            logger.error(f"[FAIL] ACT ckpt English / English: {type(exc).__name__}: {exc}")
            logger.error("=" * 78)
            logger.error(
                "English:\n"
                "  1. English ipython English ckpt English torch.load:\n"
                f"     >>> import torch; d = torch.load('{ckpt_path}', map_location='cpu');\n"
                "     >>> print(type(d), list(d.keys())[:10])\n"
                "  2. English(English DINOv2 224 ckpt ≈ 1.0~1.5GB):\n"
                f"     ls -lh '{ckpt_path}'\n"
                "  3. English save_freq English ckpt(English policy_epoch_2000 / policy_epoch_4000);\n"
                "     English policy_last.ckpt English policy_best.ckpt(English last).\n"
                "  4. backbone_type English --backbone_type(English {args.backbone}).".format(args=args)
            )
            logger.error("English ACT ckpt;English,English timm baseline English(English --compare-pretrained).")
            return 2
    logger.info("")
    logger.info(f"[DONE] all artifacts in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
