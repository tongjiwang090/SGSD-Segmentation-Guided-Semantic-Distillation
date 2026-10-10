"""
English UniRefiner English (CVPR 2026) English (Spurious Token) English.

English:
  1. FP  (Fixed Pattern)   English:tokenEnglish,English.
                            English:src token English English ref_sep English >= τ_fp.
  2. GP  (Global Proxy)    English:tokenEnglish.
                            English:(a) English(src‖ref)English src token English EnglishrefEnglish >= τ_gp;
                                  (b) English English Englishref_sepEnglish < τ_gp_sep_min (EnglishFP).
  3. AH  (Attention Hijackee) English:EnglishtokenEnglish
                            self-attentionEnglish,English"English".
                            English:English/Englishz-scoreEnglish.

English Eq.(1) / Eq.(2) English UniRefiner-main English
  unirefiner/spurious_filtering/{fp_gp,attention_hijack}.py English,
English ACT English timm-based English ViT/DINOv2/CLIP English.
"""

import io
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

# ---------------------------------------------------------------------------
# English
# ---------------------------------------------------------------------------

@dataclass
class ArtifactThresholds:
    """English.

    English UniRefiner English schema English:
      fp          : method.fp_gp_thres (English 0.5~0.7)
      gp          : method.fp_gp_thres (GP within-scene English)
      gp_sep_min  : GP Eq.(2) English(EnglishFPEnglish,EnglishfpEnglish)
      ah_coeff    : AH z-score English(English"Englishstd")
                    English attention_hijack_sigma English 0.25 / 0.5
    """
    fp: float = 0.7
    gp: float = 0.7
    gp_sep_min: float = 0.5
    ah_coeff: float = -0.25

    # -- English:English UniRefiner schema English -----------------------------------
    @classmethod
    def from_unirefiner_schema(cls, *, fp_gp_thres: float = 0.5,
                               attention_hijack_sigma: float = 0.25) -> "ArtifactThresholds":
        return cls(
            fp=fp_gp_thres,
            gp=fp_gp_thres,
            gp_sep_min=fp_gp_thres,
            ah_coeff=float(attention_hijack_sigma),
        )


# ---------------------------------------------------------------------------
# English
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """English.

    mask English(English):True English"English patch token English".
    """
    fp_mask: torch.Tensor              # [B, N]   bool, True=FP
    gp_mask: torch.Tensor              # [B, N]   bool, True=GP
    ah_mask: torch.Tensor              # [B, N]   bool, True=AH
    union_mask: torch.Tensor           # [B, N]   bool, True = FP | GP | AH
    patch_grid: Tuple[int, int]        # (grid_h, grid_w)
    stats: Dict[str, Any] = field(default_factory=dict)

    # -- English ----------------------------------------------------------------
    def summary(self) -> str:
        s = self.stats
        lines = [
            f"[Artifact Detection] total_tokens={s.get('total', -1)}",
            f"  FP : cnt={s.get('fp_cnt', -1):>5}  rate={s.get('fp_rate', -1.0):.3%}",
            f"  GP : cnt={s.get('gp_cnt', -1):>5}  rate={s.get('gp_rate', -1.0):.3%}",
            f"  AH : cnt={s.get('ah_cnt', -1):>5}  rate={s.get('ah_rate', -1.0):.3%}",
            f"  UNION: cnt={s.get('union_cnt', -1):>5}  rate={s.get('union_rate', -1.0):.3%}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_grid": list(self.patch_grid),
            "stats": dict(self.stats),
            "fp_mask": self.fp_mask.cpu().numpy().tolist(),
            "gp_mask": self.gp_mask.cpu().numpy().tolist(),
            "ah_mask": self.ah_mask.cpu().numpy().tolist(),
            "union_mask": self.union_mask.cpu().numpy().tolist(),
        }


# ============================================================================
# 1. FP / GP / AH English
# ============================================================================

class FPArtifactHead(nn.Module):
    """Fixed Pattern English —— English Eq.(1).

    English(cosineEnglish,English fp_gp.py::filter_by_fp_gp_similarity English cosine_threshold English):
      English src token, English ref_sep English token English.
      English >= threshold,English FP(EnglishtokenEnglishtokenEnglish,English).
    """

    def __init__(self, threshold: float = 0.7):
        super().__init__()
        self.register_buffer("threshold", torch.tensor(float(threshold)))

    def set_threshold(self, value: float) -> None:
        self.threshold.data = torch.tensor(float(value), device=self.threshold.device)

    @torch.no_grad()
    def forward(
        self,
        src_tokens: torch.Tensor,
        ref_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src_tokens: [B, Ns, D]   L2-normalized English raw English(English)
            ref_tokens: [B, Nr, D]   English(English)English patch tokens
        Returns:
            fp_mask:  [B, Ns]  bool, True = FP artifact
            score:    [B, Ns]  float, max_j cos(src_i, ref_j)
        """
        a = F.normalize(src_tokens.float(), dim=-1)
        b = F.normalize(ref_tokens.float(), dim=-1)
        if b.shape[0] == 1 and a.shape[0] > 1:
            b = b.expand(a.shape[0], -1, -1)
        sim = a @ b.transpose(1, 2)           # [B, Ns, Nr]
        score, _ = sim.max(dim=-1)             # [B, Ns]
        mask = score >= self.threshold.to(score.device)
        return mask, score


class GPArtifactHead(nn.Module):
    """Global Proxy English —— English Eq.(2).

    English GP:
      (a) within-scene proxy English:English src English token English English ref English token
          English  >=  gp_threshold
          (English:English -inf,English)
      (b) separable FP English:    English src token English English ref_sep English
                                  <  gp_sep_min
                                  (FP English,GP English)
    """

    def __init__(self, gp_threshold: float = 0.7, gp_sep_min: float = 0.5):
        super().__init__()
        self.register_buffer("gp_threshold", torch.tensor(float(gp_threshold)))
        self.register_buffer("gp_sep_min",   torch.tensor(float(gp_sep_min)))

    def set_thresholds(self, *, gp_threshold: Optional[float] = None,
                       gp_sep_min: Optional[float] = None) -> None:
        if gp_threshold is not None:
            self.gp_threshold.data = torch.tensor(float(gp_threshold),
                                                  device=self.gp_threshold.device)
        if gp_sep_min is not None:
            self.gp_sep_min.data = torch.tensor(float(gp_sep_min),
                                                device=self.gp_sep_min.device)

    @staticmethod
    def _split_src_ref_side(
        all_cat_tokens: torch.Tensor,
        cat_patch_grid: Tuple[int, int],
        split_axis: str,
    ) -> Tuple[torch.Tensor, int, int]:
        """English token [B, gh*gw, D] English split_axis English src English + ref English.

        Returns: (ref_side_tokens [B, n_ref, D], N_src, N_ref)
        """
        gh, gw = cat_patch_grid
        total = all_cat_tokens.shape[1]
        if gh * gw != total:
            raise ValueError(
                f"cat_patch_grid {cat_patch_grid} mismatches token count {total}"
            )
        B, _, D = all_cat_tokens.shape
        # 2D reshape -> English -> English
        img2d = all_cat_tokens.reshape(B, gh, gw, D)
        if split_axis == "horizontal":    # English => English
            if gw % 2 != 0:
                raise ValueError(f"gw={gw} must be even for horizontal split")
            src_side = img2d[:, :, : gw // 2, :].reshape(B, gh * (gw // 2), D)
            ref_side = img2d[:, :, gw // 2 :, :].reshape(B, gh * (gw // 2), D)
        elif split_axis == "vertical":    # English => English
            if gh % 2 != 0:
                raise ValueError(f"gh={gh} must be even for vertical split")
            src_side = img2d[:, : gh // 2, :, :].reshape(B, (gh // 2) * gw, D)
            ref_side = img2d[:, gh // 2 :, :, :].reshape(B, (gh // 2) * gw, D)
        else:
            raise ValueError(f"Unsupported split_axis={split_axis!r}")
        return src_side, ref_side, src_side.shape[1], ref_side.shape[1]

    @torch.no_grad()
    def forward(
        self,
        src_cat_tokens: torch.Tensor,
        all_cat_tokens: torch.Tensor,
        ref_sep_tokens: torch.Tensor,
        *,
        cat_patch_grid: Tuple[int, int],
        split_axis: str = "horizontal",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src_cat_tokens:  [B, Ns, D]  English src English patch tokens (English/English)
            all_cat_tokens:  [B, Ns+Nr, D]  English patch tokens (src ‖ ref)
            ref_sep_tokens:  [B, Nr_sep, D]  English patch tokens (EnglishFPEnglish)
            cat_patch_grid:  (gh, gw) English patch English
            split_axis:      "horizontal" / "vertical"
        Returns:
            gp_mask:  [B, Ns]  bool
            score:    [B, Ns]  float, English(a)English within-scene proxy English
        """
        src_side, ref_side, Ns, _Nr = self._split_src_ref_side(
            all_cat_tokens, cat_patch_grid, split_axis,
        )
        # English src_cat_tokens,English(English)
        if src_cat_tokens.shape[1] == Ns:
            src_norm = F.normalize(src_cat_tokens.float(), dim=-1)
        else:
            src_norm = F.normalize(src_side.float(), dim=-1)
        ref_norm = F.normalize(ref_side.float(), dim=-1)

        # ---- English (a): within-scene English proxy English --------------------------------
        proxy_sim = src_norm @ ref_norm.transpose(1, 2)    # [B, Ns, Nr_refside]
        score, _ = proxy_sim.max(dim=-1)                   # [B, Ns]
        cond_a = score >= self.gp_threshold.to(score.device)

        # ---- English (b): English(English FP) ----------------------------------------
        sep_norm = F.normalize(ref_sep_tokens.float(), dim=-1)
        if sep_norm.shape[0] == 1 and src_norm.shape[0] > 1:
            sep_norm = sep_norm.expand(src_norm.shape[0], -1, -1)
        cross_sim = src_norm @ sep_norm.transpose(1, 2)
        cross_max, _ = cross_sim.max(dim=-1)
        cond_b = cross_max < self.gp_sep_min.to(cross_max.device)

        mask = cond_a & cond_b
        return mask, score


class AHArtifactHead(nn.Module):
    """Attention Hijackee English.

    English(English unirefiner/spurious_filtering/attention_hijack.py
          ::filter_attention_hijackees English):

      Hijackee token English:English key/value English,
      English query English English token English(English"English",
      English).

      English:
        English attn [B, H, Q, K](English K English softmax):
          1. English H English -> English Q->K English [B, Q, K]
          2. inflow  = attn.mean(1).sum(1)   [B, K]    English query English K English
          3. outflow = attn.mean(1).sum(2)   [B, Q]    English query English(English≈1,
                                                         English)
        English,English hijackee_score = 1 - mean_layer( inflow )
          (English:ah_score = 1 - stack(flow_maps).mean(0).sum(1)
            English flow_maps English [Q,K] English softmax(QK^T),sum(1) English"English K English",
            English 1 - English"English",English hijackee.)

      English:English hijackee_score English z-score = (x - μ) / σ,
        ah_coeff > 0 English:score <= μ + ah_coeff * σ   => English AH (English filter_attention_hijackees English)
        ah_coeff < 0 English:score <= μ + ah_coeff * σ   => English"English"English
    """

    def __init__(self, ah_coeff: float = -0.25):
        super().__init__()
        self.register_buffer("ah_coeff", torch.tensor(float(ah_coeff)))

    def set_coeff(self, value: float) -> None:
        self.ah_coeff.data = torch.tensor(float(value), device=self.ah_coeff.device)

    @torch.no_grad()
    def forward(
        self,
        attention_maps: List[torch.Tensor],
        *,
        patch_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            attention_maps: English [B, H, S, S] English softmax English
                            S English = (1 cls_token?) + patch_count + (register?)
            patch_count:    English AH mask English patch English(English src English patch English)
        Returns:
            ah_mask: [B, patch_count]  bool
            z_score: [B, patch_count]  float,  hijackee_score English z-score
        """
        if not attention_maps:
            raise ValueError("AH detection requires at least one attention map.")
        # English K English patch_count English(English patch tokens,English cls/register English)
        layer_scores = []
        for attn in attention_maps:
            a = attn.float()                                  # [B, H, S, S]
            # English patch_count English K English(patchEnglishkeyEnglish)
            if a.shape[-1] >= patch_count:
                a_patch_k = a[..., -patch_count:]            # [B, H, S, pc]
            else:
                a_patch_k = a
            # Q English:English patch English(English patch_count English)
            if a_patch_k.shape[-2] >= patch_count:
                a_patch = a_patch_k[..., -patch_count:, :]   # [B, H, pc, pc]
            else:
                a_patch = a_patch_k
            mean_per_head = a_patch.mean(dim=1)               # [B, pc, pc]
            # inflow: English K = sum_Q A[Q,K]  (English patch token English key English)
            inflow = mean_per_head.sum(dim=-2)                # [B, pc]
            layer_scores.append(inflow)
        stacked = torch.stack(layer_scores, dim=0)            # [L, B, pc]
        mean_inflow = stacked.mean(dim=0)                     # [B, pc]

        # English hijackee_score:English ah_score = 1 - sum(flow_map) English
        # English flow_map English softmax(QK^T), sum(dim=1) English K English.
        # inflow English [0, English 1](EnglishQEnglish1,Q=pc => sum_K inflow≈pc)
        # English 1-sum English,English [0,1] English 1-x
        _min = mean_inflow.min(dim=-1, keepdim=True).values
        _max = mean_inflow.max(dim=-1, keepdim=True).values
        denom = (_max - _min).clamp(min=1e-9)
        inflow_norm = (mean_inflow - _min) / denom            # [B, pc] ∈ [0, 1]
        hijackee_score = 1.0 - inflow_norm                    # English/English AH

        # z-score English(English token English)
        mu = hijackee_score.mean(dim=-1, keepdim=True)
        sigma = hijackee_score.std(dim=-1, keepdim=True).clamp(min=1e-9)
        z = (hijackee_score - mu) / sigma
        threshold = mu + self.ah_coeff.to(z.device) * sigma
        # English:ah_coeff <= 0 English,"English"English AH;
        #       English score <= mean + sigma*coeff English clean(English coeff=0.25 English clean)
        # English:English AH English = hijackee_score English(English = English)
        # English T4 English(token 0 inflow English -> AH),English inflow_z English:
        inflow_z = -z  # English (inflow_norm - mu_in) / sigma_in
        ah_mask = inflow_z <= self.ah_coeff.to(inflow_z.device)
        return ah_mask, inflow_z


# ============================================================================
# 2. English:ArtifactDetectionHead
# ============================================================================

class ArtifactDetectionHead(nn.Module):
    """FP + GP + AH English."""

    def __init__(self, thresholds: Optional[ArtifactThresholds] = None):
        super().__init__()
        t = thresholds or ArtifactThresholds()
        self.fp_head = FPArtifactHead(threshold=t.fp)
        self.gp_head = GPArtifactHead(gp_threshold=t.gp, gp_sep_min=t.gp_sep_min)
        self.ah_head = AHArtifactHead(ah_coeff=t.ah_coeff)

    # ------------------------------------------------------------------
    def set_thresholds(self, thresholds: ArtifactThresholds) -> None:
        self.fp_head.set_threshold(thresholds.fp)
        self.gp_head.set_thresholds(gp_threshold=thresholds.gp,
                                    gp_sep_min=thresholds.gp_sep_min)
        self.ah_head.set_coeff(thresholds.ah_coeff)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        src_sep_tokens: torch.Tensor,
        ref_sep_tokens: torch.Tensor,
        src_cat_tokens: torch.Tensor,
        all_cat_tokens: torch.Tensor,
        attention_maps: List[torch.Tensor],
        *,
        cat_patch_grid: Optional[Tuple[int, int]] = None,
        split_axis: str = "horizontal",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            src_sep_tokens:  [B, N, D]  English patch tokens (English FP English)
            ref_sep_tokens:  [B, M, D]  English patch tokens (FP & GP English(b))
            src_cat_tokens:  [B, N, D]  English src English patch tokens
            all_cat_tokens:  [B, N+M, D] English patch tokens
            attention_maps:  English [B, H, S, S]
            cat_patch_grid:  English patch English;None English src_grid English/English 2
            split_axis:      "horizontal" / "vertical"
        Returns:
            fp_mask       [B, N]
            gp_mask       [B, N]
            ah_cat_full   [B, N+M]  (English patch English AH mask,English)
            ah_src        [B, N]    (English src English AH mask,English union)
            stats         dict (English & English)
        """
        device = src_sep_tokens.device
        B, N, _D = src_sep_tokens.shape
        # -------- English grid -----------------------------------------------------
        if cat_patch_grid is None:
            side = int(math.isqrt(N))
            if side * side != N:
                raise ValueError("Cannot infer cat_patch_grid, please pass explicitly.")
            if split_axis == "horizontal":
                cat_patch_grid = (side, 2 * side)
            else:
                cat_patch_grid = (2 * side, side)
        N_cat = all_cat_tokens.shape[1]

        # -------- English ------------------------------------------------------
        fp_mask, _fp_s = self.fp_head(src_sep_tokens, ref_sep_tokens)
        gp_mask, _gp_s = self.gp_head(
            src_cat_tokens, all_cat_tokens, ref_sep_tokens,
            cat_patch_grid=cat_patch_grid, split_axis=split_axis,
        )
        ah_cat_full, ah_z_full = self.ah_head(attention_maps, patch_count=N_cat)
        # English src English AH
        _src_side, _ref_side, Ns, Nr = GPArtifactHead._split_src_ref_side(
            all_cat_tokens, cat_patch_grid, split_axis,
        )
        if split_axis == "horizontal":
            gh, gw = cat_patch_grid
            src_idx_1d = (
                torch.arange(N_cat, device=device)
                .reshape(gh, gw)[:, : gw // 2]
                .reshape(-1)
            )
        else:
            gh, gw = cat_patch_grid
            src_idx_1d = (
                torch.arange(N_cat, device=device)
                .reshape(gh, gw)[: gh // 2, :]
                .reshape(-1)
            )
        ah_src = ah_cat_full.gather(1, src_idx_1d.unsqueeze(0).expand(B, -1))
        assert ah_src.shape[1] == N == Ns, f"{ah_src.shape[1]} vs {N} vs {Ns}"

        # -------- English ---------------------------------------------------------
        union = fp_mask | gp_mask | ah_src
        total = B * N
        stats = dict(
            total=int(total),
            fp_cnt=int(fp_mask.sum().item()),
            gp_cnt=int(gp_mask.sum().item()),
            ah_cnt=int(ah_src.sum().item()),
            union_cnt=int(union.sum().item()),
            fp_rate=float(fp_mask.sum().item()) / float(total),
            gp_rate=float(gp_mask.sum().item()) / float(total),
            ah_rate=float(ah_src.sum().item()) / float(total),
            union_rate=float(union.sum().item()) / float(total),
            ah_z_mean=float(ah_z_full.mean().item()),
            ah_z_std=float(ah_z_full.std().item()),
        )
        return fp_mask, gp_mask, ah_cat_full, ah_src, stats


# ============================================================================
# 3. Backbone English & English
# ============================================================================

# ACT English(ImageNet mean/std,English backbone.py/act_policy.py English)
ACT_IMAGE_MEAN = (0.485, 0.456, 0.406)
ACT_IMAGE_STD  = (0.229, 0.224, 0.225)


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float32 tensor [1, 3, H, W] ∈ [0,1],English ACT English."""
    if not isinstance(pil_img, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(pil_img)!r}")
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def get_act_image_transform(
    image_size: Union[int, Tuple[int, int]] = 224,
    *, mean: Tuple[float, float, float] = ACT_IMAGE_MEAN,
    std: Tuple[float, float, float] = ACT_IMAGE_STD,
) -> transforms.Compose:
    """English ACT English:Resize -> ToTensor -> Normalize."""
    sz = (int(image_size), int(image_size)) if isinstance(image_size, int) else tuple(image_size)
    return transforms.Compose([
        transforms.Resize(sz, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# English
# ---------------------------------------------------------------------------

def extract_backbone_state_dict_from_act_ckpt(
    ckpt: Union[str, os.PathLike, Dict[str, Any], "OrderedDict[str, Any]"],
    *,
    backbone_idx: int = 0,
) -> "OrderedDict[str, torch.Tensor]":
    """English ACT .ckpt English / state_dict English backbone.body English.

    ACT English backbone English DETRVAE.backbones (ModuleList[Joiner]) English,
    English:model.backbones.{backbone_idx}.0.body.xxx(0 English Joiner[0]=Backbone).
    """
    # ---- 1. English --------------------------------------------------------
    if isinstance(ckpt, (str, os.PathLike)):
        path = os.fspath(ckpt)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ACT checkpoint not found: {path}")
        sz = os.path.getsize(path)
        if sz == 0:
            raise ValueError(
                f"ACT checkpoint is EMPTY (0 bytes): {path}\n"
                "English:English ckpt English / English / English."
                "English ckpt English policy_best.ckpt / policy_last.ckpt."
            )
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 — English
            import textwrap
            tip = textwrap.dedent(f"""\
                English ACT checkpoint: {path} (size={sz} bytes).
                English:{exc!r}.
                English:(1) English/English;(2) Lightning / torch English;
                (3) English ckpt English ACT imitate_episodes.py English.
                English:English(English DINOv2 ACT ckpt ≈ 1~1.5 GB);
                English `python -c "import torch; d=torch.load({path!r}, map_location='cpu'); "
                "print(list(d.keys())[:5], 'state_dict keys:', list(d.get('state_dict',{{}}).keys())[:5])"` English.
            """).strip()
            raise RuntimeError(tip) from exc
    else:
        loaded = ckpt
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, (dict, OrderedDict)):
        raise TypeError(f"Cannot extract state_dict from {type(loaded)!r}")

    # ---- 2. English ----------------------------------------------------
    prefix_candidates = [
        f"model.backbones.{backbone_idx}.0.body.",
        f"backbones.{backbone_idx}.0.body.",
        f"module.model.backbones.{backbone_idx}.0.body.",
    ]
    picked: Optional[str] = None
    for cand in prefix_candidates:
        if any(k.startswith(cand) for k in loaded.keys()):
            picked = cand
            break
    if picked is None:
        # English:English state_dict English body English,English
        maybe_body_keys = {
            k for k in loaded.keys()
            if k.startswith(("cls_token", "patch_embed.", "blocks.", "norm."))
        }
        if maybe_body_keys:
            return OrderedDict((k, loaded[k]) for k in sorted(loaded.keys()))
        raise KeyError(
            "Cannot locate backbone.body keys in checkpoint. "
            f"Expected prefix among {prefix_candidates}; "
            f"got sample keys: {list(loaded.keys())[:5]}"
        )
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    plen = len(picked)
    for k, v in loaded.items():
        if k.startswith(picked):
            out[k[plen:]] = v.detach().cpu()
    if not out:
        raise KeyError(f"Prefix {picked!r} matched no keys")
    return out


def infer_body_native_image_size(
    body_sd: "OrderedDict[str, torch.Tensor]",
    *,
    fallback_patch_size: Optional[int] = None,
) -> Tuple[int, int]:
    """English ACT ckpt English body state_dict English"English timm English img_size".

    English:
    * ckpt English pos_embed English N(English cls/reg)  = num_prefix_tokens + (H/p)*(W/p)
    * patch_embed.proj.weight English / kernel_size = patch_size
    * H = W = patch_size * sqrt(N - num_prefix_tokens)

    English:
    ACT TimmVisionBackbone (backbone.py line 266) English img_size,
    English timm English size English pos_embed(English dinov2.vit_base 518×518
    → pos_embed.shape = [1, 37²+1, 768] = [1, 1370, 768]).English 224,
    English dynamic_img_size=True English pos_embed,English ckpt English pos_embed *English*English.
    English 224 English `timm.create_model` English [1, 16²+1, 768]=[1, 257, 768],
    English ckpt [1, 1370, 768] English shape mismatch.
    """
    import math as _math
    pe = body_sd.get("pos_embed", None)
    if pe is None:
        if fallback_patch_size:
            return (int(fallback_patch_size * 14), int(fallback_patch_size * 14))
        raise KeyError("body_sd English pos_embed,English img_size")
    n_tok = int(pe.shape[1])  # [1, N, C]

    # English patch_size
    proj = body_sd.get("patch_embed.proj.weight", None)
    if proj is not None and proj.ndim == 4:
        ps = int(proj.shape[-1])
    elif fallback_patch_size is not None:
        ps = int(fallback_patch_size)
    else:
        # English backbone_type English(dinov2=14 / vit+clip=16)
        ps = 14 if (n_tok - 1) % 196 == 0 and (n_tok - 1) % 256 != 0 else 16

    num_prefix = 1  # dinov2/vit_base/clip_base English 1 English cls
    if body_sd.get("reg_token", None) is not None:
        num_prefix += int(body_sd["reg_token"].shape[1])
    n_patch = n_tok - num_prefix
    side = int(round(_math.sqrt(float(n_patch))))
    if side * side != n_patch:
        raise ValueError(
            f"English grid:pos_embed={list(pe.shape)}, num_prefix={num_prefix}, "
            f"n_patch={n_patch}(English).English fallback_patch_size."
        )
    return (side * ps, side * ps)


@dataclass
class LoadedBackbone:
    """English."""
    backbone_type: str
    image_size: int
    patch_size: int
    patch_grid: Tuple[int, int]
    num_channels: int
    timm_backbone: nn.Module          # timm English body (English cls_token, blocks, norm)
    image_transform: transforms.Compose
    device: torch.device

    # ------------------------------------------------------------------
    def feature_map(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        """English ACT TimmVisionBackbone.forward English:English {"0": [B,C,gh,gw]}."""
        self.timm_backbone.eval()
        with torch.no_grad():
            feats = self.timm_backbone.forward_features(x.to(self.device))
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if feats.ndim == 4:
            fmap = feats
        else:
            num_prefix = int(getattr(self.timm_backbone, "num_prefix_tokens", 0)
                              or (1 if hasattr(self.timm_backbone, "cls_token") else 0))
            patch = feats[:, num_prefix:, :]
            gh, gw = self.patch_grid
            fmap = patch.transpose(1, 2).reshape(feats.shape[0], -1, gh, gw)
        return OrderedDict({"0": fmap})

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_patch_tokens(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Tuple[int, int]]:
        """English patch_tokens + English + patch_grid.

        Returns:
            tokens:     [B, N, C]   L2-normalized patch tokens
            attentions: list of [B, H, S, S] English attention weight (English)
            grid:       (gh, gw)
        """
        bb = self.timm_backbone
        bb.eval()
        x = x.to(self.device)

        # ---- 1. English attn.forward English(English timm English) --------
        caught_attns: List[torch.Tensor] = []
        hook_handles = []
        try:
            blocks = getattr(bb, "blocks", None)
            if blocks is None or len(blocks) == 0:
                raise RuntimeError(f"{bb.__class__.__name__} has no .blocks; "
                                   "cannot capture attention.")

            def _capture_attn(module, _inp, out):
                # timm Attention.forward English (attended, attn_weights) tuple English
                # English tensor;English out English,English forward hook English,
                # English get_attention English.English out English tuple English.
                # English:English(cuda/cpu)—— English AH head English src_idx_1d gather English.
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    w = out[1]
                    if isinstance(w, torch.Tensor) and w.ndim == 4:
                        caught_attns.append(w.detach())
                return None

            for blk in blocks:
                if hasattr(blk, "attn"):
                    h = blk.attn.register_forward_hook(_capture_attn)
                    hook_handles.append(h)
            # English
            feats = bb.forward_features(x)
        finally:
            for h in hook_handles:
                h.remove()

        # ---- 2. English hook English,English fallback:timm get_intermediate_layers --------
        if not caught_attns:
            try:
                # timm English get_intermediate_layers(..., return_literal=True / n=len(blocks))
                if hasattr(bb, "get_intermediate_layers") \
                        and callable(bb.get_intermediate_layers):
                    _out = bb.get_intermediate_layers(
                        x,
                        n=len(blocks),
                        reshape=False,
                        return_class_token=False,
                        norm=False,
                    )
                    # English:English timm English attn;
                    # English forward English block.attn English "return_attn=True"
                    caught_attns = self._attentions_via_forward_attn(x)
            except Exception:
                caught_attns = self._attentions_via_forward_attn(x)

        # ---- 3. English feats English patch tokens ------------------------------------
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        num_prefix = int(
            getattr(bb, "num_prefix_tokens", None)
            or (1 if hasattr(bb, "cls_token") else 0)
        )
        if feats.ndim == 3 and feats.shape[1] > num_prefix:
            patch_tokens = feats[:, num_prefix:, :]
        else:
            gh, gw = self.patch_grid
            patch_tokens = feats.reshape(feats.shape[0], gh * gw, -1)
        patch_tokens = F.normalize(patch_tokens.float(), dim=-1)
        return patch_tokens, caught_attns, self.patch_grid

    # ------------------------------------------------------------------
    def _attentions_via_forward_attn(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Fallback:English block.attn English forward,English return_attn=True (timm English)."""
        import math as _math
        bb = self.timm_backbone
        bb.eval()
        blocks = bb.blocks
        x0 = x.to(self.device)
        # English patch_embed + pos_embed + prefix
        if hasattr(bb, "patch_embed"):
            tok = bb.patch_embed(x0)  # English:
                                      #   - [B, N_patch, D]         (English vit, 3D)
                                      #   - [B, D, gh, gw]          (Conv2d channels-first)
                                      #   - [B, gh, gw, D]          (DINOv2 default channels-last)
            if tok.ndim == 4:
                # English(English embed_dim=768/1024 English)
                _, d1, d2, d3 = tok.shape
                num_features_d = int(getattr(bb, "num_features", 0) or d3 or 768)
                if abs(d3 - num_features_d) <= max(8, num_features_d // 100):
                    # [B, gh, gw, D]  channels-last
                    tok = tok.flatten(1, 2)  # [B, gh*gw, D]
                elif abs(d1 - num_features_d) <= max(8, num_features_d // 100):
                    # [B, D, gh, gw]  channels-first
                    tok = tok.flatten(2).transpose(1, 2)  # [B, gh*gw, D]
                else:
                    # Fallback: English batch English gh,gw,English D
                    dims = sorted([(d1,1),(d2,2),(d3,3)], key=lambda x: x[0])
                    h_idx, w_idx, c_idx = dims[0][1], dims[1][1], dims[2][1]
                    tok = tok.permute(0, c_idx, h_idx, w_idx).flatten(2).transpose(1, 2)
            elif tok.ndim != 3:
                raise ValueError(f"patch_embed output ndim={tok.ndim} unsupported")
            B, N_patch, D = tok.shape
            # English prefix(cls_token + reg_token)
            prefix_toks = []
            cls = getattr(bb, "cls_token", None)
            if cls is not None:
                prefix_toks.append(cls.expand(B, -1, -1))
            reg = getattr(bb, "reg_token", None)
            if reg is not None:
                prefix_toks.append(reg.expand(B, -1, -1))
            if prefix_toks:
                tok = torch.cat(prefix_toks + [tok], dim=1)  # [B, N_pre+N_patch, D]
            # --- English pos_embed English(ckpt native size != English size English) ---
            pe = getattr(bb, "pos_embed", None)
            if pe is not None and pe.shape[1] == tok.shape[1]:
                tok = tok + pe
            elif pe is not None:
                # ckpt English pos_embed English tok English → English prefix English,
                # patch English reshape English 2D English F.interpolate English gh/gw
                # English:src+ref English(cat)English N_patch English,English x0 English hw English.
                num_prefix = tok.shape[1] - N_patch
                pe_prefix = pe[:, :num_prefix, :]  # [1, num_prefix, D]
                pe_patch_src = pe[:, num_prefix:, :]  # [1, N_src, D]
                src_side = int(round(_math.sqrt(float(pe_patch_src.shape[1]))))
                # English x0 English hw English gh/gw(English)
                in_h, in_w = int(x0.shape[-2]), int(x0.shape[-1])
                ps: Tuple[int, int]
                ps = getattr(bb.patch_embed, "patch_size", (self.patch_size, self.patch_size))
                if not isinstance(ps, (tuple, list)):
                    ps = (int(ps), int(ps))
                gh = in_h // int(ps[0])
                gw = in_w // int(ps[1])
                if gh * gw != N_patch:
                    # English patch_embed English,English:English N_patch English factorization English
                    side1 = int(round(_math.sqrt(float(N_patch))))
                    while side1 > 1 and N_patch % side1 != 0:
                        side1 -= 1
                    gh, gw = side1, N_patch // side1
                C = pe_patch_src.shape[-1]
                pe_img = pe_patch_src.reshape(1, src_side, src_side, C).permute(0, 3, 1, 2)
                pe_img = F.interpolate(pe_img, size=(gh, gw), mode="bicubic", align_corners=False)
                pe_patch_dst = pe_img.permute(0, 2, 3, 1).reshape(1, gh * gw, C)
                pe_dst = torch.cat([pe_prefix, pe_patch_dst], dim=1)  # [1, N_pre+N_patch, D]
                tok = tok + pe_dst.to(tok.dtype)
            # English dropout / patch_drop English(eval)
            if hasattr(bb, "pos_drop"):
                tok = bb.pos_drop(tok)
        else:
            raise RuntimeError("Unsupported backbone: no patch_embed found.")

        attns: List[torch.Tensor] = []
        with torch.no_grad():
            for blk in blocks:
                # English norm1
                if hasattr(blk, "norm1"):
                    y = blk.norm1(tok)
                else:
                    y = tok
                attn_mod = blk.attn
                # English return_attn=True English
                try:
                    attn_out, w = attn_mod(y, return_attn=True)
                except TypeError:
                    # English qkv + softmax English
                    qkv = attn_mod.qkv(y) if hasattr(attn_mod, "qkv") else None
                    if qkv is None:
                        raise
                    B, N, C = qkv.shape
                    H = attn_mod.num_heads
                    head_dim = C // 3 // H
                    q, k, v = qkv.reshape(B, N, 3, H, head_dim) \
                                 .permute(2, 0, 3, 1, 4).unbind(0)  # 3 x [B,H,N,d]
                    scale = head_dim ** -0.5
                    attn_w = (q @ k.transpose(-2, -1)) * scale
                    attn_w = attn_w.softmax(dim=-1)
                    w = attn_w
                    attn_out = (attn_w @ v).transpose(1, 2).reshape(B, N, H * head_dim)
                    if hasattr(attn_mod, "proj"):
                        attn_out = attn_mod.proj(attn_out)
                if isinstance(w, torch.Tensor) and w.ndim == 4:
                    attns.append(w.detach())
                # drop path + residual
                if hasattr(blk, "drop_path1"):
                    attn_out = blk.drop_path1(attn_out)
                tok = tok + attn_out
                if hasattr(blk, "norm2") and hasattr(blk, "mlp"):
                    mlp_out = blk.mlp(blk.norm2(tok))
                    if hasattr(blk, "drop_path2"):
                        mlp_out = blk.drop_path2(mlp_out)
                    tok = tok + mlp_out
        return attns


class BackboneLoader:
    """English:English timm English & ACT .ckpt English."""

    # timm English -> English backbone_type
    _TYPE_TO_TIMM_NAME = {
        "vit":   "vit_base_patch16_224",
        "dinov2": "vit_base_patch14_dinov2.lvd142m",
        "clip":  "vit_base_patch16_clip_224.openai",
    }
    _SUPPORTED = {"vit", "dinov2", "clip"}

    # ------------------------------------------------------------------
    @classmethod
    def _resolve_timm_name(cls, backbone_type: str, name_override: Optional[str]) -> str:
        if backbone_type not in cls._SUPPORTED:
            raise ValueError(
                f"Unsupported backbone_type={backbone_type!r}. "
                f"Supported: {sorted(cls._SUPPORTED)}"
            )
        return name_override or cls._TYPE_TO_TIMM_NAME[backbone_type]

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained_timm(
        cls,
        backbone_type: str,
        *,
        image_size: int = 224,
        device: Union[str, torch.device] = "cpu",
        pretrained: bool = True,
        timm_model_name: Optional[str] = None,
    ) -> LoadedBackbone:
        try:
            import timm
        except ImportError as e:
            raise RuntimeError("timm is required to build visual backbones.") from e
        name = cls._resolve_timm_name(backbone_type, timm_model_name)
        # English attention(clip/openclip English timm English attention only English)
        model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            img_size=int(image_size),
            dynamic_img_size=True,
        )
        model.eval()
        device_t = torch.device(device)
        model.to(device_t)

        # English patch_size
        pe = getattr(model, "patch_embed", None)
        if pe is None:
            raise RuntimeError(f"timm model {name!r} has no patch_embed")
        ps = getattr(pe, "patch_size", 16)
        patch_size = int(ps[0]) if isinstance(ps, (tuple, list)) else int(ps)
        gh = int(math.ceil(image_size / patch_size))
        gw = gh
        grid = (gh, gw)
        num_channels = int(getattr(model, "num_features", 0))
        if num_channels <= 0:
            raise RuntimeError(f"Cannot read num_features from timm model {name}")
        tfm = get_act_image_transform(image_size)
        return LoadedBackbone(
            backbone_type=backbone_type,
            image_size=int(image_size),
            patch_size=patch_size,
            patch_grid=grid,
            num_channels=num_channels,
            timm_backbone=model,
            image_transform=tfm,
            device=device_t,
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_act_checkpoint(
        cls,
        ckpt_path: Union[str, os.PathLike],
        backbone_type: str,
        *,
        image_size: int = 224,
        device: Union[str, torch.device] = "cpu",
        backbone_idx: int = 0,
    ) -> LoadedBackbone:
        """English ACT policy_last.ckpt / policy_epoch_*_seed_*.ckpt English backbone.body English.

        English:
          English ACT English TimmVisionBackbone (backbone.py line 266) English
          timm.create_model English img_size,ckpt English pos_embed English"English size"
          English(English dinov2_base English 518×518,English pos_embed.shape=[1, 1370, 768]),
          English image_size=224 English.English:
            1. English body_sd
            2. English `infer_body_native_image_size` English img_size_native
            3. English img_size_native English timm(strict English ckpt)
            4. forward English patch English(English dynamic_img_size=True)
        """
        try:
            import timm
        except ImportError as e:
            raise RuntimeError("timm is required to build visual backbones.") from e
        # (1) English body_sd + English native size
        body_sd = extract_backbone_state_dict_from_act_ckpt(
            ckpt_path, backbone_idx=backbone_idx,
        )
        native_h, native_w = infer_body_native_image_size(body_sd)
        assert native_h == native_w, f"English ViT English:native_hw=({native_h},{native_w})"
        native_img_size = int(native_h)
        name = cls._resolve_timm_name(backbone_type, None)
        # (2) English native size English(English),English dynamic_img_size English forward English
        model = timm.create_model(
            name,
            pretrained=False,
            num_classes=0,
            img_size=native_img_size,
            dynamic_img_size=True,
        )
        missing, unexpected = model.load_state_dict(body_sd, strict=False)
        for m in missing:
            if not any(part in m for part in ("cls_token", "pos_embed", "reg_token")):
                raise RuntimeError(
                    f"Missing non-optional key loading ACT ckpt: {m}\n"
                    f"  native_img_size English {native_img_size}×{native_img_size}(English pos_embed={list(body_sd.get('pos_embed',()).shape)})"
                    f"  timm_name={name}; English backbone_type English."
                )
        model.eval()
        device_t = torch.device(device)
        model.to(device_t)
        # (3) English patch_size & English.English patch_grid / image_size English"English image_size"
        #     English(English ACT English feed English),forward English resize.
        pe = getattr(model, "patch_embed", None)
        if pe is None:
            raise RuntimeError(f"timm model {name!r} has no patch_embed")
        ps = getattr(pe, "patch_size", 16)
        patch_size = int(ps[0]) if isinstance(ps, (tuple, list)) else int(ps)
        gh = int(math.ceil(image_size / patch_size))
        gw = gh
        num_channels = int(getattr(model, "num_features", 0))
        if num_channels <= 0:
            raise RuntimeError(f"Cannot read num_features from timm model {name}")
        tfm = get_act_image_transform(image_size)
        bb = LoadedBackbone(
            backbone_type=backbone_type,
            image_size=int(image_size),
            patch_size=patch_size,
            patch_grid=(gh, gw),
            num_channels=num_channels,
            timm_backbone=model,
            image_transform=tfm,
            device=device_t,
        )
        # English"English size"English,debug/log English
        bb._ckpt_native_hw = (native_h, native_w)  # type: ignore[attr-defined]
        return bb


# ============================================================================
# 4. ArtifactDetector English
# ============================================================================

class ArtifactDetector:
    """English:English PIL English -> English DetectionResult."""

    def __init__(
        self,
        loaded_backbone: LoadedBackbone,
        *,
        thresholds: Optional[ArtifactThresholds] = None,
    ):
        self.backbone = loaded_backbone
        self.head = ArtifactDetectionHead(thresholds)
        self.head.eval()

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained_timm(
        cls,
        backbone_type: str,
        *,
        image_size: int = 224,
        device: Union[str, torch.device] = "cpu",
        thresholds: Optional[ArtifactThresholds] = None,
    ) -> "ArtifactDetector":
        bb = BackboneLoader.from_pretrained_timm(
            backbone_type, image_size=image_size, device=device,
        )
        return cls(bb, thresholds=thresholds)

    @classmethod
    def from_act_checkpoint(
        cls,
        ckpt_path: Union[str, os.PathLike],
        backbone_type: str,
        *,
        image_size: int = 224,
        device: Union[str, torch.device] = "cpu",
        thresholds: Optional[ArtifactThresholds] = None,
        backbone_idx: int = 0,
    ) -> "ArtifactDetector":
        bb = BackboneLoader.from_act_checkpoint(
            ckpt_path, backbone_type, image_size=image_size, device=device,
            backbone_idx=backbone_idx,
        )
        return cls(bb, thresholds=thresholds)

    # ------------------------------------------------------------------
    def set_thresholds(self, thresholds: ArtifactThresholds) -> None:
        self.head.set_thresholds(thresholds)

    # ------------------------------------------------------------------
    def _compose_images(
        self, src_pil: Image.Image, ref_pil: Image.Image, split_axis: str,
    ) -> Image.Image:
        sz = (self.backbone.image_size, self.backbone.image_size)
        s = src_pil.convert("RGB").resize(sz, Image.BICUBIC)
        r = ref_pil.convert("RGB").resize(sz, Image.BICUBIC)
        if split_axis == "horizontal":
            out = Image.new("RGB", (2 * sz[0], sz[1]))
            out.paste(s, (0, 0))
            out.paste(r, (sz[0], 0))
        elif split_axis == "vertical":
            out = Image.new("RGB", (sz[0], 2 * sz[1]))
            out.paste(s, (0, 0))
            out.paste(r, (0, sz[1]))
        else:
            raise ValueError(f"split_axis={split_axis!r}")
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def detect(
        self,
        src_image: Image.Image,
        ref_image: Image.Image,
        *,
        split_axis: str = "horizontal",
    ) -> DetectionResult:
        """
        English.English:
          src_sep, ref_sep English backbone English FP / GP(b) English;
          src‖ref English backbone English GP(a) & AH English.
        """
        tfm = self.backbone.image_transform
        sz = self.backbone.image_size
        # ---- English -----------------------------------------------------------
        src_t = tfm(src_image.convert("RGB")).unsqueeze(0).to(self.backbone.device)
        ref_t = tfm(ref_image.convert("RGB")).unsqueeze(0).to(self.backbone.device)
        src_sep, _, _ = self.backbone.extract_patch_tokens(src_t)
        ref_sep, _, _ = self.backbone.extract_patch_tokens(ref_t)

        # ---- English -------------------------------------------------------------
        cat_pil = self._compose_images(src_image, ref_image, split_axis)
        if split_axis == "horizontal":
            cat_sz = (2 * sz, sz)
            gh, gw_src = self.backbone.patch_grid
            cat_grid = (gh, 2 * gw_src)
        else:
            cat_sz = (sz, 2 * sz)
            gh_src, gw = self.backbone.patch_grid
            cat_grid = (2 * gh_src, gw)
        cat_tfm = get_act_image_transform(cat_sz)
        cat_t = cat_tfm(cat_pil).unsqueeze(0).to(self.backbone.device)
        all_cat, cat_attns, _ = self.backbone.extract_patch_tokens(cat_t)

        # English src English token
        _src_side, _ref_side, Ns, _Nr = GPArtifactHead._split_src_ref_side(
            all_cat, cat_grid, split_axis,
        )
        src_cat = _src_side
        # ---- English -------------------------------------------------------------
        fp, gp, _ah_full, ah_src, stats = self.head(
            src_sep_tokens=src_sep,
            ref_sep_tokens=ref_sep,
            src_cat_tokens=src_cat,
            all_cat_tokens=all_cat,
            attention_maps=cat_attns,
            cat_patch_grid=cat_grid,
            split_axis=split_axis,
        )
        union = fp | gp | ah_src
        return DetectionResult(
            fp_mask=fp.cpu(),
            gp_mask=gp.cpu(),
            ah_mask=ah_src.cpu(),
            union_mask=union.cpu(),
            patch_grid=self.backbone.patch_grid,
            stats=stats,
        )

    # ------------------------------------------------------------------
    def visualize(
        self,
        src_image: Image.Image,
        result: DetectionResult,
        *,
        save_path: Optional[Union[str, os.PathLike]] = None,
        dot_style: bool = True,
        show: bool = False,
        alpha_overlay: float = 0.35,
    ) -> Image.Image:
        """English src_image English(English,matplotlib Agg English)."""
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        gh, gw = result.patch_grid
        sz = self.backbone.image_size
        img = src_image.convert("RGB").resize((sz, sz), Image.BICUBIC)
        arr = np.asarray(img)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=100)
        labels = [("FP",    result.fp_mask,    "#d62728"),
                  ("GP",    result.gp_mask,    "#ff7f0e"),
                  ("AH",    result.ah_mask,    "#9467bd"),
                  ("UNION", result.union_mask, "#1f77b4")]
        for ax, (title, mask, color) in zip(axes, labels):
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.imshow(arr)
            # English mask English 2D English
            m2d = mask[0].reshape(gh, gw).numpy().astype(np.float32)
            m_up = np.array(Image.fromarray(m2d).resize((sz, sz), Image.NEAREST))
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                f"cmap_{title}", ["#00000000", color], N=256,
            )
            ax.imshow(m_up, cmap=cmap, vmin=0.0, vmax=1.0, alpha=alpha_overlay)
            if dot_style:
                # English patch English
                idxs_y, idxs_x = np.where(mask[0].reshape(gh, gw).numpy())
                ph = sz / gh
                pw = sz / gw
                for iy, ix in zip(idxs_y, idxs_x):
                    cx = (ix + 0.5) * pw
                    cy = (iy + 0.5) * ph
                    ax.add_patch(Circle((cx, cy), radius=min(ph, pw) * 0.22,
                                        fill=True, facecolor=color,
                                        edgecolor="white", linewidth=0.8))
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buf.seek(0)
        out_img = Image.open(buf).convert("RGB")
        if save_path is not None:
            os.makedirs(os.path.dirname(os.fspath(save_path)) or ".", exist_ok=True)
            out_img.save(save_path)
        if show:
            try:
                out_img.show()
            except Exception:
                pass
        return out_img


# ============================================================================
# 5. English:English ACT English
# ============================================================================

def run_artifact_detection_on_act_ckpt(
    ckpt_path: Union[str, os.PathLike],
    backbone_type: str,
    src_image: Union[str, os.PathLike, Image.Image],
    ref_image: Union[str, os.PathLike, Image.Image],
    *,
    image_size: int = 224,
    device: Union[str, torch.device] = "cpu",
    thresholds: Optional[ArtifactThresholds] = None,
    save_plot_to: Optional[Union[str, os.PathLike]] = None,
    save_stats_csv: Optional[Union[str, os.PathLike]] = None,
) -> DetectionResult:
    """English ACT English .ckpt English."""
    # ---- English ------------------------------------------------------------
    def _load(p):
        if isinstance(p, Image.Image):
            return p
        return Image.open(p).convert("RGB")
    src_pil = _load(src_image)
    ref_pil = _load(ref_image)

    det = ArtifactDetector.from_act_checkpoint(
        ckpt_path, backbone_type, image_size=image_size, device=device,
        thresholds=thresholds,
    )
    res = det.detect(src_pil, ref_pil)
    if save_plot_to is not None:
        det.visualize(src_pil, res, save_path=save_plot_to, dot_style=True)
    if save_stats_csv is not None:
        import csv
        os.makedirs(os.path.dirname(os.fspath(save_stats_csv)) or ".", exist_ok=True)
        with open(save_stats_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in res.stats.items():
                w.writerow([k, v])
    return res


# ============================================================================
# __all__ English(English import English)
# ============================================================================

__all__ = [
    "AHArtifactHead",
    "ACT_IMAGE_MEAN",
    "ACT_IMAGE_STD",
    "ArtifactDetectionHead",
    "ArtifactDetector",
    "ArtifactThresholds",
    "BackboneLoader",
    "DetectionResult",
    "FPArtifactHead",
    "GPArtifactHead",
    "LoadedBackbone",
    "extract_backbone_state_dict_from_act_ckpt",
    "get_act_image_transform",
    "pil_to_tensor",
    "run_artifact_detection_on_act_ckpt",
]
