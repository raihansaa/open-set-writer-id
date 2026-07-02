#!/usr/bin/env python3
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))


# INPUT_MODE is propagated so spawned DataLoader workers inherit on re-import.
_INPUT_MODE = _os.environ.get('INPUT_MODE', '')
_IS_SKELETON_MODE = _INPUT_MODE == 'skeleton_dt'
_IS_PLAIN_SKELETON_MODE = _INPUT_MODE == 'plain_skeleton'



# ======================================================================
# 1. Setup & Config
# ======================================================================

import argparse
import os
import time
import math
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

@dataclass
class Config:
    # -- Paths --------------------------------------------------------
    data_dir: Path = Path(os.environ.get("CIRCLEID_DATA", Path(__file__).resolve().parent.parent / "icdar-2026-circleid-writer-identification"))
    train_csv: str = "train.csv"
    additional_train_csv: str = "additional_train.csv"
    test_csv: str = "test.csv"
    output_dir: Path = Path("./outputs")
    checkpoint_dir: Path = Path("./checkpoints")

    # -- DINOv2 --------------------------------------------------------
    dinov2_repo: str = "facebookresearch/dinov2"  
    dinov2_weights: str = ""  

    # -- Dataset ------------------------------------------------------
    num_known_writers: int = 44
    num_unknown_writers: int = 7
    num_pens: int = 8
    image_size: int = 224
    image_load_mode: str = "cv2"

    # -- Splits -------------------------------------------------------
    num_splits: int = 3
    held_out_writers_per_split: int = 7
    train_val_ratio: float = 0.8
    split_seed: int = 42

    # -- Backbone -----------------------------------------------------
    dinov2_variant: str = "dinov2_vitb14_reg"
    dinov2_embed_dim: int = 768
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_last_n_blocks: int = 6  

    # -- Architecture -------------------------------------------------
    projection_dim: int = 512
    writer_embed_dim: int = 512

    # -- ArcFace ------------------------------------------------------
    arcface_scale: float = 30.0
    arcface_margin: float = 0.3
    arcface_margin_warmup_start: int = 0
    arcface_margin_warmup_end: int = 15

    # -- Training -----------------------------------------------------
    epochs: int = 40
    batch_size: int = 48
    grad_accum_steps: int = 2
    lr_lora: float = 5e-5
    lr_heads: float = 5e-4
    lr_unfreeze: float = 1e-5
    unfreeze_norm_bias_blocks: int = 4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    # -- EMA ----------------------------------------------------------
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_start_epoch: int = 5

    # -- Early Stopping ------------------------------------------------
    early_stop_patience: int = 10
    early_stop_val_fraction: float = 0.1
    eval_every_n_epochs: int = 3  

    # -- OOD ----------------------------------------------------------
    ood_alpha: float = 0.7
    ood_train_lambda: float = 0.5  # outlier exposure weight
    ood_train_warmup: int = 5  

    # -- Augmentation -------------------------------------------------
    aug_hflip: bool = True
    aug_rotation_deg: float = 15.0

    # -- TTA ----------------------------------------------------------
    tta_variants: int = 3
    tta_hflip: bool = True
    tta_brightness_range: float = 0.15
    tta_contrast_range: float = 0.15

    # -- Label Smoothing ----------------------------------------------
    label_smoothing: float = 0.05

    # -- Preprocessing ------------------------------------------------
    eps: float = 1e-8

    # -- Ensemble -----------------------------------------------------
    ensemble_seeds: List[int] = field(default_factory=lambda: [42, 137])

    # -- Factorial Contrastive ----------------------------------------
    use_factorial: bool = True
    pen_embed_dim: int = 128
    pen_hidden_dim: int = 512
    fac_temperature: float = 0.07
    fac_lambda_max: float = 0.5
    fac_warmup_start: int = 5
    fac_warmup_end: int = 15
    fac_num_writers: int = 6
    fac_pens_per_writer: int = 2
    fac_samples_per_cell: int = 2

    # -- Polar Encoder ------------------------------------------------
    use_polar: bool = True
    polar_radial: int = 64
    polar_angular: int = 256
    polar_embed_dim: int = 512
    polar_lr: float = 1e-4
    polar_invariant: bool = True
    polar_circ_shift_aug: bool = True
    polar_gate_init: float = 1.0

    # -- NetVLAD -------------------------------------------------------
    use_vlad: bool = True
    vlad_clusters: int = 64  

    # -- Pen Classification Head ----------------------------------------
    use_pen_head: bool = True
    pen_head_dim: int = 8  # 8 pen types
    pen_head_lambda: float = 0.3  # weight for pen CE loss
    pen_head_warmup: int = 5 
    bad_pen_writers: List[str] = field(default_factory=lambda: ["W41", "W50"])

    # -- Handcrafted Features -----------------------------------------
    use_handcrafted: bool = False
    handcrafted_dim: int = 64
    handcrafted_embed_dim: int = 512
    handcrafted_lr: float = 2e-4
    handcrafted_gate_init: float = 0.5
    handcrafted_n_fourier: int = 16
    handcrafted_n_radial_bins: int = 16

    # -- System -------------------------------------------------------
    num_workers: int = 2  
    pin_memory: bool = True
    device: str = "cuda"
    seed: int = 42
    use_amp: bool = True
    use_compile: bool = False  


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    
    seed = torch.initial_seed() % (2**32) + worker_id
    np.random.seed(seed)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, 'rng'):
        worker_info.dataset.rng = np.random.default_rng(seed)


# ======================================================================
# 2. Preprocessing
# ======================================================================

@dataclass
class PreprocessStats:
    adaptive_fallback_raw: int = 0
    adaptive_fallback_final: int = 0
    uniform_fallback_count: int = 0

    def log_rates(self, n):
        N = max(n, 1)
        print(f"  Adaptive fallback raw:   {self.adaptive_fallback_raw}/{N} ({100*self.adaptive_fallback_raw/N:.1f}%)")
        print(f"  Adaptive fallback final: {self.adaptive_fallback_final}/{N} ({100*self.adaptive_fallback_final/N:.1f}%)")
        print(f"  Uniform pooling:         {self.uniform_fallback_count}/{N} ({100*self.uniform_fallback_count/N:.1f}%)")

    def check_alerts(self, n):
        N = max(n, 1)
        alerts = []
        if self.adaptive_fallback_raw / N > 0.05: alerts.append("WARNING: >5% adaptive fallback raw")
        if self.adaptive_fallback_final / N > 0.05: alerts.append("WARNING: >5% adaptive fallback final")
        if self.uniform_fallback_count / N > 0.0: alerts.append("WARNING: uniform pooling triggered")
        return alerts


def to_gray(image, mode="cv2"):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if mode == "cv2" else cv2.COLOR_RGB2GRAY)


def compute_ink_mask(image, mode="cv2"):
    gray = to_gray(image, mode)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    used_adaptive = False
    if mask.mean() < 5 or mask.mean() > 250:
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        used_adaptive = True
    return mask.astype(np.float32) / 255.0, used_adaptive



def apply_intensity_jitter(image, brightness_range=0.15, contrast_range=0.15, rng=None):
    if rng is None: rng = np.random.default_rng()
    bf = 1.0 + rng.uniform(-brightness_range, brightness_range)
    cf = 1.0 + rng.uniform(-contrast_range, contrast_range)
    img = image.astype(np.float32)
    img = ((img - img.mean()) * cf + img.mean()) * bf
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_random_rotation(image, max_deg=15.0, rng=None):
    """Rotate image by a random angle in [-max_deg, +max_deg] with white fill."""
    if rng is None: rng = np.random.default_rng()
    angle = rng.uniform(-max_deg, max_deg)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def apply_scanner_bridging(image, rng=None):
    if image is None or image.size == 0:
        return image
    try:
        return _apply_scanner_bridging_impl(image, rng)
    except cv2.error:
        return image  

def _apply_scanner_bridging_impl(image, rng=None):
    
    if rng is None: rng = np.random.default_rng()
    img = image.copy()
    h, w = img.shape[:2]

    
    if rng.random() < 0.5:
        scale = rng.uniform(0.5, 2.0)
        new_h, new_w = max(16, int(h * scale)), max(16, int(w * scale))
        interp_down = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        img = cv2.resize(img, (new_w, new_h), interpolation=interp_down)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. Morphological augmentation 
    if rng.random() < 0.4:
        kernel_size = rng.choice([2, 3])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if rng.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)   # thinner strokes
        else:
            img = cv2.dilate(img, kernel, iterations=1)  # thicker strokes

    # 3. Random Gaussian blur/sharpen
    if rng.random() < 0.3:
        if rng.random() < 0.5:
            ksize = rng.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        else:
            # Unsharp masking
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            alpha = rng.uniform(1.2, 1.8)
            img = cv2.addWeighted(img, alpha, blurred, 1 - alpha, 0)
            img = np.clip(img, 0, 255).astype(np.uint8)

    # 4. Random gamma correction (scanner exposure differences)
    if rng.random() < 0.5:
        gamma = rng.uniform(0.5, 1.8)
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
        if len(img.shape) == 3:
            for c in range(img.shape[2]):
                img[:, :, c] = cv2.LUT(img[:, :, c], lut)
        else:
            img = cv2.LUT(img, lut)

    # 5. JPEG compression artifacts (different image processing pipelines)
    if rng.random() < 0.3:
        quality = rng.integers(40, 95)
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR if len(img.shape) == 3 else cv2.IMREAD_GRAYSCALE)

    # 6. Background intensity shift (different paper/scanner whiteness)
    if rng.random() < 0.4:
        shift = rng.integers(-30, 30)
        img = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)

    # 7. Random padding/border crop (different image cropping in pipelines)
    if rng.random() < 0.3:
        pad = rng.integers(2, 15)
        if rng.random() < 0.5:
            # Add white border then resize back
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                      cv2.BORDER_CONSTANT, value=(255, 255, 255))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        else:
            # Crop edges (if image is big enough)
            if h > 2 * pad + 16 and w > 2 * pad + 16:
                img = img[pad:h-pad, pad:w-pad]
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    return img


def compute_distance_transform_image(image):
    
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Otsu, INVERTED so ink=255 and background=0 
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dt_max = float(dt.max())
    if dt_max > 0:
        dt_uint8 = (dt / dt_max * 255.0).astype(np.uint8)
    else:
        dt_uint8 = np.zeros_like(gray, dtype=np.uint8)
    return np.stack([dt_uint8, dt_uint8, dt_uint8], axis=-1)


def compute_plain_skeleton_image(image):

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    try:
        skel = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except (AttributeError, cv2.error):
        # Iterative thinning fallback (slower)
        img_iter = binary.copy()
        skel = np.zeros_like(img_iter)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(img_iter, element)
            opened = cv2.dilate(eroded, element)
            temp = cv2.subtract(img_iter, opened)
            skel = cv2.bitwise_or(skel, temp)
            img_iter = eroded.copy()
            if cv2.countNonZero(img_iter) == 0:
                break
    return np.stack([skel, skel, skel], axis=-1)


def preprocess_image(image, mode="cv2", image_size=224,
                     apply_jitter=False, jitter_brightness=0.15, jitter_contrast=0.15,
                     apply_rotation=False, rotation_max_deg=15.0,
                     apply_hflip=False,
                     apply_scanner_aug=False,
                     rng=None, stats=None):
  
    if apply_scanner_aug:
        image = apply_scanner_bridging(image, rng)
   
    final_image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if apply_rotation:
        final_image = apply_random_rotation(final_image, rotation_max_deg, rng)
    if apply_hflip:
        final_image = np.ascontiguousarray(final_image[:, ::-1])
    if apply_jitter:
        final_image = apply_intensity_jitter(final_image, jitter_brightness, jitter_contrast, rng)
    
    if _IS_SKELETON_MODE:
        final_image = compute_distance_transform_image(final_image)
    elif _IS_PLAIN_SKELETON_MODE:
        final_image = compute_plain_skeleton_image(final_image)
   
    mask_final, adaptive_final = compute_ink_mask(final_image, mode)
    if stats and adaptive_final: stats.adaptive_fallback_final += 1
    return final_image, mask_final


def ink_weighted_pooling(patch_tokens, mask_float, eps=1e-8):
    num_patches = patch_tokens.shape[1]
    grid_size = int(num_patches ** 0.5)
    assert grid_size * grid_size == num_patches, \
        f"Non-square patch count {num_patches}, expected {grid_size}^2"
    mask_patches = F.adaptive_avg_pool2d(mask_float.unsqueeze(1), (grid_size, grid_size)).flatten(1)
    mask_sum = mask_patches.sum(dim=1, keepdim=True)
    uniform = torch.ones_like(mask_patches) / mask_patches.shape[1]
    use_uniform = (mask_sum < eps).float()
    mask_patches = (1 - use_uniform) * mask_patches + use_uniform * uniform
    num_uniform = int(use_uniform.sum().item())
    weights = mask_patches / (mask_patches.sum(dim=1, keepdim=True) + eps)
    pooled = (patch_tokens * weights.unsqueeze(-1)).sum(dim=1)
    return pooled, num_uniform


# -- Polar Preprocessing ----------------------------------------------

def find_circle_center(image, mode="cv2"):
    """4-stage fallback: contour -> distance transform -> moments -> image center."""
    gray = to_gray(image, mode)
    h, w = gray.shape

    # Stage 1: Largest contour -> minimum enclosing circle
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 0.01 * h * w:
            (cx, cy), radius = cv2.minEnclosingCircle(largest)
            if 0.1 * min(h, w) < radius < 0.9 * max(h, w):
                return (cx, cy), radius

    # Stage 2: Distance transform
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    if max_val > 0.05 * min(h, w):
        return (float(max_loc[0]), float(max_loc[1])), max_val

    # Stage 3: Moments
    M = cv2.moments(thresh)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return (cx, cy), min(h, w) / 2.0

    # Stage 4: Image center
    return (w / 2.0, h / 2.0), min(h, w) / 2.0


def extract_polar_strip(image, center, radius, radial=64, angular=256):
    """cv2.warpPolar to create [radial, angular] polar strip."""
    gray = to_gray(image, "cv2") if len(image.shape) == 3 else image
    polar = cv2.warpPolar(gray, (angular, radial), center, radius,
                          cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)
    return polar.astype(np.float32) / 255.0


def compute_centering_quality(polar_strip):
    """QC metric: std of darkest row index across angular columns."""
    darkest_rows = np.argmin(polar_strip, axis=0)
    return float(np.std(darkest_rows))


def preprocess_polar(image, mode="cv2", radial=64, angular=256,
                     circ_shift_aug=False, rng=None):
    """Full polar preprocessing pipeline."""
    center, radius = find_circle_center(image, mode)
    polar = extract_polar_strip(image, center, radius, radial, angular)
    if circ_shift_aug:
        if rng is None:
            rng = np.random.default_rng()
        shift = rng.integers(0, angular)
        polar = np.roll(polar, shift, axis=1)
    return polar  # [radial, angular], float32 [0,1]


# -- Handcrafted Feature Extraction -----------------------------------

def extract_handcrafted_features(image, mode="cv2", n_fourier=16, n_radial_bins=16):
    
    feat = np.zeros(64, dtype=np.float32)
    try:
        gray = to_gray(image, mode)
        h, w = gray.shape

        # Threshold and find contours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return feat
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 10:
            return feat
        perimeter = cv2.arcLength(contour, True)

        # ---- Group A: Shape (8D, indices 0-7) ----
        # circularity
        feat[0] = (4 * np.pi * area / (perimeter ** 2 + 1e-8))

        # eccentricity, aspect_ratio, tilt_angle from fitEllipse (needs >= 5 points)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (cx_e, cy_e), (minor_ax, major_ax), angle = ellipse
            minor_ax = max(minor_ax, 1e-8)
            major_ax = max(major_ax, 1e-8)
            if minor_ax > major_ax:
                minor_ax, major_ax = major_ax, minor_ax
            ratio = minor_ax / major_ax
            feat[1] = np.sqrt(max(0, 1 - ratio ** 2))  # eccentricity
            feat[2] = ratio  # aspect_ratio
            feat[3] = angle / 180.0  # tilt_angle normalized
        else:
            feat[1] = 0.0
            feat[2] = 1.0
            feat[3] = 0.0

        # solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        feat[4] = area / (hull_area + 1e-8)

        # solidity_enc (area / enclosing circle area)
        _, enc_radius = cv2.minEnclosingCircle(contour)
        feat[5] = area / (np.pi * enc_radius ** 2 + 1e-8)

        # area_ratio
        feat[6] = area / (h * w + 1e-8)

        # max_angular_gap: largest gap in contour angles from centroid
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx_m = M["m10"] / M["m00"]
            cy_m = M["m01"] / M["m00"]
        else:
            cx_m, cy_m = w / 2.0, h / 2.0
        pts = contour.reshape(-1, 2).astype(np.float64)
        angles = np.arctan2(pts[:, 1] - cy_m, pts[:, 0] - cx_m)
        angles_sorted = np.sort(angles)
        if len(angles_sorted) > 1:
            gaps = np.diff(angles_sorted)
            wrap_gap = (2 * np.pi) - (angles_sorted[-1] - angles_sorted[0])
            max_gap = max(np.max(gaps), wrap_gap)
            feat[7] = max_gap / np.pi  # normalized by pi
        else:
            feat[7] = 2.0

        # ---- Group B: Stroke Width (4D, indices 8-11) ----
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        ink_pixels = dist[thresh > 0]
        if len(ink_pixels) > 10:
            sw_mean = np.mean(ink_pixels)
            sw_std = np.std(ink_pixels)
            feat[8] = sw_mean
            feat[9] = sw_std
            # skewness
            if sw_std > 1e-8:
                feat[10] = np.mean(((ink_pixels - sw_mean) / sw_std) ** 3)
                # kurtosis
                feat[11] = np.mean(((ink_pixels - sw_mean) / sw_std) ** 4) - 3.0
            else:
                feat[10] = 0.0
                feat[11] = 0.0

        # ---- Group C: Curvature (6D, indices 12-17) ----
        pts_c = contour.reshape(-1, 2).astype(np.float64)
        if len(pts_c) > 5:
            # Smooth contour slightly for stable derivatives
            dx = np.gradient(pts_c[:, 0])
            dy = np.gradient(pts_c[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            denom = (dx ** 2 + dy ** 2) ** 1.5 + 1e-8
            curvature = np.abs(dx * ddy - dy * ddx) / denom
            # Clip extreme curvature values
            curvature = np.clip(curvature, 0, 10.0)
            feat[12] = np.mean(curvature)
            feat[13] = np.std(curvature)
            feat[14] = np.max(curvature)
            feat[15] = np.median(curvature)
            q75, q25 = np.percentile(curvature, [75, 25])
            feat[16] = q75 - q25  # IQR
            # n_inflections_norm: count sign changes in curvature derivative
            curv_diff = np.diff(np.sign(np.gradient(curvature)))
            n_inflections = np.sum(curv_diff != 0)
            feat[17] = n_inflections / (len(curvature) + 1e-8)

        # ---- Group D: Ink Texture (8D, indices 18-25) ----
        gray_f = gray.astype(np.float32) / 255.0
        feat[18] = np.mean(gray_f)  # ink_mean (inverted: lower = darker)
        feat[19] = np.std(gray_f)   # ink_std
        ink_mask = thresh.astype(np.float32) / 255.0
        feat[20] = np.mean(ink_mask)  # ink_ratio

        # Quadrant densities and asymmetry
        mid_h, mid_w = h // 2, w // 2
        quads = [
            ink_mask[:mid_h, :mid_w],      # top-left
            ink_mask[:mid_h, mid_w:],       # top-right
            ink_mask[mid_h:, :mid_w],       # bottom-left
            ink_mask[mid_h:, mid_w:],       # bottom-right
        ]
        quad_means = [np.mean(q) if q.size > 0 else 0.0 for q in quads]
        feat[21] = np.std(quad_means)  # quad_asymmetry
        for i, qm in enumerate(quad_means):
            feat[22 + i] = qm

        # ---- Group E: Fourier Descriptors (16D, indices 26-41) ----
        pts_f = contour.reshape(-1, 2).astype(np.float64)
        if len(pts_f) > n_fourier * 2:
            complex_pts = pts_f[:, 0] + 1j * pts_f[:, 1]
            fft_coeffs = np.fft.fft(complex_pts)
            magnitudes = np.abs(fft_coeffs)
            # Normalize by DC component
            dc = magnitudes[0] + 1e-8
            # Take first n_fourier coefficients (skip DC)
            n_take = min(n_fourier, len(magnitudes) - 1)
            feat[26:26 + n_take] = magnitudes[1:1 + n_take] / dc

        # ---- Group F: Radial Profile (22D, indices 42-63) ----
        # Angular-binned mean radius from centroid
        radii = np.sqrt((pts[:, 0] - cx_m) ** 2 + (pts[:, 1] - cy_m) ** 2)
        mean_radius = np.mean(radii) + 1e-8
        bin_edges = np.linspace(-np.pi, np.pi, n_radial_bins + 1)
        bin_means = np.zeros(n_radial_bins, dtype=np.float64)
        for bi in range(n_radial_bins):
            mask = (angles >= bin_edges[bi]) & (angles < bin_edges[bi + 1])
            if np.any(mask):
                bin_means[bi] = np.mean(radii[mask]) / mean_radius
            else:
                bin_means[bi] = 1.0  # neutral

        feat[42:42 + n_radial_bins] = bin_means

        # Radial statistics (6D, indices 58-63)
        feat[58] = np.std(bin_means)       # radial_std
        feat[59] = np.max(bin_means) - np.min(bin_means)  # radial_range
        feat[60] = np.sum(bin_means ** 2)  # radial_energy
        # radial_entropy
        bm_pos = np.clip(bin_means, 1e-8, None)
        bm_norm = bm_pos / bm_pos.sum()
        feat[61] = -np.sum(bm_norm * np.log(bm_norm + 1e-8))  # radial_entropy
        if np.std(bin_means) > 1e-8:
            centered = bin_means - np.mean(bin_means)
            std_bm = np.std(bin_means)
            feat[62] = np.mean((centered / std_bm) ** 3)  # radial_skew
            feat[63] = np.mean((centered / std_bm) ** 4) - 3.0  # radial_kurtosis

    except Exception:
        pass  
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


# ======================================================================
# 3. Dataset & Splits
# ======================================================================

def build_label_maps(df):
    writers = sorted(df["writer_id"].unique())
    writer2idx = {w: i for i, w in enumerate(writers)}
    idx2writer = {i: w for w, i in writer2idx.items()}
    return writer2idx, idx2writer


def create_writer_disjoint_splits(df, writer2idx, num_splits=5,
                                   held_out_per_split=7, train_val_ratio=0.8, seed=42):
    all_writers = sorted(writer2idx.keys())
    rng = np.random.default_rng(seed)
    splits = []
    for i in range(num_splits):
        shuffled = rng.permutation(all_writers)
        unknown_writers = set(shuffled[:held_out_per_split])
        known_writers = set(shuffled[held_out_per_split:])
        unknown_idx = df.index[df["writer_id"].isin(unknown_writers)].tolist()
        known_idx = df.index[df["writer_id"].isin(known_writers)].tolist()
        known_df = df.loc[known_idx]
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_val_ratio, random_state=seed+i)
        train_rel, val_rel = next(sss.split(known_df, known_df["writer_id"]))
        splits.append({
            "split_id": i,
            "train_idx": [known_idx[j] for j in train_rel],
            "val_idx": [known_idx[j] for j in val_rel],
            "unknown_idx": unknown_idx,
            "known_writers": known_writers,
            "unknown_writers": unknown_writers,
        })
    return splits


class WriterDataset(Dataset):
    def __init__(self, df, indices, data_dir, writer2idx,
                 image_size=224, image_load_mode="cv2", is_training=False,
                 jitter_brightness=0.15, jitter_contrast=0.15,
                 aug_hflip=False, aug_rotation_deg=0.0,
                 stats=None,
                 use_polar=False, polar_radial=64, polar_angular=256,
                 polar_circ_shift_aug=False,
                 use_handcrafted=False, hc_n_fourier=16, hc_n_radial_bins=16):
        self.df = df
        self.indices = indices
        self.data_dir = Path(data_dir)
        self.writer2idx = writer2idx
        self.image_size = image_size
        self.mode = image_load_mode
        self.is_training = is_training
        self.jitter_brightness = jitter_brightness
        self.jitter_contrast = jitter_contrast
        self.aug_hflip = aug_hflip
        self.aug_rotation_deg = aug_rotation_deg
        self.stats = stats
        self.use_polar = use_polar
        self.polar_radial = polar_radial
        self.polar_angular = polar_angular
        self.polar_circ_shift_aug = polar_circ_shift_aug and is_training
        self.use_handcrafted = use_handcrafted
        self.hc_n_fourier = hc_n_fourier
        self.hc_n_radial_bins = hc_n_radial_bins
        self.rng = np.random.default_rng()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        self._polar_cache = {}
        self._hc_cache = {}

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        row = self.df.iloc[self.indices[idx]]
        df_idx = self.indices[idx]
        image = cv2.imread(str(self.data_dir / row["image_path"]))
        if image is None: raise FileNotFoundError(str(self.data_dir / row["image_path"]))

        hc_feats = None
        if self.use_handcrafted:
            if df_idx in self._hc_cache:
                hc_feats = self._hc_cache[df_idx]
            else:
                hc_feats = torch.from_numpy(extract_handcrafted_features(
                    image, mode=self.mode,
                    n_fourier=self.hc_n_fourier, n_radial_bins=self.hc_n_radial_bins))
                self._hc_cache[df_idx] = hc_feats

        polar_strip = None
        if self.use_polar:
            if df_idx in self._polar_cache:
                polar = self._polar_cache[df_idx]
            else:
                polar = preprocess_polar(image, mode=self.mode,
                                         radial=self.polar_radial, angular=self.polar_angular,
                                         circ_shift_aug=False, rng=None)
                self._polar_cache[df_idx] = polar
           
            if self.polar_circ_shift_aug:
                shift = self.rng.integers(0, self.polar_angular)
                polar = np.roll(polar, shift, axis=1)
            polar_strip = torch.from_numpy(polar).unsqueeze(0)  # [1, R, A]

        # Standard preprocessing
        do_hflip = self.is_training and self.aug_hflip and self.rng.random() < 0.5
        do_rotation = self.is_training and self.aug_rotation_deg > 0 and self.rng.random() < 0.5
        do_jitter = self.is_training and self.rng.random() < 0.7
        do_scanner = self.is_training  # scanner-bridging always during training
        final_image, mask_float = preprocess_image(
            image, mode=self.mode, image_size=self.image_size,
            apply_jitter=do_jitter,
            jitter_brightness=self.jitter_brightness,
            jitter_contrast=self.jitter_contrast,
            apply_rotation=do_rotation,
            rotation_max_deg=self.aug_rotation_deg,
            apply_hflip=do_hflip,
            apply_scanner_aug=do_scanner,
            rng=self.rng, stats=self.stats,
        )
        img_tensor = torch.from_numpy(final_image).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor[[2, 1, 0], ...]  # BGR -> RGB
        img_tensor = (img_tensor - self.mean) / self.std
        mask_tensor = torch.from_numpy(mask_float).float()
        writer_label = self.writer2idx.get(row["writer_id"], -1)
        pen_label = int(row["pen_id"]) - 1  # 0-indexed (1-8 -> 0-7)

        result = {"image": img_tensor, "mask": mask_tensor,
                  "writer_label": writer_label, "pen_label": pen_label}
        if polar_strip is not None:
            result["polar_strip"] = polar_strip
        if hc_feats is not None:
            result["handcrafted"] = hc_feats
        return result


class WriterTestDataset(Dataset):
    def __init__(self, df, data_dir, image_size=224, image_load_mode="cv2",
                 apply_jitter=False, jitter_brightness=0.15, jitter_contrast=0.15,
                 apply_hflip=False,
                 use_polar=False, polar_radial=64, polar_angular=256,
                 use_handcrafted=False, hc_n_fourier=16, hc_n_radial_bins=16):
        self.df = df
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.mode = image_load_mode
        self.apply_jitter = apply_jitter
        self.jitter_brightness = jitter_brightness
        self.jitter_contrast = jitter_contrast
        self.apply_hflip = apply_hflip
        self.use_polar = use_polar
        self.polar_radial = polar_radial
        self.polar_angular = polar_angular
        self.use_handcrafted = use_handcrafted
        self.hc_n_fourier = hc_n_fourier
        self.hc_n_radial_bins = hc_n_radial_bins
        self.rng = np.random.default_rng()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self._polar_cache = {}
        self._hc_cache = {}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(str(self.data_dir / row["image_path"]))
        if image is None: raise FileNotFoundError(str(self.data_dir / row["image_path"]))

        # Handcrafted features (cached)
        hc_feats = None
        if self.use_handcrafted:
            if idx in self._hc_cache:
                hc_feats = self._hc_cache[idx]
            else:
                hc_feats = torch.from_numpy(extract_handcrafted_features(
                    image, mode=self.mode,
                    n_fourier=self.hc_n_fourier, n_radial_bins=self.hc_n_radial_bins))
                self._hc_cache[idx] = hc_feats

        
        polar_strip = None
        if self.use_polar:
            if idx in self._polar_cache:
                polar = self._polar_cache[idx]
            else:
                polar = preprocess_polar(image, mode=self.mode,
                                         radial=self.polar_radial, angular=self.polar_angular,
                                         circ_shift_aug=False, rng=self.rng)
                self._polar_cache[idx] = polar
            polar_strip = torch.from_numpy(polar).unsqueeze(0)  # [1, R, A]

        final_image, mask_float = preprocess_image(
            image, mode=self.mode, image_size=self.image_size,
            apply_jitter=self.apply_jitter,
            jitter_brightness=self.jitter_brightness,
            jitter_contrast=self.jitter_contrast,
            apply_hflip=self.apply_hflip,
            rng=self.rng,
        )
        img_tensor = torch.from_numpy(final_image).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor[[2, 1, 0], ...]
        img_tensor = (img_tensor - self.mean) / self.std
        mask_tensor = torch.from_numpy(mask_float).float()
        result = {"image": img_tensor, "mask": mask_tensor}
        if polar_strip is not None:
            result["polar_strip"] = polar_strip
        if hc_feats is not None:
            result["handcrafted"] = hc_feats
        return result


class PenAxisBatchSampler:
  
    def __init__(self, df, indices, writer2idx, num_writers=4,
                 pens_per_writer=2, samples_per_cell=4, rng_seed=42):
        self.num_writers = num_writers
        self.pens_per_writer = pens_per_writer
        self.samples_per_cell = samples_per_cell
        self.batch_size = num_writers * pens_per_writer * samples_per_cell
        self.rng = np.random.default_rng(rng_seed)

        # Build index: writer_label -> pen_label -> [dataset_idx]
        self.writer_pen_index = {}
        self.writer_to_pens = {}

        for dataset_idx, df_idx in enumerate(indices):
            row = df.iloc[df_idx]
            pen_label = int(row["pen_id"]) - 1
            writer_label = writer2idx.get(row["writer_id"], -1)
            if writer_label == -1:
                continue
            if writer_label not in self.writer_pen_index:
                self.writer_pen_index[writer_label] = {}
                self.writer_to_pens[writer_label] = set()
            self.writer_to_pens[writer_label].add(pen_label)
            if pen_label not in self.writer_pen_index[writer_label]:
                self.writer_pen_index[writer_label][pen_label] = []
            self.writer_pen_index[writer_label][pen_label].append(dataset_idx)

        self.eligible_writers = sorted(
            [w for w, pens in self.writer_to_pens.items() if len(pens) >= pens_per_writer]
        )
        self.num_batches = max(1, len(indices) // self.batch_size)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            n_pick = min(self.num_writers, len(self.eligible_writers))
            writers = self.rng.choice(self.eligible_writers, n_pick,
                                      replace=n_pick > len(self.eligible_writers))
            for writer in writers:
                avail_pens = sorted(self.writer_to_pens[writer])
                pens = self.rng.choice(avail_pens, self.pens_per_writer,
                                       replace=len(avail_pens) < self.pens_per_writer)
                for pen in pens:
                    pool = self.writer_pen_index[writer][pen]
                    samples = self.rng.choice(pool, self.samples_per_cell,
                                              replace=len(pool) < self.samples_per_cell)
                    batch.extend(samples.tolist())

            yield batch


# ======================================================================
# 4. Model
# ======================================================================

class NetVLAD(nn.Module):
    
    def __init__(self, feature_dim, num_clusters=64, normalize_input=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.normalize_input = normalize_input
        # Soft-assignment: linear layer to compute cluster membership
        self.conv = nn.Conv1d(feature_dim, num_clusters, kernel_size=1, bias=True)
        # Cluster centers — initialized properly via init_clusters(), not random
        self.centroids = nn.Parameter(torch.randn(num_clusters, feature_dim) * 0.01)
        # Output dimension: num_clusters * feature_dim, then project down
        self.out_dim = num_clusters * feature_dim
        self._initialized = False

    def init_clusters(self, patch_descriptors):
       
        from sklearn.cluster import KMeans
        print(f"  NetVLAD K-means init: {patch_descriptors.shape[0]} descriptors → {self.num_clusters} clusters...")
        descs = patch_descriptors.numpy()
        if self.normalize_input:
            norms = np.linalg.norm(descs, axis=1, keepdims=True) + 1e-8
            descs = descs / norms
        km = KMeans(n_clusters=self.num_clusters, n_init=3, max_iter=100, random_state=42)
        km.fit(descs)
        centers = torch.from_numpy(km.cluster_centers_).float()

        # Set centroids
        self.centroids.data.copy_(centers)

        alpha = 1.0  # soft assignment temperature
        self.conv.weight.data.copy_((2.0 * alpha * centers).unsqueeze(2))  # [K, D, 1]
        self.conv.bias.data.copy_(-alpha * (centers ** 2).sum(dim=1))      # [K]

        self._initialized = True
        print(f"  NetVLAD K-means init: done (inertia={km.inertia_:.1f})")

    def forward(self, x, mask=None):
        """
        x: [B, N, D] local descriptors (e.g., DINOv2 patch tokens)
        mask: [B, N] optional foreground mask (1=ink, 0=background)
        Returns: [B, num_clusters * D] VLAD descriptor
        """
        B, N, D = x.shape
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=2)

        # Soft assignment: [B, K, N]
        soft_assign = self.conv(x.permute(0, 2, 1))  # [B, K, N]
        if mask is not None:
            mask_bool = mask.unsqueeze(1).bool()  # [B, 1, N]
            # If a sample has NO foreground patches, fall back to uniform (all patches)
            has_fg = mask_bool.any(dim=2, keepdim=True)  # [B, 1, 1]
            # Only mask background when sample actually has foreground
            effective_mask = mask_bool | ~has_fg  
            soft_assign = soft_assign.masked_fill(~effective_mask, float('-inf'))
        soft_assign = F.softmax(soft_assign, dim=2)  # [B, K, N]

       
        vlad = torch.zeros(B, self.num_clusters, D, device=x.device, dtype=x.dtype)
        for k in range(self.num_clusters):
            # residuals: [B, N, D], assignment weights: [B, N, 1]
            res_k = x - self.centroids[k].unsqueeze(0).unsqueeze(0)  # [B, N, D]
            w_k = soft_assign[:, k, :].unsqueeze(2)  # [B, N, 1]
            vlad[:, k] = (res_k * w_k).sum(dim=1)  # [B, D]

        # Intra-normalize (per-cluster L2 norm)
        vlad = F.normalize(vlad, p=2, dim=2)
        # Flatten
        vlad = vlad.reshape(B, -1)  # [B, K*D]
        # L2 normalize final descriptor
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


class LoRALinear(nn.Module):
    def __init__(self, original, rank=8, alpha=16):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(original.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))
        original.weight.requires_grad = False
        if original.bias is not None: original.bias.requires_grad = False

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


def apply_lora(model, rank=8, alpha=16, last_n_blocks=4):
    blocks = model.blocks
    n = len(blocks)
    lora_params = []
    for i in range(n - last_n_blocks, n):
        # LoRA on attention QKV
        lora_qkv = LoRALinear(blocks[i].attn.qkv, rank, alpha)
        blocks[i].attn.qkv = lora_qkv
        lora_params.extend([lora_qkv.lora_A, lora_qkv.lora_B])
        # LoRA on MLP fc1 and fc2
        if hasattr(blocks[i], 'mlp') and hasattr(blocks[i].mlp, 'fc1'):
            lora_fc1 = LoRALinear(blocks[i].mlp.fc1, rank, alpha)
            blocks[i].mlp.fc1 = lora_fc1
            lora_params.extend([lora_fc1.lora_A, lora_fc1.lora_B])
        if hasattr(blocks[i], 'mlp') and hasattr(blocks[i].mlp, 'fc2'):
            lora_fc2 = LoRALinear(blocks[i].mlp.fc2, rank, alpha)
            blocks[i].mlp.fc2 = lora_fc2
            lora_params.extend([lora_fc2.lora_A, lora_fc2.lora_B])
    return model, lora_params


def enable_gradient_checkpointing(backbone, last_n_blocks=9):
    
    blocks = backbone.blocks
    n = len(blocks)
    for i in range(n - last_n_blocks, n):
        orig_forward = blocks[i].forward
        # Use a closure to capture the correct block reference
        def make_ckpt_forward(fn):
            def ckpt_forward(*args, **kwargs):
                return grad_checkpoint(fn, *args, use_reentrant=False, **kwargs)
            return ckpt_forward
        blocks[i].forward = make_ckpt_forward(orig_forward)
    print(f"  Gradient checkpointing: enabled for last {last_n_blocks} blocks")


def get_arcface_margin(epoch, max_margin=0.5, warmup_start=0, warmup_end=15):
    """Linear ramp: 0 at warmup_start, max_margin at warmup_end, constant after."""
    if epoch < warmup_start:
        return 0.0
    if epoch >= warmup_end:
        return max_margin
    return max_margin * (epoch - warmup_start) / (warmup_end - warmup_start)


class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim, num_classes, scale=32.0, margin=0.3):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, embeddings, labels=None):
        with torch.amp.autocast("cuda", enabled=False):
            embeddings = embeddings.float()
            proto_norm = F.normalize(self.prototypes.float(), dim=1)
            cosine = F.linear(embeddings, proto_norm)
        if labels is not None and self.training:
            theta = torch.acos(cosine.clamp(-1+1e-7, 1-1e-7))
            idx = torch.arange(len(labels), device=labels.device)
            target_theta = theta[idx, labels] + self.margin
            margin_cosine = cosine.clone()
            margin_cosine[idx, labels] = torch.cos(target_theta)
            logits = self.scale * margin_cosine
        else:
            logits = self.scale * cosine
        return {"logits": logits, "cosine": cosine}


class PenEmbeddingBranch(nn.Module):
    """MLP producing L2-normalized pen embeddings for contrastive loss during training."""
    def __init__(self, input_dim=512, hidden_dim=512, embed_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.mlp(x), p=2, dim=1)


class CircularPolarEncoder(nn.Module):
    """Small CNN for polar strip encoding."""
    def __init__(self, radial=64, angular=256, embed_dim=512, invariant=True):
        super().__init__()
        self.invariant = invariant
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
        )
        if invariant:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(256, embed_dim)
        else:
            # Phase-preserving: avg pool over radial, attention pool over angular
            self.attn = nn.Linear(256, 1)
            self.fc = nn.Linear(256, embed_dim)

    def forward(self, x):
        """x: [B, 1, radial, angular]"""
        feat = self.conv(x)  # [B, 256, R/16, A/16]
        if self.invariant:
            pooled = self.pool(feat).flatten(1)  # [B, 256]
        else:
            pooled = feat.mean(dim=2)  # [B, 256, A/16]
            pooled = pooled.permute(0, 2, 1)  # [B, A/16, 256]
            attn_w = torch.softmax(self.attn(pooled), dim=1)  # [B, A/16, 1]
            pooled = (pooled * attn_w).sum(dim=1)  # [B, 256]
        return self.fc(pooled)  # [B, embed_dim]


class HandcraftedEncoder(nn.Module):
    """MLP encoder for handcrafted geometric/texture features."""
    def __init__(self, input_dim=64, embed_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x)


class WriterIDModel(nn.Module):
    def __init__(self, backbone, num_writers=44, dinov2_dim=768,
                 projection_dim=512, writer_embed_dim=256,
                 arcface_scale=32.0, arcface_margin=0.3,
                 use_factorial=False, pen_embed_dim=128, pen_hidden_dim=512,
                 use_polar=False, polar_encoder=None, polar_gate_init=1.0,
                 use_handcrafted=False, hc_encoder=None, hc_gate_init=0.5,
                 use_vlad=False, vlad_clusters=64):
        super().__init__()
        self.backbone = backbone
        self.use_polar = use_polar
        self.use_factorial = use_factorial
        self.use_handcrafted = use_handcrafted
        self.use_vlad = use_vlad

        if use_vlad:
            # NetVLAD on patch tokens + CLS token
            self.vlad = NetVLAD(dinov2_dim, num_clusters=vlad_clusters)
            # Project VLAD output (K*D) + CLS (D) down to projection_dim
            vlad_out_dim = vlad_clusters * dinov2_dim
            self.vlad_proj = nn.Sequential(
                nn.Linear(vlad_out_dim, projection_dim),
                nn.LayerNorm(projection_dim), nn.GELU(),
            )
            # CLS + VLAD fusion
            self.projection = nn.Sequential(
                nn.Linear(projection_dim + dinov2_dim, projection_dim),
                nn.LayerNorm(projection_dim), nn.GELU(),
                nn.Dropout(0.15),
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(dinov2_dim * 2, projection_dim),
                nn.LayerNorm(projection_dim), nn.GELU(),
                nn.Dropout(0.15),
            )

        # Polar fusion
        if use_polar and polar_encoder is not None:
            self.polar_encoder = polar_encoder
            self.dino_ln = nn.LayerNorm(projection_dim)
            self.polar_ln = nn.LayerNorm(projection_dim)
            self.gate_alpha = nn.Parameter(torch.tensor(float(polar_gate_init)))
            self.fusion_mlp = nn.Sequential(
                nn.Linear(projection_dim, projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
            )

        # Handcrafted fusion
        if use_handcrafted and hc_encoder is not None:
            self.hc_encoder = hc_encoder
            self.hc_ln = nn.LayerNorm(projection_dim)
            self.gate_hc = nn.Parameter(torch.tensor(float(hc_gate_init)))
            # If polar is not used, we still need dino_ln and fusion_mlp
            if not hasattr(self, 'dino_ln'):
                self.dino_ln = nn.LayerNorm(projection_dim)
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(projection_dim, projection_dim),
                    nn.GELU(),
                    nn.Linear(projection_dim, projection_dim),
                )

        self.writer_proj = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim), nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(projection_dim, writer_embed_dim),
        )
        self.writer_head = ArcFaceHead(writer_embed_dim, num_writers, arcface_scale, arcface_margin)

        # Pen embedding branch (factorial contrastive)
        if use_factorial:
            self.pen_branch = PenEmbeddingBranch(projection_dim, pen_hidden_dim, pen_embed_dim)

        # Pen classification head
        self.pen_head = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 8),  # 8 pen types
        )

    def forward(self, images, masks, writer_labels=None, polar_strips=None, hc_feats=None, eps=1e-8):
       
        self.backbone.eval()
        out = self.backbone.forward_features(images)
        cls_token = out["x_norm_clstoken"]
        patch_tokens = out["x_norm_patchtokens"]
        grid_size = int(patch_tokens.shape[1] ** 0.5)
        assert grid_size * grid_size == patch_tokens.shape[1], (
            f"Non-square patch count: {patch_tokens.shape[1]}"
        )

        if self.use_vlad:
            # Foreground mask for VLAD: only ink patches contribute
            mask_patches = F.adaptive_avg_pool2d(
                masks.unsqueeze(1), (grid_size, grid_size)
            ).flatten(1)  # [B, num_patches]
            # Threshold: patches with >10% ink are foreground
            fg_mask = (mask_patches > 0.1).float()
            vlad_desc = self.vlad(patch_tokens, fg_mask)  # [B, K*D]
            vlad_feat = self.vlad_proj(vlad_desc)  # [B, proj_dim]
            combined = torch.cat([cls_token, vlad_feat], dim=1)
            num_uniform = 0
        else:
            patch_avg, num_uniform = ink_weighted_pooling(patch_tokens, masks, eps)
            combined = torch.cat([cls_token, patch_avg], dim=1)

        projected = self.projection(combined)

        # Multi-modality fusion
        has_polar = self.use_polar and polar_strips is not None and hasattr(self, 'polar_encoder')
        has_hc = self.use_handcrafted and hc_feats is not None and hasattr(self, 'hc_encoder')

        if has_polar or has_hc:
            fused = self.dino_ln(projected)
            if has_polar:
                polar_feat = self.polar_encoder(polar_strips)
                fused = fused + self.gate_alpha * self.polar_ln(polar_feat)
            if has_hc:
                hc_encoded = self.hc_encoder(hc_feats)
                fused = fused + self.gate_hc * self.hc_ln(hc_encoded)
            projected = self.fusion_mlp(fused)

        writer_emb = F.normalize(self.writer_proj(projected), p=2, dim=1)
        writer_out = self.writer_head(writer_emb, writer_labels)

        result = {
            "writer_logits": writer_out["logits"],
            "writer_cosine": writer_out["cosine"],
            "writer_emb": writer_emb,
            "num_uniform_fallbacks": num_uniform,
        }

        # Pen embedding for factorial contrastive loss
        if self.use_factorial and hasattr(self, 'pen_branch'):
            result["pen_emb"] = self.pen_branch(projected)

        # Pen classification
        if hasattr(self, 'pen_head'):
            result["pen_logits"] = self.pen_head(projected)

        if hasattr(self, 'gate_alpha'):
            result["gate_alpha"] = self.gate_alpha.item()
        if hasattr(self, 'gate_hc'):
            result["gate_hc"] = self.gate_hc.item()

        return result


class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
       
        self.shadow = {}
        self._buf = {}  
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.detach().cpu().clone()
                self._buf[n] = torch.empty_like(p.data, device="cpu")
        self.backup = {}

    def reinit_shadow(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self._buf[n].copy_(p.data)
                self.shadow[n].copy_(self._buf[n])

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self._buf[n].copy_(p.data)
                self.shadow[n].mul_(self.decay).add_(self._buf[n], alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.detach().cpu().clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


def build_writer_model(cfg):
    backbone = torch.hub.load(cfg.dinov2_repo, cfg.dinov2_variant, pretrained=True)
    backbone.eval()
    with torch.no_grad():
        out = backbone.forward_features(torch.randn(1, 3, 224, 224))
    assert "x_norm_clstoken" in out and "x_norm_patchtokens" in out
    print(f"  DINOv2: CLS {out['x_norm_clstoken'].shape}, patches {out['x_norm_patchtokens'].shape}")
    for p in backbone.parameters(): p.requires_grad = False
    backbone, lora_params = apply_lora(backbone, cfg.lora_rank, cfg.lora_alpha, cfg.lora_last_n_blocks)
    print(f"  LoRA: {sum(p.numel() for p in lora_params):,} params")
   
    unfreeze_params = []
    if cfg.unfreeze_norm_bias_blocks > 0:
        blocks = backbone.blocks
        n = len(blocks)
        for i in range(n - cfg.unfreeze_norm_bias_blocks, n):
            blk = blocks[i]
            for name, param in blk.named_parameters():
                is_norm = "norm1" in name or "norm2" in name
                is_bias = name.endswith(".bias")
                if (is_norm or is_bias) and not param.requires_grad:
                    param.requires_grad = True
                    unfreeze_params.append(param)
        print(f"  Unfrozen norm+bias: {sum(p.numel() for p in unfreeze_params):,} params in last {cfg.unfreeze_norm_bias_blocks} blocks")

    # Build polar encoder if needed
    polar_encoder = None
    if cfg.use_polar:
        polar_encoder = CircularPolarEncoder(
            cfg.polar_radial, cfg.polar_angular, cfg.polar_embed_dim, cfg.polar_invariant
        )

    # Build handcrafted encoder if needed
    hc_encoder = None
    if cfg.use_handcrafted:
        hc_encoder = HandcraftedEncoder(cfg.handcrafted_dim, cfg.handcrafted_embed_dim)

    model = WriterIDModel(backbone, cfg.num_known_writers, cfg.dinov2_embed_dim,
                          cfg.projection_dim, cfg.writer_embed_dim,
                          cfg.arcface_scale, cfg.arcface_margin,
                          use_factorial=cfg.use_factorial,
                          pen_embed_dim=cfg.pen_embed_dim,
                          pen_hidden_dim=cfg.pen_hidden_dim,
                          use_polar=cfg.use_polar,
                          polar_encoder=polar_encoder,
                          polar_gate_init=cfg.polar_gate_init,
                          use_handcrafted=cfg.use_handcrafted,
                          hc_encoder=hc_encoder,
                          hc_gate_init=cfg.handcrafted_gate_init,
                          use_vlad=cfg.use_vlad,
                          vlad_clusters=cfg.vlad_clusters)

    lora_param_ids = {id(p) for p in lora_params}
    unfreeze_param_ids = {id(p) for p in unfreeze_params}
    polar_param_ids = set()
    if cfg.use_polar and polar_encoder is not None:
        polar_param_ids = {id(p) for p in polar_encoder.parameters()}
    hc_param_ids = set()
    if cfg.use_handcrafted and hc_encoder is not None:
        hc_param_ids = {id(p) for p in hc_encoder.parameters()}

    exclude_ids = lora_param_ids | unfreeze_param_ids | polar_param_ids | hc_param_ids
    # gate_alpha / gate_hc are NOT in encoder.parameters(), so they go into head_params
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and id(p) not in exclude_ids]

    polar_params = list(polar_encoder.parameters()) if polar_encoder is not None else []
    hc_params = list(hc_encoder.parameters()) if hc_encoder is not None else []

    total_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_a = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_t:,} trainable / {total_a:,} total ({100*total_t/total_a:.2f}%)")
    if polar_encoder is not None:
        print(f"  Polar encoder: {sum(p.numel() for p in polar_params):,} params")
    if hc_encoder is not None:
        print(f"  Handcrafted encoder: {sum(p.numel() for p in hc_params):,} params")
    return model, lora_params, head_params, unfreeze_params, polar_params, hc_params


def init_vlad_clusters(model, df, data_dir, cfg, max_images=500):
    
    if not cfg.use_vlad or not hasattr(model, 'vlad') or model.vlad._initialized:
        return

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # Sample a subset of images (balanced across writers)
    rng = np.random.default_rng(cfg.seed)
    indices = rng.choice(len(df), size=min(max_images, len(df)), replace=False)

    all_patches = []
    model.backbone.eval()
    with torch.no_grad():
        for i in range(0, len(indices), 16):
            batch_idx = indices[i:i+16]
            imgs = []
            for idx in batch_idx:
                row = df.iloc[idx]
                img = cv2.imread(str(Path(data_dir) / row["image_path"]))
                if img is None:
                    continue
                img = cv2.resize(img, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_AREA)
                t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
                t = t[[2, 1, 0], ...]  # BGR->RGB
                imgs.append(t)
            if not imgs:
                continue
            batch = torch.stack(imgs)
            batch = (batch - mean) / std
            out = model.backbone.forward_features(batch)
            patches = out["x_norm_patchtokens"]  # [B, N, D]
            all_patches.append(patches.reshape(-1, patches.shape[-1]))

    all_patches = torch.cat(all_patches, dim=0)
    # Subsample if too many (K-means on >100K is slow)
    if len(all_patches) > 50000:
        sub_idx = rng.choice(len(all_patches), 50000, replace=False)
        all_patches = all_patches[sub_idx]

    model.vlad.init_clusters(all_patches)


def maybe_compile(model, cfg):
    """Apply torch.compile if available and enabled."""
    if not cfg.use_compile:
        return model
    if not hasattr(torch, 'compile'):
        print("  torch.compile not available (needs PyTorch >= 2.0)")
        return model
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile: enabled (reduce-overhead mode)")
    except Exception as e:
        print(f"  torch.compile failed, continuing without: {e}")
    return model


# ======================================================================
# 5. OOD Evaluation & Calibration
# ======================================================================

def extract_ood_scores(writer_logits, writer_cosine):
    """3 signals. All oriented: higher = more likely known."""
    max_cos = writer_cosine.max(axis=1)
    neg_energy = writer_logits.max(axis=1) + np.log(
        np.exp(writer_logits - writer_logits.max(axis=1, keepdims=True)).sum(axis=1) + 1e-10
    )
    probs = np.exp(writer_logits - writer_logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    neg_entropy = (probs * np.log(probs + 1e-10)).sum(axis=1)
    return np.stack([max_cos, neg_energy, neg_entropy], axis=1)


def fit_ood_calibrator(scores_list, labels_list):
    all_X = np.vstack(scores_list)
    all_y = np.concatenate(labels_list)
    cal = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(cv=5, scoring="roc_auc", class_weight="balanced", max_iter=2000),
    )
    cal.fit(all_X, all_y)
    return cal


def find_optimal_threshold(p_known, is_known, alpha=0.7, n_points=200):
    thresholds = np.linspace(p_known.min(), p_known.max(), n_points)
    best_score, best_tau = -1, thresholds[0]
    km, um = is_known == 1, is_known == 0
    for tau in thresholds:
        ka = (p_known[km] >= tau).mean() if km.sum() > 0 else 1.0
        ur = (p_known[um] < tau).mean() if um.sum() > 0 else 1.0
        score = alpha * ka + (1 - alpha) * ur
        if score > best_score:
            best_score, best_tau = score, tau
    return best_tau


def make_writer_predictions(writer_logits, writer_cosine, calibrator, threshold):
    scores = extract_ood_scores(writer_logits, writer_cosine)
    p_known = calibrator.predict_proba(scores)[:, 1]
    writer_raw = writer_cosine.argmax(axis=1)
    return {
        "writer_preds": np.where(p_known >= threshold, writer_raw, -1),
        "p_known": p_known,
    }


# ======================================================================
# 6. Loss Functions & Training Loop
# ======================================================================

def compute_factorial_contrastive_loss(writer_emb, pen_emb, writer_labels, pen_labels,
                                        temperature=0.07):
   
    with torch.amp.autocast("cuda", enabled=False):
        writer_emb = writer_emb.float()
        pen_emb = pen_emb.float()
        B = writer_emb.shape[0]
        device = writer_emb.device

        writer_sim = torch.mm(writer_emb, writer_emb.t()) / temperature
        pen_sim = torch.mm(pen_emb, pen_emb.t()) / temperature

        eye = torch.eye(B, device=device, dtype=torch.bool)
        writer_pos = (writer_labels.unsqueeze(0) == writer_labels.unsqueeze(1)) & (pen_labels.unsqueeze(0) != pen_labels.unsqueeze(1)) & ~eye
        pen_pos = (pen_labels.unsqueeze(0) == pen_labels.unsqueeze(1)) & (writer_labels.unsqueeze(0) != writer_labels.unsqueeze(1)) & ~eye
        all_neg = ~eye

        stats = {"writer_pos_pairs": int(writer_pos.sum().item()),
                 "pen_pos_pairs": int(pen_pos.sum().item()),
                 "orphans": 0}

        NEGINF = -1e9

        def _infonce(sim, pos_mask, neg_mask):
            has_pos = pos_mask.any(dim=1)
            if not has_pos.any():
                return torch.tensor(0.0, device=device), int((~has_pos).sum().item())
            pos_logits = sim.masked_fill(~pos_mask, NEGINF)
            neg_logits = sim.masked_fill(~neg_mask, NEGINF)
            log_numer = torch.logsumexp(pos_logits, dim=1)
            log_denom = torch.logsumexp(neg_logits, dim=1)
            loss = -(log_numer - log_denom)
            loss = loss[has_pos].mean()
            orphans = int((~has_pos).sum().item())
            return loss, orphans

        loss_w, orphans_w = _infonce(writer_sim, writer_pos, all_neg)
        loss_p, orphans_p = _infonce(pen_sim, pen_pos, all_neg)
        stats["orphans"] = orphans_w + orphans_p

        return loss_w + loss_p, stats


def get_factorial_lambda(epoch, warmup_start=5, warmup_end=15, lambda_max=0.5):

    if epoch < warmup_start:
        return 0.0
    if epoch >= warmup_end:
        return lambda_max
    return lambda_max * (epoch - warmup_start) / (warmup_end - warmup_start)


def log_batch_pair_stats(epoch_stats, epoch, lambda_fac):

    n = max(epoch_stats.get("n_batches", 1), 1)
    wp = epoch_stats.get("writer_pos_pairs", 0) / n
    pp = epoch_stats.get("pen_pos_pairs", 0) / n
    orph = epoch_stats.get("orphans", 0) / n
    fac_loss = epoch_stats.get("fac_loss", 0.0) / n
    print(f"    Fac: L_fac={fac_loss:.4f} lambda={lambda_fac:.3f} "
          f"w_pos={wp:.0f} p_pos={pp:.0f} orphans={orph:.1f}")


def build_optimizer_scheduler(lora_params, head_params, cfg, steps_per_epoch,
                              unfreeze_params=None, polar_params=None, hc_params=None):
    param_groups = [
        {"params": lora_params, "lr": cfg.lr_lora},
        {"params": head_params, "lr": cfg.lr_heads},
    ]
    if unfreeze_params:
        param_groups.append({"params": unfreeze_params, "lr": cfg.lr_unfreeze})
    if polar_params:
        param_groups.append({"params": polar_params, "lr": cfg.polar_lr})
    if hc_params:
        param_groups.append({"params": hc_params, "lr": cfg.handcrafted_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
    return optimizer, scheduler


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, cfg,
                    ema=None, scaler=None, ood_loader=None):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # ArcFace margin warmup
    current_margin = get_arcface_margin(epoch, cfg.arcface_margin,
                                         cfg.arcface_margin_warmup_start,
                                         cfg.arcface_margin_warmup_end)
    model.writer_head.margin = current_margin

    tot_loss = tot_arc = tot_ood = 0.0
    c_w = n = uf = 0
    fac_stats_accum = {"writer_pos_pairs": 0, "pen_pos_pairs": 0, "orphans": 0,
                       "fac_loss": 0.0, "n_batches": 0}
    lambda_fac = get_factorial_lambda(epoch, cfg.fac_warmup_start, cfg.fac_warmup_end,
                                      cfg.fac_lambda_max) if cfg.use_factorial else 0.0

    # Outlier exposure: cycle through OOD samples
    use_ood = (ood_loader is not None and epoch >= cfg.ood_train_warmup
               and cfg.ood_train_lambda > 0)
    ood_iter = iter(ood_loader) if use_ood else None

    optimizer.zero_grad()
    num_steps = len(dataloader)
    use_amp = scaler is not None
    for step, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        wl     = batch["writer_label"].to(device).long()
        assert (wl >= 0).all(), f"Negative label leaked into training batch: {wl}"

        polar_strips = batch.get("polar_strip")
        if polar_strips is not None:
            polar_strips = polar_strips.to(device)
        hc_feats = batch.get("handcrafted")
        if hc_feats is not None:
            hc_feats = hc_feats.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(images, masks, writer_labels=wl, polar_strips=polar_strips,
                        hc_feats=hc_feats, eps=cfg.eps)
            loss_arc = criterion(out["writer_logits"], wl)
            loss = loss_arc

            # Pen classification loss (skip W41=idx34, W50=idx42)
            if cfg.use_pen_head and epoch >= cfg.pen_head_warmup and "pen_logits" in out:
                pen_labels = batch["pen_label"].to(device).long()
                # Mask out bad pen writers (W41, W50)
                pen_valid = torch.ones(wl.shape[0], dtype=torch.bool, device=device)
                for bad_w in cfg._bad_pen_writer_indices:
                    pen_valid = pen_valid & (wl != bad_w)
                if pen_valid.sum() > 0:
                    loss_pen = F.cross_entropy(out["pen_logits"][pen_valid],
                                               pen_labels[pen_valid])
                    loss = loss + cfg.pen_head_lambda * loss_pen

            # Factorial contrastive loss
            if cfg.use_factorial and lambda_fac > 0 and "pen_emb" in out:
                pen_labels = batch["pen_label"].to(device).long()
                loss_fac, fac_stats = compute_factorial_contrastive_loss(
                    out["writer_emb"], out["pen_emb"], wl, pen_labels, cfg.fac_temperature
                )
                loss = loss + lambda_fac * loss_fac
                fac_stats_accum["writer_pos_pairs"] += fac_stats["writer_pos_pairs"]
                fac_stats_accum["pen_pos_pairs"] += fac_stats["pen_pos_pairs"]
                fac_stats_accum["orphans"] += fac_stats["orphans"]
                fac_stats_accum["fac_loss"] += loss_fac.item()
                fac_stats_accum["n_batches"] += 1

            # Outlier exposure
            if use_ood:
                try:
                    ood_batch = next(ood_iter)
                except StopIteration:
                    ood_iter = iter(ood_loader)
                    ood_batch = next(ood_iter)

                ood_imgs = ood_batch["image"].to(device)
                ood_masks = ood_batch["mask"].to(device)
                ood_polar = ood_batch.get("polar_strip")
                if ood_polar is not None:
                    ood_polar = ood_polar.to(device)
                ood_hc = ood_batch.get("handcrafted")
                if ood_hc is not None:
                    ood_hc = ood_hc.to(device)

                ood_out = model(ood_imgs, ood_masks, polar_strips=ood_polar,
                                hc_feats=ood_hc, eps=cfg.eps)
                # Target: uniform distribution over all writers (maximum uncertainty)
                num_classes = ood_out["writer_logits"].shape[1]
                uniform = torch.full_like(ood_out["writer_logits"], 1.0 / num_classes)
                loss_ood = F.kl_div(
                    F.log_softmax(ood_out["writer_logits"], dim=1),
                    uniform, reduction="batchmean"
                )
                loss = loss + cfg.ood_train_lambda * loss_ood
                tot_ood += loss_ood.item() * ood_imgs.size(0)

        scaled_loss = loss / cfg.grad_accum_steps
        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        is_accum_step = (step + 1) % cfg.grad_accum_steps == 0
        is_last_step = (step + 1) == num_steps
        if is_accum_step or is_last_step:
            if use_amp:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if ema and epoch >= cfg.ema_start_epoch: ema.update(model)
        B = images.size(0)
        tot_loss += loss.item() * B
        tot_arc += loss_arc.item() * B
        c_w += (out["writer_cosine"].argmax(1) == wl).sum().item()
        n += B
        uf += out["num_uniform_fallbacks"]

    result = {"loss": tot_loss/n, "loss_arc": tot_arc/n, "writer_acc": c_w/n,
              "uniform_fallbacks": uf, "arcface_margin": current_margin,
              "loss_ood": tot_ood / max(n, 1)}
    if cfg.use_factorial:
        result["fac_stats"] = fac_stats_accum
        result["lambda_fac"] = lambda_fac
    if hasattr(model, 'gate_alpha'):
        result["gate_alpha"] = model.gate_alpha.item()
    if hasattr(model, 'gate_hc'):
        result["gate_hc"] = model.gate_hc.item()
    return result


@torch.no_grad()
def extract_writer_embeddings(model, dataloader, device, eps=1e-8, use_amp=False):
    model.eval()
    wl_list, wc_list, we_list, lbl_list = [], [], [], []
    for batch in dataloader:
        polar_strips = batch.get("polar_strip")
        if polar_strips is not None:
            polar_strips = polar_strips.to(device)
        hc_feats = batch.get("handcrafted")
        if hc_feats is not None:
            hc_feats = hc_feats.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            o = model(batch["image"].to(device), batch["mask"].to(device),
                      polar_strips=polar_strips, hc_feats=hc_feats, eps=eps)
        wl_list.append(o["writer_logits"].cpu().numpy())
        wc_list.append(o["writer_cosine"].cpu().numpy())
        we_list.append(o["writer_emb"].cpu().numpy())
        if "writer_label" in batch:
            lbl_list.append(batch["writer_label"].numpy())
    result = {
        "writer_logits": np.concatenate(wl_list),
        "writer_cosine": np.concatenate(wc_list),
        "writer_emb":    np.concatenate(we_list),
    }
    if lbl_list: result["writer_labels"] = np.concatenate(lbl_list)
    return result


@torch.no_grad()
def eval_val_loss(model, val_loader, device, use_amp=False):
    
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in val_loader:
        polar_strips = batch.get("polar_strip")
        if polar_strips is not None:
            polar_strips = polar_strips.to(device)
        hc_feats = batch.get("handcrafted")
        if hc_feats is not None:
            hc_feats = hc_feats.to(device)
        labels = batch["writer_label"].to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            o = model(batch["image"].to(device), batch["mask"].to(device),
                      writer_labels=labels, polar_strips=polar_strips, hc_feats=hc_feats)
        loss = F.cross_entropy(o["writer_logits"], labels)
        preds = o["writer_cosine"].argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_n += labels.size(0)
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def run_tta_inference(model, df_test, data_dir, cfg, device):
    
    tta_grid = [False, True]  # original + hflip
    total_tta = len(tta_grid)
    print(f"  TTA grid: {total_tta} variants (original + hflip)")

    all_wl, all_wc, all_we = [], [], []
    with torch.no_grad():
        for vi, use_hflip in enumerate(tta_grid):
            test_ds = WriterTestDataset(
                df_test, data_dir, cfg.image_size, cfg.image_load_mode,
                apply_jitter=False,
                apply_hflip=use_hflip,
                use_polar=cfg.use_polar,
                polar_radial=cfg.polar_radial,
                polar_angular=cfg.polar_angular,
                use_handcrafted=cfg.use_handcrafted,
                hc_n_fourier=cfg.handcrafted_n_fourier,
                hc_n_radial_bins=cfg.handcrafted_n_radial_bins,
            )
            test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                                      num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                                      worker_init_fn=worker_init_fn)
            wl_t, wc_t, we_t = [], [], []
            for batch in test_loader:
                polar_strips = batch.get("polar_strip")
                if polar_strips is not None:
                    polar_strips = polar_strips.to(device)
                hc_feats = batch.get("handcrafted")
                if hc_feats is not None:
                    hc_feats = hc_feats.to(device)
                with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                    o = model(batch["image"].to(device), batch["mask"].to(device),
                              polar_strips=polar_strips, hc_feats=hc_feats, eps=cfg.eps)
                wl_t.append(o["writer_logits"].cpu().numpy())
                wc_t.append(o["writer_cosine"].cpu().numpy())
                we_t.append(o["writer_emb"].cpu().numpy())
            all_wl.append(np.concatenate(wl_t))
            all_wc.append(np.concatenate(wc_t))
            all_we.append(np.concatenate(we_t))
            print(f"    TTA {vi+1}/{total_tta} ({'hflip' if use_hflip else 'original'}) done")

    writer_logits = np.mean(all_wl, axis=0)
    writer_cosine = np.mean(all_wc, axis=0)
    writer_emb = np.mean(all_we, axis=0)
    writer_emb /= (np.linalg.norm(writer_emb, axis=1, keepdims=True) + 1e-8)
    return writer_logits, writer_cosine, writer_emb


# ======================================================================
# 7-10. Main: Data Exploration -> Training -> Calibration -> Submission
# ======================================================================

def main():
    
    parser = argparse.ArgumentParser(description="Writer ID training")
    parser.add_argument("--use-skeleton", action="store_true",
                        help="Use distance-transform stroke-width map as input "
                             "(skeleton-DT branch). Writes to outputs_skeleton/ "
                             "and checkpoints_skeleton/.")
    parser.add_argument("--plain-skeleton", action="store_true",
                        help="Use 1-px Zhang-Suen skeleton as input (winner's "
                             "ablation). Mutually exclusive with --use-skeleton. "
                             "Writes to outputs_plain_skel/ and checkpoints_plain_skel/.")
    args, _ = parser.parse_known_args()

    if args.use_skeleton and args.plain_skeleton:
        raise SystemExit("Cannot pass both --use-skeleton and --plain-skeleton.")

    if args.use_skeleton:
        _os.environ["INPUT_MODE"] = "skeleton_dt"
        globals()["_IS_SKELETON_MODE"] = True
    elif args.plain_skeleton:
        _os.environ["INPUT_MODE"] = "plain_skeleton"
        globals()["_IS_PLAIN_SKELETON_MODE"] = True

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg = Config()
    if cfg.use_amp and not torch.cuda.is_available():
        print("WARNING: AMP disabled (no CUDA)")
        cfg.use_amp = False
  
    if _IS_SKELETON_MODE:
        cfg.output_dir = Path("./outputs_skeleton")
        cfg.checkpoint_dir = Path("./checkpoints_skeleton")
        print(f"[INPUT_MODE] skeleton_dt -- output_dir={cfg.output_dir}, "
              f"checkpoint_dir={cfg.checkpoint_dir}")
    elif _IS_PLAIN_SKELETON_MODE:
        cfg.output_dir = Path("./outputs_plain_skel")
        cfg.checkpoint_dir = Path("./checkpoints_plain_skel")
        print(f"[INPUT_MODE] plain_skeleton -- output_dir={cfg.output_dir}, "
              f"checkpoint_dir={cfg.checkpoint_dir}")
    print(f"Config: {cfg.dinov2_variant}, epochs={cfg.epochs}, ood_alpha={cfg.ood_alpha}, amp={cfg.use_amp}")
    print(f"  LoRA: rank={cfg.lora_rank}, alpha={cfg.lora_alpha}, blocks={cfg.lora_last_n_blocks}")
    print(f"  Unfreeze norm+bias: last {cfg.unfreeze_norm_bias_blocks} blocks, lr={cfg.lr_unfreeze}")
    print(f"  Factorial: {cfg.use_factorial}, Polar: {cfg.use_polar}, Handcrafted: {cfg.use_handcrafted}")
    print(f"  VLAD: {cfg.use_vlad}, clusters={cfg.vlad_clusters}")
    print(f"  Ensemble seeds: {cfg.ensemble_seeds}")
    torch.backends.cudnn.benchmark = True
    set_seed(cfg.seed)

    # -- Step 0: Data Exploration -------------------------------------
    data_dir = Path(cfg.data_dir)
    df_train = pd.read_csv(data_dir / cfg.train_csv)
    df_add = pd.read_csv(data_dir / cfg.additional_train_csv)

    # Split additional into known and OOD
    df_add_known = df_add[df_add["writer_id"] != "-1"].copy()
    df_ood = df_add[df_add["writer_id"] == "-1"].copy()

    # Merge all known data
    df = pd.concat([df_train, df_add_known], ignore_index=True)

    # Fix known pen_id annotation errors for W41 and W50
    pen_error_writers = {"W41", "W50"}
    bad_mask = df["writer_id"].isin(pen_error_writers)
    n_bad = bad_mask.sum()
    if n_bad > 0:
        rng_pen = np.random.default_rng(cfg.seed)
        df.loc[bad_mask, "pen_id"] = rng_pen.choice([1, 2], size=n_bad)
        print(f"Pen-ID fix: reassigned {n_bad} samples from {pen_error_writers} to random 2-pen groups")

    print(f"Original train: {len(df_train)} samples")
    print(f"Additional known: {len(df_add_known)} samples")
    print(f"Total known: {len(df)} samples, {df['writer_id'].nunique()} writers")
    print(f"Real OOD samples: {len(df_ood)} (for threshold calibration)")
    print(f"Columns: {list(df.columns)}")

    writer2idx, idx2writer = build_label_maps(df)
    print(f"Writer map: {len(writer2idx)} writers -> 0..{len(writer2idx)-1}")

    # Precompute bad pen writer indices for pen head loss masking
    cfg._bad_pen_writer_indices = [writer2idx[w] for w in cfg.bad_pen_writers if w in writer2idx]
    print(f"Bad pen writers (masked from pen loss): {cfg.bad_pen_writers} -> indices {cfg._bad_pen_writer_indices}")

    # Factorial coverage audit
    cross_tab = pd.crosstab(df['writer_id'], df['pen_id'])
    eligible = (cross_tab > 0).sum(axis=1)
    print(f"\nFactorial coverage audit:")
    print(f"  Writers with >=2 pens: {(eligible >= 2).sum()} / {len(eligible)}")
    print(f"  Writers with <2 pens:  {(eligible < 2).sum()} / {len(eligible)}")
    print(f"  Samples per cell -- min: {cross_tab[cross_tab > 0].min().min()}, "
          f"median: {cross_tab[cross_tab > 0].stack().median():.0f}, "
          f"max: {cross_tab.max().max()}")
    print(f"  Missing combinations: {(cross_tab == 0).sum().sum()} / {cross_tab.size}")

    # -- Training Pipeline (5 splits) ---------------------------------
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits = create_writer_disjoint_splits(
        df, writer2idx, cfg.num_splits, cfg.held_out_writers_per_split,
        cfg.train_val_ratio, cfg.split_seed
    )

    all_ood_scores, all_ood_labels, all_thresholds, split_results = [], [], [], []

    for split in splits:
        sid = split["split_id"]
        print(f"\n{'='*60}")
        print(f"SPLIT {sid+1}/{cfg.num_splits} -- "
              f"{len(split['known_writers'])} known, {len(split['unknown_writers'])} unknown")
        print(f"  Train: {len(split['train_idx'])}, Val: {len(split['val_idx'])}, "
              f"Unknown: {len(split['unknown_idx'])}")
        print(f"{'='*60}")
        set_seed(cfg.seed + sid)

        known_sorted = sorted(split["known_writers"])
        s_w2i = {w: i for i, w in enumerate(known_sorted)}
        for w in split["unknown_writers"]: s_w2i[w] = -1

        kw = dict(num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                  worker_init_fn=worker_init_fn,
                  persistent_workers=False)
        hc_kw = dict(use_handcrafted=cfg.use_handcrafted,
                     hc_n_fourier=cfg.handcrafted_n_fourier,
                     hc_n_radial_bins=cfg.handcrafted_n_radial_bins)
        train_ds = WriterDataset(df, split["train_idx"], data_dir, s_w2i,
                                  cfg.image_size, cfg.image_load_mode, is_training=True,
                                  jitter_brightness=cfg.tta_brightness_range,
                                  jitter_contrast=cfg.tta_contrast_range,
                                  aug_hflip=cfg.aug_hflip,
                                  aug_rotation_deg=cfg.aug_rotation_deg,
                                  use_polar=cfg.use_polar,
                                  polar_radial=cfg.polar_radial,
                                  polar_angular=cfg.polar_angular,
                                  polar_circ_shift_aug=cfg.polar_circ_shift_aug,
                                  **hc_kw)
        val_ds   = WriterDataset(df, split["val_idx"],   data_dir, s_w2i,
                                  cfg.image_size, cfg.image_load_mode, is_training=False,
                                  use_polar=cfg.use_polar,
                                  polar_radial=cfg.polar_radial,
                                  polar_angular=cfg.polar_angular,
                                  **hc_kw)
        unk_ds   = WriterDataset(df, split["unknown_idx"], data_dir, s_w2i,
                                  cfg.image_size, cfg.image_load_mode, is_training=False,
                                  use_polar=cfg.use_polar,
                                  polar_radial=cfg.polar_radial,
                                  polar_angular=cfg.polar_angular,
                                  **hc_kw)
        # Real OOD dataset (from additional_train.csv, writer_id="-1")
        ood_ds = WriterTestDataset(df_ood, data_dir, cfg.image_size, cfg.image_load_mode,
                                    use_polar=cfg.use_polar,
                                    polar_radial=cfg.polar_radial,
                                    polar_angular=cfg.polar_angular,
                                    **hc_kw)

        # Use PenAxisBatchSampler when factorial is enabled
        if cfg.use_factorial:
            batch_sampler = PenAxisBatchSampler(
                df, split["train_idx"], s_w2i,
                num_writers=cfg.fac_num_writers,
                pens_per_writer=cfg.fac_pens_per_writer,
                samples_per_cell=cfg.fac_samples_per_cell,
                rng_seed=cfg.seed + sid,
            )
            train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, **kw)
        else:
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                      drop_last=True, **kw)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, **kw)
        unk_loader   = DataLoader(unk_ds,   batch_size=cfg.batch_size, shuffle=False, **kw)
        ood_loader   = DataLoader(ood_ds,   batch_size=cfg.batch_size, shuffle=False, **kw)
        # Shuffled OOD loader for outlier exposure during training
        ood_train_loader = DataLoader(ood_ds, batch_size=max(8, cfg.batch_size // 3),
                                       shuffle=True, drop_last=True, **kw) if len(df_ood) > 0 else None

        split_cfg = Config()
        split_cfg.__dict__.update(cfg.__dict__)
        split_cfg.num_known_writers = len(known_sorted)
        # Update bad pen writer indices for this split's writer mapping
        split_cfg._bad_pen_writer_indices = [s_w2i[w] for w in cfg.bad_pen_writers
                                              if w in s_w2i and s_w2i[w] >= 0]
        model, lora_params, head_params, unfreeze_params, polar_params, hc_params = build_writer_model(split_cfg)
        init_vlad_clusters(model, df, data_dir, cfg)  # K-means init BEFORE .to(device)
        model = model.to(device)
        model = maybe_compile(model, cfg)

        steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
        optimizer, scheduler = build_optimizer_scheduler(
            lora_params, head_params, cfg, steps_per_epoch,
            unfreeze_params=unfreeze_params if unfreeze_params else None,
            polar_params=polar_params if polar_params else None,
            hc_params=hc_params if hc_params else None)
        ema = EMAModel(model, cfg.ema_decay) if cfg.use_ema else None
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp) if cfg.use_amp else None

        save_dir = Path(cfg.checkpoint_dir) / f"split_{sid}"
        save_dir.mkdir(parents=True, exist_ok=True)
        best_auroc = 0.0
        best_val_acc = 0.0
        best_ema_auroc = 0.0
        best_ema_val_acc = 0.0
        ema_initialized = False

        for epoch in range(cfg.epochs):
            if ema and epoch == cfg.ema_start_epoch and not ema_initialized:
                ema.reinit_shadow(model)
                ema_initialized = True
                print(f"  EMA shadow re-initialized at epoch {epoch}")

            t0 = time.time()
            tm = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, cfg, ema, scaler,
                                 ood_loader=ood_train_loader)

            is_last = (epoch + 1) == cfg.epochs
            do_full_eval = (epoch % cfg.eval_every_n_epochs == 0) or is_last or (epoch >= cfg.epochs - 5)

            w_acc = 0.0
            auroc = 0.0
            ema_tag = ""

            if do_full_eval:
                torch.cuda.empty_cache()
                val_data = extract_writer_embeddings(model, val_loader, device, cfg.eps, cfg.use_amp)
                unk_data = extract_writer_embeddings(model, unk_loader, device, cfg.eps, cfg.use_amp)
                w_acc = (val_data["writer_cosine"].argmax(1) == val_data["writer_labels"]).mean()
                ks = extract_ood_scores(val_data["writer_logits"], val_data["writer_cosine"])
                us = extract_ood_scores(unk_data["writer_logits"], unk_data["writer_cosine"])
                ik = np.concatenate([np.ones(len(ks)), np.zeros(len(us))])
                mc = np.concatenate([ks[:, 0], us[:, 0]])
                auroc = roc_auc_score(ik, mc) if len(np.unique(ik)) > 1 else 0.0
                del val_data, unk_data  # free numpy arrays

                if auroc > best_auroc:
                    best_auroc = auroc
                    torch.save(model.state_dict(), save_dir / "best_model.pt")
                if w_acc > best_val_acc:
                    best_val_acc = w_acc

                if ema and epoch >= cfg.ema_start_epoch:
                    ema.apply_shadow(model)
                    ema_val = extract_writer_embeddings(model, val_loader, device, cfg.eps, cfg.use_amp)
                    ema_unk = extract_writer_embeddings(model, unk_loader, device, cfg.eps, cfg.use_amp)
                    ema_ks = extract_ood_scores(ema_val["writer_logits"], ema_val["writer_cosine"])
                    ema_us = extract_ood_scores(ema_unk["writer_logits"], ema_unk["writer_cosine"])
                    ema_ik = np.concatenate([np.ones(len(ema_ks)), np.zeros(len(ema_us))])
                    ema_mc = np.concatenate([ema_ks[:, 0], ema_us[:, 0]])
                    ema_auroc = roc_auc_score(ema_ik, ema_mc) if len(np.unique(ema_ik)) > 1 else 0.0
                    ema_w_acc = (ema_val["writer_cosine"].argmax(1) == ema_val["writer_labels"]).mean()
                    if ema_auroc > best_ema_auroc:
                        best_ema_auroc = ema_auroc
                        torch.save(model.state_dict(), save_dir / "best_ema.pt")
                    if ema_w_acc > best_ema_val_acc:
                        best_ema_val_acc = ema_w_acc
                    ema_tag = f" ema_AUC={ema_auroc:.3f} ema_W={ema_w_acc:.3f}"
                    ema.restore(model)
                    del ema_val, ema_unk  # free numpy arrays

            elapsed = time.time() - t0
            uf_str = f" WARNING:uf={tm['uniform_fallbacks']}" if tm["uniform_fallbacks"] > 0 else ""
            gate_str = f" gate={tm['gate_alpha']:.3f}" if "gate_alpha" in tm else ""
            if "gate_hc" in tm:
                gate_str += f" gate_hc={tm['gate_hc']:.3f}"
            margin_str = f" m={tm['arcface_margin']:.3f}" if "arcface_margin" in tm else ""
            ood_str = f" L_ood={tm['loss_ood']:.3f}" if tm.get('loss_ood', 0) > 0 else ""
            eval_str = f" | val_W={w_acc:.3f} AUC={auroc:.3f}{ema_tag}" if do_full_eval else ""
            print(f"  E{epoch+1:02d} ({elapsed:.0f}s) L={tm['loss']:.4f} L_arc={tm['loss_arc']:.4f} "
                  f"train_W={tm['writer_acc']:.3f}{eval_str}{gate_str}{margin_str}{ood_str}{uf_str}")
            if cfg.use_factorial and "fac_stats" in tm:
                log_batch_pair_stats(tm["fac_stats"], epoch, tm.get("lambda_fac", 0.0))

        if (save_dir / "best_ema.pt").exists() and best_ema_auroc >= best_auroc:
            bp = save_dir / "best_ema.pt"
            print(f"  Loading best EMA checkpoint (val_acc={best_ema_val_acc:.4f}, AUROC={best_ema_auroc:.4f})")
        else:
            bp = save_dir / "best_model.pt"
            print(f"  Loading best model checkpoint (val_acc={best_val_acc:.4f}, AUROC={best_auroc:.4f})")
        model.load_state_dict(torch.load(bp, map_location=device, weights_only=False))
        torch.cuda.empty_cache()
        val_data = extract_writer_embeddings(model, val_loader, device, cfg.eps, cfg.use_amp)
        unk_data = extract_writer_embeddings(model, unk_loader, device, cfg.eps, cfg.use_amp)
        

        ood_data = extract_writer_embeddings(model, ood_loader, device, cfg.eps, cfg.use_amp)
        ks = extract_ood_scores(val_data["writer_logits"], val_data["writer_cosine"])
        us_held = extract_ood_scores(unk_data["writer_logits"], unk_data["writer_cosine"])
        us_real = extract_ood_scores(ood_data["writer_logits"], ood_data["writer_cosine"])
        # Combine held-out + real OOD as unknown class
        us = np.vstack([us_held, us_real])
        combined = np.vstack([ks, us])
        ik = np.concatenate([np.ones(len(ks)), np.zeros(len(us))])
        all_ood_scores.append(combined)
        all_ood_labels.append(ik)
        tau = find_optimal_threshold(combined[:, 0], ik, cfg.ood_alpha)
        all_thresholds.append(tau)
        auroc_final = roc_auc_score(ik, combined[:, 0])
        split_results.append({"split": sid, "auroc": auroc_final, "tau": tau})
        print(f"\n  Split {sid+1} final -- AUROC: {auroc_final:.4f}")
        print(f"  At tau={tau:.4f}: known_acc={(combined[:len(ks),0]>=tau).mean():.3f}, "
              f"unknown_recall={(combined[len(ks):,0]<tau).mean():.3f}")
        print(f"  OOD sources: held-out={len(us_held)}, real_ood={len(us_real)}")

        del model, optimizer, scheduler, ema, scaler
        torch.cuda.empty_cache()

    # -- Step 9: OOD Calibration --
    if len(all_ood_scores) > 0:
        calibrator = fit_ood_calibrator(all_ood_scores, all_ood_labels)

        cal_thresholds = []
        for scores, labels in zip(all_ood_scores, all_ood_labels):
            p_cal = calibrator.predict_proba(scores)[:, 1]
            cal_thresholds.append(find_optimal_threshold(p_cal, labels, cfg.ood_alpha))
        median_tau = float(np.median(cal_thresholds))

        aurocs = [r["auroc"] for r in split_results]
        print(f"\nAUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
        print(f"Per-split: {[f'{a:.4f}' for a in aurocs]}")
        print(f"Calibrated tau: {median_tau:.4f}")
        print(f"Go criterion (AUROC>0.80 on >=4/5): "
              f"{'PASS' if sum(a > 0.80 for a in aurocs) >= 4 else 'FAIL'}")

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(cfg.output_dir) / "calibrator.pkl", "wb") as f:
            pickle.dump(calibrator, f)
        np.savez(Path(cfg.output_dir) / "calibration.npz",
                 aurocs=np.array(aurocs), median_tau=median_tau)
    else:
        # Skip CV: use threshold from split 1 (AUROC 0.889, tau=0.379)
        median_tau = 0.3794
        print(f"\nSkipped CV. Using hardcoded tau={median_tau:.4f} from split 1 (AUROC=0.889)")
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # -- Step 10: Ensemble Submission --
    final_cfg = Config()
    final_cfg.__dict__.update(cfg.__dict__)
    final_cfg.num_known_writers = len(writer2idx)

    df_test = pd.read_csv(data_dir / cfg.test_csv)
    print(f"\nTest size: {len(df_test)}")
    s_idx2writer = {i: w for w, i in writer2idx.items()}

    all_idx = df.index.tolist()
    ensemble_logits, ensemble_cosine = [], []

    for seed_i, seed_val in enumerate(cfg.ensemble_seeds):
        print(f"\n{'='*60}")
        print(f"ENSEMBLE SEED {seed_i+1}/{len(cfg.ensemble_seeds)} -- seed={seed_val}")
        print(f"{'='*60}")
        set_seed(seed_val)

        # Early stopping: stratified 90/10 split for validation
        val_frac = cfg.early_stop_val_fraction
        if val_frac > 0:
            writer_labels_all = np.array([writer2idx[df.iloc[i]["writer_id"]] for i in all_idx])
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed_val)
            tr_pos, va_pos = next(sss.split(all_idx, writer_labels_all))
            train_idx = [all_idx[p] for p in tr_pos]
            val_idx = [all_idx[p] for p in va_pos]
            print(f"  Early stopping: train={len(train_idx)}, val={len(val_idx)} "
                  f"({val_frac:.0%} held out), patience={cfg.early_stop_patience}")
        else:
            train_idx = all_idx
            val_idx = []

        hc_kw_f = dict(use_handcrafted=cfg.use_handcrafted,
                       hc_n_fourier=cfg.handcrafted_n_fourier,
                       hc_n_radial_bins=cfg.handcrafted_n_radial_bins)
        final_ds = WriterDataset(df, train_idx, data_dir, writer2idx,
                                  cfg.image_size, cfg.image_load_mode, is_training=True,
                                  jitter_brightness=cfg.tta_brightness_range,
                                  jitter_contrast=cfg.tta_contrast_range,
                                  aug_hflip=cfg.aug_hflip,
                                  aug_rotation_deg=cfg.aug_rotation_deg,
                                  use_polar=cfg.use_polar,
                                  polar_radial=cfg.polar_radial,
                                  polar_angular=cfg.polar_angular,
                                  polar_circ_shift_aug=cfg.polar_circ_shift_aug,
                                  **hc_kw_f)

        kw_final = dict(num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                        worker_init_fn=worker_init_fn)
        if cfg.use_factorial:
            final_batch_sampler = PenAxisBatchSampler(
                df, train_idx, writer2idx,
                num_writers=cfg.fac_num_writers,
                pens_per_writer=cfg.fac_pens_per_writer,
                samples_per_cell=cfg.fac_samples_per_cell,
                rng_seed=seed_val,
            )
            final_loader = DataLoader(final_ds, batch_sampler=final_batch_sampler, **kw_final)
        else:
            final_loader = DataLoader(final_ds, batch_size=cfg.batch_size, shuffle=True,
                                       drop_last=True, **kw_final)

        # OOD training loader for ensemble phase
        ood_ds_f = WriterTestDataset(df_ood, data_dir, cfg.image_size, cfg.image_load_mode,
                                      use_polar=cfg.use_polar, polar_radial=cfg.polar_radial,
                                      polar_angular=cfg.polar_angular,
                                      **hc_kw_f) if len(df_ood) > 0 else None
        ood_train_loader_f = DataLoader(ood_ds_f, batch_size=max(8, cfg.batch_size // 3),
                                         shuffle=True, drop_last=True,
                                         **kw_final) if ood_ds_f is not None else None

        # Build validation loader for early stopping
        val_loader_es = None
        if val_idx:
            val_ds_es = WriterDataset(df, val_idx, data_dir, writer2idx,
                                       cfg.image_size, cfg.image_load_mode, is_training=False,
                                       use_polar=cfg.use_polar,
                                       polar_radial=cfg.polar_radial,
                                       polar_angular=cfg.polar_angular,
                                       **hc_kw_f)
            val_loader_es = DataLoader(val_ds_es, batch_size=cfg.batch_size, shuffle=False,
                                        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                                        worker_init_fn=worker_init_fn)

        model, lora_params_f, head_params_f, unfreeze_params_f, polar_params_f, hc_params_f = build_writer_model(final_cfg)
        init_vlad_clusters(model, df, data_dir, cfg)  
        model = model.to(device)
        model = maybe_compile(model, cfg)
        steps_f = math.ceil(len(final_loader) / cfg.grad_accum_steps)
        optimizer_f, scheduler_f = build_optimizer_scheduler(
            lora_params_f, head_params_f, final_cfg, steps_f,
            unfreeze_params=unfreeze_params_f if unfreeze_params_f else None,
            polar_params=polar_params_f if polar_params_f else None,
            hc_params=hc_params_f if hc_params_f else None)
        ema_f = EMAModel(model, cfg.ema_decay)
        scaler_f = torch.amp.GradScaler("cuda", enabled=cfg.use_amp) if cfg.use_amp else None

        best_val_loss = float("inf")
        patience_counter = 0
        best_ema_state = None

        ema_f_initialized = False
        for epoch in range(cfg.epochs):
            if epoch == cfg.ema_start_epoch and not ema_f_initialized:
                ema_f.reinit_shadow(model)
                ema_f_initialized = True
                print(f"  EMA shadow re-initialized at epoch {epoch}")
            tm = train_one_epoch(model, final_loader, optimizer_f, scheduler_f, device,
                                 epoch, final_cfg, ema_f, scaler_f,
                                 ood_loader=ood_train_loader_f)
            gate_str = f" gate={tm['gate_alpha']:.3f}" if "gate_alpha" in tm else ""
            if "gate_hc" in tm:
                gate_str += f" gate_hc={tm['gate_hc']:.3f}"
            margin_str = f" m={tm['arcface_margin']:.3f}" if "arcface_margin" in tm else ""
            log_line = (f"  E{epoch+1:02d} L={tm['loss']:.4f} L_arc={tm['loss_arc']:.4f} "
                        f"W={tm['writer_acc']:.3f}{gate_str}{margin_str}")
            if cfg.use_factorial and "fac_stats" in tm:
                log_batch_pair_stats(tm["fac_stats"], epoch, tm.get("lambda_fac", 0.0))

            
            if val_loader_es is not None and ema_f_initialized:
                ema_f.apply_shadow(model)
                val_loss, val_acc = eval_val_loss(model, val_loader_es, device, cfg.use_amp)
                ema_f.restore(model)

                log_line += f" | val_L={val_loss:.4f} val_acc={val_acc:.3f}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_ema_state = {k: v.clone() for k, v in ema_f.shadow.items()}
                    log_line += " *"
                else:
                    patience_counter += 1
                    log_line += f" ({patience_counter}/{cfg.early_stop_patience})"

            print(log_line)

            if patience_counter >= cfg.early_stop_patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(best val_loss={best_val_loss:.4f})")
                break

        if best_ema_state is not None:
            ema_f.shadow = best_ema_state
            ema_f.apply_shadow(model)
        elif ema_f_initialized:
            ema_f.apply_shadow(model)
        model.eval()

        # Save seed checkpoint
        ckpt_path = Path(cfg.checkpoint_dir) / f"final_seed_{seed_val}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

        torch.cuda.empty_cache()
        wl_seed, wc_seed, we_seed = run_tta_inference(model, df_test, data_dir, cfg, device)
        ensemble_logits.append(wl_seed)
        ensemble_cosine.append(wc_seed)

        # Extract train embeddings
        print("  Extracting train embeddings...")
        train_emb_ds = WriterDataset(df, all_idx, data_dir, writer2idx,
                                      cfg.image_size, cfg.image_load_mode, is_training=False,
                                      use_polar=cfg.use_polar,
                                      polar_radial=cfg.polar_radial,
                                      polar_angular=cfg.polar_angular,
                                      **hc_kw_f)
        train_emb_loader = DataLoader(train_emb_ds, batch_size=cfg.batch_size, shuffle=False,
                                       num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                                       worker_init_fn=worker_init_fn)
        with torch.no_grad():
            train_data = extract_writer_embeddings(model, train_emb_loader, device, cfg.eps, cfg.use_amp)

        # Save everything for this seed
        emb_path = Path(cfg.output_dir) / f"embeddings_seed_{seed_val}.npz"
        np.savez(emb_path,
                 train_emb=train_data["writer_emb"],
                 train_labels=train_data["writer_labels"],
                 train_logits=train_data["writer_logits"],
                 train_cosine=train_data["writer_cosine"],
                 test_emb=we_seed,
                 test_logits=wl_seed,
                 test_cosine=wc_seed)
        print(f"  Saved embeddings: {emb_path}")
        print(f"    train: {train_data['writer_emb'].shape}, test: {we_seed.shape}")

        
        del model, optimizer_f, scheduler_f, ema_f, scaler_f
        del lora_params_f, head_params_f, unfreeze_params_f, polar_params_f, hc_params_f
        del train_data, we_seed, best_ema_state
        torch.cuda.empty_cache()

    # Average logits/cosine across ensemble seeds
    writer_logits = np.mean(ensemble_logits, axis=0)
    writer_cosine = np.mean(ensemble_cosine, axis=0)
    print(f"\nEnsemble: averaged {len(cfg.ensemble_seeds)} seeds")

    # Save averaged test outputs (so threshold sweeps never need re-extraction)
    tta_path = Path(cfg.output_dir) / "tta_outputs.npz"
    np.savez(tta_path,
             writer_logits=writer_logits,
             writer_cosine=writer_cosine,
             seeds=np.array(cfg.ensemble_seeds))
    print(f"Saved averaged test outputs: {tta_path}")

    preds = make_writer_predictions(writer_logits, writer_cosine, calibrator, median_tau)
    w_orig = [s_idx2writer.get(int(wp), "-1") if wp != -1 else "-1" for wp in preds["writer_preds"]]

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"image_id": df_test["image_id"], "writer_id": w_orig})
    submission.to_csv(Path(cfg.output_dir) / "submission_writer.csv", index=False)
    print(submission.head())
    n_unknown = (submission['writer_id'] == "-1").sum()
    print(f"Unknown (-1): {n_unknown} / {len(submission)} "
          f"({100*n_unknown/len(submission):.1f}%)")

    # Also save an all-known submission (no OOD rejection) for debugging
    writer_raw = writer_cosine.argmax(axis=1)
    w_all_known = [s_idx2writer[int(wp)] for wp in writer_raw]
    sub_all_known = pd.DataFrame({"image_id": df_test["image_id"], "writer_id": w_all_known})
    sub_all_known.to_csv(Path(cfg.output_dir) / "submission_all_known.csv", index=False)
    print(f"\nAll-known submission saved (no -1 predictions) for debugging")

    # Percentile-based unknown submissions for threshold tuning
    scores = extract_ood_scores(writer_logits, writer_cosine)
    max_cos = scores[:, 0]
    print("\nPercentile-based submissions:")
    for pct in [2, 5, 10, 30, 50, 70]:
        threshold = np.percentile(max_cos, pct)
        w_pct = []
        for i, wp in enumerate(writer_raw):
            if max_cos[i] < threshold:
                w_pct.append("-1")
            else:
                w_pct.append(s_idx2writer[int(wp)])
        sub_pct = pd.DataFrame({"image_id": df_test["image_id"], "writer_id": w_pct})
        fname = f"submission_unk{pct}pct.csv"
        sub_pct.to_csv(Path(cfg.output_dir) / fname, index=False)
        n = sum(1 for x in w_pct if x == "-1")
        print(f"  {fname}: {n}/{len(sub_pct)} unknown ({100*n/len(sub_pct):.1f}%)")


if __name__ == "__main__":
    main()
