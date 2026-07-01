"""
Patch extraction for whole-page writer-identification datasets
(CVL, HWI).

Two modes:
- Random patches: used during training as augmentation
  (different ink regions sampled each epoch).
- Fixed/deterministic patches: used during val/test for stable scoring
  and reproducibility (seeded by image_id).

A patch is accepted if its ink coverage >= min_ink_frac.
Falls back to center crops on pages with no detectable ink.

Skeleton-DT branch:
- Toggled via `INPUT_MODE=skeleton_dt` env var (set by train_writerid.py's
  --use-skeleton flag before DataLoader workers spawn).
- Each 224x224 RGB patch is replaced by a distance-transform map encoding
  stroke half-width: 3 tiled channels carrying topology AND thickness.
"""

import os as _os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DINO_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATCH_SIZE = 224

_IS_SKELETON_MODE = _os.environ.get("INPUT_MODE") == "skeleton_dt"


def compute_distance_transform_image(image):
    #Replace RGB pixels with stroke half-width via cv2.distanceTransform.
    
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    # Otsu, INVERTED so ink=255 and background=0 (required by cv2.distanceTransform)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dt_max = float(dt.max())
    if dt_max > 0:
        dt_uint8 = (dt / dt_max * 255.0).astype(np.uint8)
    else:
        dt_uint8 = np.zeros_like(gray, dtype=np.uint8)
    return np.stack([dt_uint8, dt_uint8, dt_uint8], axis=-1)


def load_image(path):
    
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    if arr[:, :, 0].mean() < 80.0:
        arr = 255 - arr
    return arr


def ink_mask(img_rgb, block_size=51, c=10):
    """Adaptive-Gaussian binarization. 1 = ink, 0 = background.

    Block size must be odd. Larger c -> more conservative (less ink detected).
    """
    if block_size % 2 == 0:
        block_size += 1
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c,
    )
    return (binary > 0).astype(np.uint8)


def _ensure_min_size(img_rgb, mask, patch_size):
    """If page is smaller than patch_size in either dim, upscale proportionally."""
    h, w = mask.shape
    if h >= patch_size and w >= patch_size:
        return img_rgb, mask
    scale = max(patch_size / h, patch_size / w) * 1.1
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(
        (mask * 255).astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
    )
    return img_resized, (mask_resized > 127).astype(np.uint8)


def _crop_resize(img_rgb, y0, x0, view_size, patch_size):
    """Crop view_size window then resize to patch_size if they differ."""
    crop = img_rgb[y0:y0 + view_size, x0:x0 + view_size]
    if view_size != patch_size:
        crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    return crop


def sample_patches(
    img_rgb,
    mask,
    n_patches,
    patch_size=PATCH_SIZE,
    view_size=None,
    min_ink_frac=0.05,
    seed=None,
    max_attempts_per_patch=8,
):
    
    if view_size is None:
        view_size = patch_size

    img_rgb, mask = _ensure_min_size(img_rgb, mask, view_size)
    h, w = mask.shape
    rng = np.random.default_rng(seed)
    half = view_size // 2

    ink_ys, ink_xs = np.where(mask > 0)
    if len(ink_ys) == 0:
        # No ink: return n_patches center crops
        y0 = max(0, (h - view_size) // 2)
        x0 = max(0, (w - view_size) // 2)
        return np.stack([_crop_resize(img_rgb, y0, x0, view_size, patch_size)] * n_patches, axis=0)

    patches = []
    max_attempts = n_patches * max_attempts_per_patch
    for _ in range(max_attempts):
        if len(patches) >= n_patches:
            break
        idx = rng.integers(0, len(ink_ys))
        cy, cx = int(ink_ys[idx]), int(ink_xs[idx])
        y0 = max(0, min(cy - half, h - view_size))
        x0 = max(0, min(cx - half, w - view_size))
        patch_mask = mask[y0:y0 + view_size, x0:x0 + view_size]
        if patch_mask.mean() >= min_ink_frac:
            patches.append(_crop_resize(img_rgb, y0, x0, view_size, patch_size))

    # Fallback: relax ink threshold if we still don't have enough
    relax = min_ink_frac * 0.5
    fallback_attempts = n_patches * max_attempts_per_patch * 2
    for _ in range(fallback_attempts):
        if len(patches) >= n_patches:
            break
        idx = rng.integers(0, len(ink_ys))
        cy, cx = int(ink_ys[idx]), int(ink_xs[idx])
        y0 = max(0, min(cy - half, h - view_size))
        x0 = max(0, min(cx - half, w - view_size))
        patch_mask = mask[y0:y0 + view_size, x0:x0 + view_size]
        if patch_mask.mean() >= relax or len(patches) == 0:
            patches.append(_crop_resize(img_rgb, y0, x0, view_size, patch_size))

    
    if len(patches) < n_patches:
        if len(patches) == 0:
            y0 = max(0, (h - view_size) // 2)
            x0 = max(0, (w - view_size) // 2)
            patches.append(_crop_resize(img_rgb, y0, x0, view_size, patch_size))
        while len(patches) < n_patches:
            patches.append(patches[0])

    return np.stack(patches[:n_patches], axis=0)


def is_binarized_image(img_rgb, max_unique_levels=8):
    """Heuristic: treat as binary if grayscale has <=max_unique_levels distinct values."""
    gray = img_rgb[:, :, 0]
    return len(np.unique(gray)) <= max_unique_levels


def augment_patch(
    patch_rgb,
    rng,
    max_rotation_deg=5.0,
    brightness=0.15,
    contrast=0.15,
    binary_mode=False,
):
    
    h, w = patch_rgb.shape[:2]

    if max_rotation_deg > 0:
        angle = float(rng.uniform(-max_rotation_deg, max_rotation_deg))
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        interp = cv2.INTER_NEAREST if binary_mode else cv2.INTER_LINEAR
        patch_rgb = cv2.warpAffine(
            patch_rgb, M, (w, h),
            flags=interp,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),  # white background fill
        )
        if binary_mode:
            patch_rgb = np.where(patch_rgb >= 128, 255, 0).astype(np.uint8)

    if not binary_mode:
        if brightness > 0:
            f = 1.0 + float(rng.uniform(-brightness, brightness))
            patch_rgb = np.clip(patch_rgb.astype(np.float32) * f, 0, 255).astype(np.uint8)
        if contrast > 0:
            f = 1.0 + float(rng.uniform(-contrast, contrast))
            mean = float(patch_rgb.mean())
            patch_rgb = np.clip(
                (patch_rgb.astype(np.float32) - mean) * f + mean, 0, 255
            ).astype(np.uint8)

    return patch_rgb


def patches_to_input_tensors(patches_uint8):
    
    # Compute masks on ORIGINAL RGB patches (before any DT transform)
    masks = []
    for p in patches_uint8:
        m = ink_mask(p)
        masks.append(m)
    masks = torch.from_numpy(np.stack(masks, axis=0).astype(np.float32))

    # Optionally replace RGB with distance-transform map
    if _IS_SKELETON_MODE:
        patches_uint8 = np.stack(
            [compute_distance_transform_image(p) for p in patches_uint8], axis=0
        )

    arr = patches_uint8.astype(np.float32) / 255.0
    arr = (arr - DINO_MEAN) / DINO_STD
    images = torch.from_numpy(arr.transpose(0, 3, 1, 2)).contiguous().float()

    return images, masks


def extract_patches_for_image(
    image_path,
    n_patches,
    seed=None,
    min_ink_frac=0.05,
    view_size=None,
):
    
    img = load_image(image_path)
    mask = ink_mask(img)
    patches = sample_patches(
        img, mask, n_patches,
        seed=seed, min_ink_frac=min_ink_frac, view_size=view_size,
    )
    return patches_to_input_tensors(patches)


if __name__ == "__main__":
    
    import sys
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cvl/cvl/0001-1-cropped.tif")
    print(f"Testing on {p}")
    imgs, msks = extract_patches_for_image(p, n_patches=4, seed=0)
    print(f"  Patches: {imgs.shape} (dtype {imgs.dtype})")
    print(f"  Masks:   {msks.shape} (mean ink frac = {msks.mean().item():.3f})")
    # Save preview
    pil = patches_to_input_tensors  # silence linter
    print("  OK")
