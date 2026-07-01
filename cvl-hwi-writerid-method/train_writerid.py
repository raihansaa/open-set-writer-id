"""
Open-set writer-identification training.

Pipeline:
    DINOv2 ViT-B/14 frozen + LoRA -> NetVLAD -> ArcFace.
    No polar encoder, no factorial contrastive, no pen head, no OE.
              
Usage:
    # Color (default)
    python train_writerid.py \
        --data-dir cvl \
        --splits-dir cvl_splits \
        --out-dir runs/cvl_seed42 \
        --epochs 40 --batch-size 32 --seed 42

    # Skeleton-DT branch (stroke half-width via distance transform)
    python train_writerid.py \
        --data-dir cvl \
        --splits-dir cvl_splits \
        --out-dir runs/cvl_skeleton_seed42 \
        --use-skeleton \
        --epochs 40 --batch-size 32 --seed 42
"""

import argparse
import copy
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Local
import patches as patch_lib


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    data_dir: str = "cvl"
    splits_dir: str = "cvl/cvl_splits"
    out_dir: str = "runs/cvl_seed42"
    seed: int = 42

    # Backbone + LoRA
    dinov2_variant: str = "dinov2_vitb14_reg"
    dinov2_repo: str = "facebookresearch/dinov2"
    dinov2_dim: int = 768
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_last_n_blocks: int = 6

    # Head / VLAD
    vlad_clusters: int = 64
    aggregator: str = "netvlad"   # NetVLAD with learned cluster centers
    projection_dim: int = 512
    writer_embed_dim: int = 512
    arcface_scale: float = 30.0
    arcface_margin: float = 0.3

    # Training
    epochs: int = 40
    batch_size: int = 32
    lr_lora: float = 5e-5
    lr_heads: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    label_smoothing: float = 0.05
    use_amp: bool = True
    grad_checkpointing: bool = True

    # EMA
    ema_decay: float = 0.999
    ema_start_epoch: int = 5

    # Patch extraction
    n_patches_eval: int = 64
    min_ink_frac: float = 0.05
    patch_zoom: float = 1.0  # >1.0 = zoom in (sample smaller crop from page, upscale to 224)

    # Train-time augmentation
    aug_enabled: bool = True
    aug_rotation_deg: float = 5.0
    aug_brightness: float = 0.15
    aug_contrast: float = 0.15
    aug_binary_mode: str = "auto"   # "auto", "yes", "no"

    # PK-style writer-balanced batching (HWI-friendly few-shot metric learning)
    pk_sampler: bool = False
    pk_pages_per_writer: int = 2   # K in P×K batches
    n_crops: int = 1               # crops per page per __getitem__ (>=2 helps HWI)

    # SupCon auxiliary loss
    lambda_supcon: float = 0.0     # 0 = disabled; 0.05–0.2 typical
    supcon_tau: float = 0.07

    # Checkpoint selection metric: "prototype_auroc" (default) | "centroid_known_top1"
    checkpoint_metric: str = "prototype_auroc"

    # Page-level aggregation: "mean" (default, unchanged) | "attention" (learned MIL)
    page_agg: str = "mean"
    # Auxiliary per-crop ArcFace loss weight (only used when page_agg=="attention")
    page_aux_weight: float = 0.1
    # K values to randomly subsample per batch when page_agg=="attention"
    page_train_k_choices: tuple = (4, 5, 6, 8)

    # Multi-prototype ArcFace: K prototypes per writer to capture script/era diversity
    multi_proto_k: int = 1

    # Full-gallery hard-negative mining: triplet term using closest non-true prototype
    hard_neg_weight: float = 0.0   # 0 = disabled
    hard_neg_margin: float = 0.1   # required gap between true and hardest negative cosine

    # DataLoader
    num_workers: int = 2
    pin_memory: bool = True


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────
class TrainPatchDataset(Dataset):
    

    def __init__(self, df, data_dir, writer2idx, min_ink_frac=0.05,
                 aug_enabled=False, aug_rotation_deg=0.0,
                 aug_brightness=0.0, aug_contrast=0.0,
                 binary_mode=False, view_size=None, n_crops=1):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.writer2idx = writer2idx
        self.min_ink_frac = min_ink_frac
        self.aug_enabled = aug_enabled
        self.aug_rotation_deg = aug_rotation_deg
        self.aug_brightness = aug_brightness
        self.aug_contrast = aug_contrast
        self.binary_mode = binary_mode
        self.view_size = view_size
        self.n_crops = max(1, int(n_crops))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.data_dir / row["image_path"]
        img = patch_lib.load_image(path)
        mask = patch_lib.ink_mask(img)
        patch_arr = patch_lib.sample_patches(
            img, mask, n_patches=self.n_crops, seed=None,
            min_ink_frac=self.min_ink_frac, view_size=self.view_size,
        )
        if self.aug_enabled:
            rng = np.random.default_rng()  # fresh randomness per call
            for k in range(len(patch_arr)):
                patch_arr[k] = patch_lib.augment_patch(
                    patch_arr[k], rng,
                    max_rotation_deg=self.aug_rotation_deg,
                    brightness=self.aug_brightness,
                    contrast=self.aug_contrast,
                    binary_mode=self.binary_mode,
                )
        images, masks = patch_lib.patches_to_input_tensors(patch_arr)
        label = int(self.writer2idx[row["writer_id"]])
        if self.n_crops == 1:
            return {"image": images[0], "mask": masks[0], "label": label}
        return {"image": images, "mask": masks, "label": label}


class EvalPagePatchDataset(Dataset):

    def __init__(self, df, data_dir, n_patches=32, min_ink_frac=0.05, view_size=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.n_patches = n_patches
        self.min_ink_frac = min_ink_frac
        self.view_size = view_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.data_dir / row["image_path"]
        seed = int(row["image_id"])
        images, masks = patch_lib.extract_patches_for_image(
            path, n_patches=self.n_patches, seed=seed,
            min_ink_frac=self.min_ink_frac, view_size=self.view_size,
        )
        return {
            "image": images,                # (N, 3, 224, 224)
            "mask": masks,                  # (N, 224, 224)
            "writer_id": str(row["writer_id"]),
            "image_id": int(row["image_id"]),
        }


def eval_collate(batch):
    """Collate a list of variable-size eval samples into stacked tensors."""
    
    images = torch.stack([b["image"] for b in batch], dim=0)  # (B, N, 3, H, W)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    writer_ids = [b["writer_id"] for b in batch]
    image_ids = [b["image_id"] for b in batch]
    return {"image": images, "mask": masks, "writer_id": writer_ids, "image_id": image_ids}


def train_multicrop_collate(batch):
    
    images, masks, labels = [], [], []
    for b in batch:
        im = b["image"]
        if im.ndim == 3:  # single crop
            images.append(im.unsqueeze(0))
            masks.append(b["mask"].unsqueeze(0))
            labels.append(b["label"])
        else:  # multi-crop (K, 3, 224, 224)
            images.append(im)
            masks.append(b["mask"])
            labels.extend([b["label"]] * im.shape[0])
    return {
        "image": torch.cat(images, dim=0),
        "mask": torch.cat(masks, dim=0),
        "label": torch.tensor(labels, dtype=torch.long),
    }


def train_pageagg_collate(batch):
    
    images_list, masks_list, labels = [], [], []
    for b in batch:
        im = b["image"]
        mk = b["mask"]
        if im.ndim == 3:   # single crop -> add K dim
            im = im.unsqueeze(0)
            mk = mk.unsqueeze(0)
        images_list.append(im)   # (K_i, 3, 224, 224)
        masks_list.append(mk)    # (K_i, 224, 224)
        labels.append(b["label"])

    # Trim all to the minimum K so we can stack
    min_k = min(im.shape[0] for im in images_list)
    images_list = [im[:min_k] for im in images_list]
    masks_list = [mk[:min_k] for mk in masks_list]

    return {
        "image": torch.stack(images_list, dim=0),   # (B, K, 3, 224, 224)
        "mask": torch.stack(masks_list, dim=0),     # (B, K, 224, 224)
        "label": torch.tensor(labels, dtype=torch.long),  # (B,)
    }


class PKBatchSampler(Sampler):

    def __init__(self, labels, writers_per_batch, pages_per_writer=2,
                 batches_per_epoch=None, seed=42):
        self.labels = np.asarray(labels)
        self.writers_per_batch = int(writers_per_batch)
        self.pages_per_writer = int(pages_per_writer)
        self.seed = int(seed)
        self.epoch = 0

        # writer_id -> list of dataset indices
        self.writer_to_idx = {}
        for i, w in enumerate(self.labels):
            self.writer_to_idx.setdefault(int(w), []).append(i)
        self.writer_ids = sorted(self.writer_to_idx.keys())

        if batches_per_epoch is None:
            batches_per_epoch = max(
                1, len(self.labels) // (self.writers_per_batch * self.pages_per_writer)
            )
        self.batches_per_epoch = int(batches_per_epoch)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        writer_pool = list(self.writer_ids)
        rng.shuffle(writer_pool)
        cursor = 0
        for _ in range(self.batches_per_epoch):
            # Refill the writer pool if we've used everyone this epoch
            if cursor + self.writers_per_batch > len(writer_pool):
                rng.shuffle(writer_pool)
                cursor = 0
            chosen = writer_pool[cursor:cursor + self.writers_per_batch]
            cursor += self.writers_per_batch
            batch = []
            for w in chosen:
                pool = self.writer_to_idx[w]
                if len(pool) >= self.pages_per_writer:
                    picks = rng.choice(pool, size=self.pages_per_writer, replace=False)
                else:
                    picks = rng.choice(pool, size=self.pages_per_writer, replace=True)
                batch.extend(int(p) for p in picks)
            yield batch


# ──────────────────────────────────────────────────────────────────────
# Architecture (copied/stripped from train_v4.py)
# ──────────────────────────────────────────────────────────────────────
class NetVLAD(nn.Module):
    def __init__(self, feature_dim, num_clusters=64, normalize_input=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.normalize_input = normalize_input
        self.conv = nn.Conv1d(feature_dim, num_clusters, kernel_size=1, bias=True)
        self.centroids = nn.Parameter(torch.randn(num_clusters, feature_dim) * 0.01)
        self.out_dim = num_clusters * feature_dim
        self._initialized = False

    def init_clusters(self, patch_descriptors):
        descs = patch_descriptors.numpy()
        if self.normalize_input:
            norms = np.linalg.norm(descs, axis=1, keepdims=True) + 1e-8
            descs = descs / norms
        km = KMeans(n_clusters=self.num_clusters, n_init=3, max_iter=100, random_state=42)
        km.fit(descs)
        centers = torch.from_numpy(km.cluster_centers_).float()
        self.centroids.data.copy_(centers)
        alpha = 1.0
        self.conv.weight.data.copy_((2.0 * alpha * centers).unsqueeze(2))
        self.conv.bias.data.copy_(-alpha * (centers ** 2).sum(dim=1))
        self._initialized = True
        print(f"  NetVLAD init: K-means inertia={km.inertia_:.1f}")

    def forward(self, x, mask=None):
        B, N, D = x.shape
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=2)
        soft_assign = self.conv(x.permute(0, 2, 1))  # (B, K, N)
        if mask is not None:
            mask_bool = mask.unsqueeze(1).bool()
            has_fg = mask_bool.any(dim=2, keepdim=True)
            effective = mask_bool | ~has_fg
            soft_assign = soft_assign.masked_fill(~effective, float("-inf"))
        soft_assign = F.softmax(soft_assign, dim=2)

        vlad = torch.zeros(B, self.num_clusters, D, device=x.device, dtype=x.dtype)
        for k in range(self.num_clusters):
            res_k = x - self.centroids[k].unsqueeze(0).unsqueeze(0)
            w_k = soft_assign[:, k, :].unsqueeze(2)
            vlad[:, k] = (res_k * w_k).sum(dim=1)

        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.reshape(B, -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


class LoRALinear(nn.Module):
    def __init__(self, original, rank=16, alpha=32):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(original.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


def apply_lora(backbone, rank, alpha, last_n_blocks):
    blocks = backbone.blocks
    n = len(blocks)
    lora_params = []
    for i in range(n - last_n_blocks, n):
        lora_qkv = LoRALinear(blocks[i].attn.qkv, rank, alpha)
        blocks[i].attn.qkv = lora_qkv
        lora_params.extend([lora_qkv.lora_A, lora_qkv.lora_B])
        if hasattr(blocks[i], "mlp") and hasattr(blocks[i].mlp, "fc1"):
            lora_fc1 = LoRALinear(blocks[i].mlp.fc1, rank, alpha)
            blocks[i].mlp.fc1 = lora_fc1
            lora_params.extend([lora_fc1.lora_A, lora_fc1.lora_B])
        if hasattr(blocks[i], "mlp") and hasattr(blocks[i].mlp, "fc2"):
            lora_fc2 = LoRALinear(blocks[i].mlp.fc2, rank, alpha)
            blocks[i].mlp.fc2 = lora_fc2
            lora_params.extend([lora_fc2.lora_A, lora_fc2.lora_B])
    return backbone, lora_params


def enable_gradient_checkpointing(backbone, last_n_blocks):
    blocks = backbone.blocks
    n = len(blocks)
    for i in range(n - last_n_blocks, n):
        orig = blocks[i].forward

        def make_ckpt(fn):
            def fwd(*args, **kwargs):
                return grad_checkpoint(fn, *args, use_reentrant=False, **kwargs)
            return fwd
        blocks[i].forward = make_ckpt(orig)


class ArcFaceHead(nn.Module):
    
    def __init__(self, embed_dim, num_classes, scale=30.0, margin=0.3, k=1):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.k = max(1, int(k))
        self.num_classes = num_classes
        # Layout: (num_classes * k, embed_dim) so we can use F.linear, then
        # reshape to (B, C, K) for max-reduce.
        self.prototypes = nn.Parameter(torch.randn(num_classes * self.k, embed_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, embeddings, labels=None):
        with torch.amp.autocast("cuda", enabled=False):
            embeddings = embeddings.float()
            proto = F.normalize(self.prototypes.float(), dim=1)
            cosine_all = F.linear(embeddings, proto)  # (B, C*K) or (B, C) if K=1
            if self.k > 1:
                B = cosine_all.shape[0]
                cosine = cosine_all.view(B, self.num_classes, self.k).max(dim=2)[0]
            else:
                cosine = cosine_all
        if labels is not None and self.training:
            theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
            idx = torch.arange(len(labels), device=labels.device)
            target_theta = theta[idx, labels] + self.margin
            margin_cosine = cosine.clone()
            margin_cosine[idx, labels] = torch.cos(target_theta)
            logits = self.scale * margin_cosine
        else:
            logits = self.scale * cosine
        return {"logits": logits, "cosine": cosine}


class AttentionMIL(nn.Module):
    """Gated attention-MIL page aggregator (Ilse et al., 2018).

    x : (B, K, dim) per-crop features
    quality : (B, K) optional float in [0,1] — added to logits via learned beta
    returns : pooled (B, dim), attention weights (B, K)
    """

    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, quality: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.ln(x)                                        # (B, K, dim)
        a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)) # (B, K, hidden)
        logit = self.w(a).squeeze(-1)                         # (B, K)
        if quality is not None:
            logit = logit + self.beta * quality
        logit = self.drop(logit)
        w = torch.softmax(logit, dim=1)                       # (B, K)
        pooled = (w.unsqueeze(-1) * x).sum(dim=1)            # (B, dim)
        return pooled, w


class WriterIDModel(nn.Module):
    def __init__(self, backbone, num_writers, cfg):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.num_writers = num_writers

        self.register_buffer(
            "centroid_sum", torch.zeros(num_writers, cfg.writer_embed_dim))
        self.register_buffer(
            "centroid_count", torch.zeros(num_writers))

        self.vlad = NetVLAD(cfg.dinov2_dim, num_clusters=cfg.vlad_clusters)
        vlad_out_dim = self.vlad.out_dim
        self.vlad_proj = nn.Sequential(
            nn.Linear(vlad_out_dim, cfg.projection_dim),
            nn.LayerNorm(cfg.projection_dim), nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(cfg.projection_dim + cfg.dinov2_dim, cfg.projection_dim),
            nn.LayerNorm(cfg.projection_dim), nn.GELU(),
            nn.Dropout(0.15),
        )
        self.writer_proj = nn.Sequential(
            nn.Linear(cfg.projection_dim, cfg.projection_dim),
            nn.LayerNorm(cfg.projection_dim), nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(cfg.projection_dim, cfg.writer_embed_dim),
        )
        self.writer_head = ArcFaceHead(
            cfg.writer_embed_dim, num_writers,
            scale=cfg.arcface_scale, margin=cfg.arcface_margin,
            k=cfg.multi_proto_k,
        )
        if cfg.page_agg == "attention":
            self.page_attn = AttentionMIL(cfg.projection_dim, hidden=128, dropout=0.1)

    def extract_features(self, images, masks):
        """Common DINOv2 -> VLAD -> projection path. Returns 512-d projected feat."""
        self.backbone.eval()
        out = self.backbone.forward_features(images)
        cls_token = out["x_norm_clstoken"]            # (B, D)
        patch_tokens = out["x_norm_patchtokens"]      # (B, N, D)
        grid = int(round(math.sqrt(patch_tokens.shape[1])))
        # Foreground mask: pool 224 mask down to grid x grid then flatten
        fg_mask = F.adaptive_avg_pool2d(masks.unsqueeze(1), (grid, grid)).flatten(1)
        fg_mask = (fg_mask > 0.1).float()
        vlad_desc = self.vlad(patch_tokens, fg_mask)
        vlad_feat = self.vlad_proj(vlad_desc)
        combined = torch.cat([cls_token, vlad_feat], dim=1)
        return self.projection(combined)

    def forward(self, images, masks, labels=None):
        feat = self.extract_features(images, masks)
        emb = F.normalize(self.writer_proj(feat), p=2, dim=1)
        head_out = self.writer_head(emb, labels)
        return {
            "embedding": emb,
            "logits": head_out["logits"],
            "cosine": head_out["cosine"],
        }

    def forward_page(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        quality: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Page-level forward using the attention-MIL aggregator.

        images  : (B, K, 3, 224, 224)
        masks   : (B, K, 224, 224)
        quality : (B, K) float in [0,1]  — ink density per crop
        labels  : (B,) long, or None at eval time
        """
        B, K = images.shape[:2]
        flat_images = images.view(B * K, 3, 224, 224)
        flat_masks = masks.view(B * K, 224, 224)
        feat = self.extract_features(flat_images, flat_masks)  # (B*K, projection_dim)
        feat = feat.view(B, K, -1)                             # (B, K, projection_dim)

        page_feat, attn_w = self.page_attn(feat, quality)      # (B, dim), (B, K)
        page_emb = F.normalize(self.writer_proj(page_feat), p=2, dim=1)
        page_head = self.writer_head(page_emb, labels)

        # Auxiliary per-crop loss (used during training only when labels given)
        crop_emb = F.normalize(
            self.writer_proj(feat.view(B * K, -1)), p=2, dim=1
        )  # (B*K, writer_embed_dim)
        crop_labels = labels.repeat_interleave(K) if labels is not None else None
        crop_head = self.writer_head(crop_emb, crop_labels)

        return {
            "page_embedding": page_emb,
            "page_logits": page_head["logits"],
            "page_cosine": page_head["cosine"],
            "crop_logits": crop_head["logits"],
            "crop_cosine": crop_head["cosine"],
            "attn_w": attn_w,
        }

    @torch.no_grad()
    def reset_centroids(self):
        self.centroid_sum.zero_()
        self.centroid_count.zero_()

    @torch.no_grad()
    def update_centroids(self, embeddings, labels):
        """Accumulate per-writer running sum of (already L2-normalized) embeddings."""
        embeddings = embeddings.detach().float()
        labels = labels.detach()
        self.centroid_sum.index_add_(0, labels, embeddings)
        ones = torch.ones_like(labels, dtype=torch.float32)
        self.centroid_count.index_add_(0, labels, ones)

    @torch.no_grad()
    def get_centroids(self):
        """Return L2-normalized per-writer centroid (zeros for unseen writers)."""
        counts = self.centroid_count.clamp_min(1.0).unsqueeze(1)
        c = self.centroid_sum / counts
        return F.normalize(c, p=2, dim=1)


# ──────────────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────────────
class EMAState:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        """Copy shadow weights into the model. Returns a backup dict to restore."""
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model, backup):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])


# ──────────────────────────────────────────────────────────────────────
# VLAD K-means init
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def init_vlad_kmeans(model, train_ds, device, n_images=200, max_tokens_per_image=128):
    print(f"\nVLAD init: forward {n_images} images through frozen backbone for K-means...")
    model.eval()
    indices = np.random.default_rng(0).choice(len(train_ds), size=min(n_images, len(train_ds)), replace=False)
    all_tokens = []
    for i, idx in enumerate(indices):
        sample = train_ds[int(idx)]
        img = sample["image"]
        mask_t = sample["mask"]
        if img.ndim == 4:  # multi-crop dataset: pick first crop
            img = img[0]
            mask_t = mask_t[0]
        img = img.unsqueeze(0).to(device)
        out = model.backbone.forward_features(img)
        tokens = out["x_norm_patchtokens"][0].cpu()  # (N, D)
        # Filter by ink mask: keep tokens whose patch had foreground
        mask = mask_t.unsqueeze(0).unsqueeze(0)
        grid = int(round(math.sqrt(tokens.shape[0])))
        fg = F.adaptive_avg_pool2d(mask, (grid, grid)).flatten(1)[0]
        keep = (fg > 0.1).nonzero().flatten()
        if len(keep) > 0:
            tokens = tokens[keep]
        if len(tokens) > max_tokens_per_image:
            sel = torch.randperm(len(tokens))[:max_tokens_per_image]
            tokens = tokens[sel]
        all_tokens.append(tokens)
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(indices)}")
    all_tokens = torch.cat(all_tokens, dim=0)
    print(f"  Collected {all_tokens.shape[0]} patch tokens")
    model.vlad.init_clusters(all_tokens)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def supcon_loss(embeddings, labels, tau=0.07):
    """Supervised contrastive loss on L2-normalized embeddings.

    Forced to float32 (logsumexp + -inf masking are numerically unsafe in
    float16 / autocast). Skips rows that have no positive in the batch.
    """
    with torch.amp.autocast("cuda", enabled=False):
        z = embeddings.float()  # ensure fp32; already L2-normalized at model output
        sim = (z @ z.t()) / tau
        n = z.shape[0]
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        pos_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
        sim = sim.masked_fill(self_mask, -1e9)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos_count = pos_mask.sum(dim=1).clamp_min(1)
        per_row = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count
        has_pos = pos_mask.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return torch.zeros((), device=z.device)
        return per_row[has_pos].mean()


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg, epoch):
    model.train()
    model.reset_centroids()
    total, correct, sum_loss = 0, 0, 0.0
    rng_k = np.random.default_rng(epoch)  # reproducible K-subsample per epoch

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if cfg.page_agg == "attention":
            # ── Attention-MIL branch ──────────────────────────────────
            # images: (B, K, 3, 224, 224)  masks: (B, K, 224, 224)
            B, K_full = images.shape[:2]
            # Randomly subsample K crops from the available K_full
            k_choices = [k for k in cfg.page_train_k_choices if k <= K_full]
            K = int(rng_k.choice(k_choices)) if k_choices else K_full
            if K < K_full:
                idx = torch.from_numpy(
                    rng_k.choice(K_full, size=K, replace=False)
                ).sort().values
                images = images[:, idx]
                masks = masks[:, idx]
            # Quality = ink density per crop
            quality = masks.view(B, K, -1).mean(dim=2)   # (B, K)
            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                out = model.forward_page(images, masks, quality, labels=labels)
                loss = F.cross_entropy(
                    out["page_logits"], labels,
                    label_smoothing=cfg.label_smoothing,
                )
                if cfg.page_aux_weight > 0.0:
                    crop_labels = labels.repeat_interleave(K)
                    aux = F.cross_entropy(
                        out["crop_logits"], crop_labels,
                        label_smoothing=cfg.label_smoothing,
                    )
                    loss = loss + cfg.page_aux_weight * aux
            if cfg.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.update_centroids(out["page_embedding"], labels)
            sum_loss += loss.item() * B
            total += B
            correct += (out["page_cosine"].argmax(dim=1) == labels).sum().item()

        else:
            # ── Mean-pool branch (unchanged) ─────────────────────────
            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                out = model(images, masks, labels=labels)
                loss = F.cross_entropy(
                    out["logits"], labels, label_smoothing=cfg.label_smoothing
                )
                if cfg.lambda_supcon > 0.0:
                    # SupCon on full-precision normalized embeddings
                    sc = supcon_loss(out["embedding"].float(), labels, tau=cfg.supcon_tau)
                    loss = loss + cfg.lambda_supcon * sc
                if cfg.hard_neg_weight > 0.0:
                    # Full-gallery hard-negative mining: closest non-true prototype
                    # across ALL writers (not just in-batch — fixes PK's failure mode).
                    with torch.amp.autocast("cuda", enabled=False):
                        cos = out["cosine"].float()  # (B, C), max-reduced over K
                        # Mask out the true class to find the hardest non-true negative
                        mask = torch.zeros_like(cos)
                        mask.scatter_(1, labels.unsqueeze(1), float("-inf"))
                        hard_neg_idx = (cos + mask).argmax(dim=1)
                        true_cos = cos.gather(1, labels.unsqueeze(1)).squeeze(1)
                        hn_cos = cos.gather(1, hard_neg_idx.unsqueeze(1)).squeeze(1)
                        triplet = F.relu(cfg.hard_neg_margin + hn_cos - true_cos).mean()
                    loss = loss + cfg.hard_neg_weight * triplet
            if cfg.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Accumulate per-writer centroid from this step's embeddings (fresh
            # each epoch). Used by --checkpoint-metric=centroid at val time.
            model.update_centroids(out["embedding"], labels)

            sum_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (out["cosine"].argmax(dim=1) == labels).sum().item()

    return sum_loss / total, correct / total


@torch.no_grad()
def extract_page_embeddings(model, eval_loader, device, cfg, n_writers, tta_hflip=False):
    
    model.eval()
    all_emb, all_raw, all_q, all_cos, all_wid, all_iid = [], [], [], [], [], []
    for batch in tqdm(eval_loader, desc="eval", leave=False, dynamic_ncols=True):
        # batch images: (B, N, 3, H, W). Flatten to (B*N, 3, H, W)
        B, N = batch["image"].shape[:2]
        images = batch["image"].view(B * N, 3, 224, 224).to(device, non_blocking=True)
        masks = batch["mask"].view(B * N, 224, 224).to(device, non_blocking=True)
        # Per-patch quality = ink density. Cheap, computed from mask directly.
        quality = masks.view(B, N, -1).mean(dim=2).cpu().numpy()  # (B, N)
        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            out = model(images, masks)
            patch_emb = out["embedding"]                  # (B*N, D)  L2-normed
            if tta_hflip:
                # Horizontal flip TTA: flip W axis of both image and mask, forward again, average
                images_f = torch.flip(images, dims=[3])
                masks_f = torch.flip(masks, dims=[2])
                out_f = model(images_f, masks_f)
                patch_emb = (patch_emb + out_f["embedding"]) * 0.5
                patch_emb = F.normalize(patch_emb, p=2, dim=1)
        # Per-patch embeddings (already L2-normalized inside the head)
        raw = patch_emb.view(B, N, -1)                    # (B, N, D)
        if cfg.page_agg == "attention":
            # Re-extract pre-writer_proj features for the MIL aggregator.
            # patch_emb is already writer_proj -> L2-normed (dim=writer_embed_dim).
            # We need projection_dim features, so re-run extract_features.
            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                feat = model.extract_features(
                    batch["image"].view(B * N, 3, 224, 224).to(device, non_blocking=True),
                    batch["mask"].view(B * N, 224, 224).to(device, non_blocking=True),
                )                                          # (B*N, projection_dim)
            feat = feat.view(B, N, -1)                    # (B, N, projection_dim)
            q_tensor = torch.from_numpy(quality).to(device, non_blocking=True)  # (B, N)
            page_feat, _ = model.page_attn(feat, q_tensor)  # (B, projection_dim)
            emb = F.normalize(model.writer_proj(page_feat), p=2, dim=1)
        else:
            emb = raw.mean(dim=1)                         # mean-aggregate per page
            emb = F.normalize(emb, p=2, dim=1)
        # Use the head's forward to get cosines (handles multi-proto max-reduce)
        head_out = model.writer_head(emb)
        cos = head_out["cosine"]
        all_emb.append(emb.cpu().numpy())
        all_raw.append(raw.cpu().numpy())
        all_q.append(quality)
        all_cos.append(cos.cpu().numpy())
        all_wid.extend(batch["writer_id"])
        all_iid.extend(batch["image_id"])
    return {
        "emb":         np.concatenate(all_emb, axis=0),
        "raw_patches": np.concatenate(all_raw, axis=0),
        "quality":     np.concatenate(all_q, axis=0),
        "cosine":      np.concatenate(all_cos, axis=0),
        "writer_id":   all_wid,
        "image_id":    all_iid,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--splits-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-patches-eval", type=int, default=64)
    ap.add_argument("--dinov2-variant", type=str, default="dinov2_vitb14_reg",
                    choices=["dinov2_vitb14_reg", "dinov2_vitl14_reg",
                             "dinov2_vits14_reg", "dinov2_vitg14_reg"],
                    help="DINOv2 backbone variant. dim auto-set: s=384, b=768, l=1024, g=1536")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-blocks", type=int, default=6)
    ap.add_argument("--vlad-clusters", type=int, default=64)
    ap.add_argument("--aggregator", type=str, default="netvlad",
                    choices=["netvlad"],
                    help="Patch aggregation layer (NetVLAD, K learned cluster "
                         "centers via K-means init).")
    ap.add_argument("--arcface-scale", type=float, default=30.0)
    ap.add_argument("--arcface-margin", type=float, default=0.3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-aug", action="store_true",
                    help="Disable train-time augmentation (rotation + color jitter)")
    ap.add_argument("--aug-rotation-deg", type=float, default=5.0)
    ap.add_argument("--aug-brightness", type=float, default=0.15)
    ap.add_argument("--aug-contrast", type=float, default=0.15)
    ap.add_argument("--aug-binary-mode", type=str, default="auto",
                    choices=["auto", "yes", "no"],
                    help="auto = detect from sample images; yes = force binary-safe "
                         "(skip color jitter); no = always apply color jitter")
    ap.add_argument("--use-skeleton", action="store_true",
                    help="Replace each 224x224 RGB patch with a distance-transform "
                         "map encoding stroke half-width (Otsu binarize -> "
                         "cv2.distanceTransform -> 3-channel tile). Sets "
                         "INPUT_MODE=skeleton_dt so DataLoader workers inherit it.")
    ap.add_argument("--patch-zoom", type=float, default=1.0,
                    help="Zoom factor for patch sampling. >1.0 = zoom in (sample "
                         "smaller crop from page, upscale to 224 before DINOv2). "
                         "Use 2.0 for HistoryWI to recover stroke detail from "
                         "densely-written pages.")
    ap.add_argument("--pk-sampler", action="store_true",
                    help="Use PK batch sampler (P writers × K pages each). Guarantees "
                         "same-writer pairs in every batch. Required for SupCon and "
                         "improves ArcFace gradient quality on 720-class HWI.")
    ap.add_argument("--pk-pages-per-writer", type=int, default=2,
                    help="K in PK sampling: pages per writer per batch (default 2).")
    ap.add_argument("--n-crops", type=int, default=1,
                    help="Random crops per page per training step (default 1). "
                         "Set to 2+ to teach within-page crop invariance. Effective "
                         "batch items = (pages_per_batch) * n_crops; pages_per_batch "
                         "is auto-derived from --batch-size / --n-crops.")
    ap.add_argument("--lambda-supcon", type=float, default=0.0,
                    help="Supervised contrastive loss weight (added to ArcFace CE). "
                         "0 = disabled (default). 0.05–0.2 typical. Needs PK sampler "
                         "to have same-writer pairs in each batch.")
    ap.add_argument("--supcon-tau", type=float, default=0.07,
                    help="SupCon temperature (default 0.07).")
    ap.add_argument("--checkpoint-metric", type=str, default="prototype_auroc",
                    choices=["prototype_auroc", "centroid_known_top1"],
                    help="Metric used to select best checkpoint: prototype_auroc "
                         "(default — val OOD AUROC vs ArcFace prototypes) or "
                         "centroid_known_top1 (val Top1 of known writers vs "
                         "running per-writer train-embedding centroids; more "
                         "honest, addresses audit finding #3).")
    ap.add_argument("--multi-proto-k", type=int, default=1,
                    help="K prototypes per writer in ArcFace head (default 1). "
                         "K=2 or 3 lets writers span multiple style/script modes — "
                         "directly addresses HWI's within-writer cosine variance.")
    ap.add_argument("--hard-neg-weight", type=float, default=0.0,
                    help="Weight on full-gallery hard-negative triplet loss "
                         "(default 0 = disabled). 0.1-0.5 typical. Adds a triplet "
                         "term using the closest non-true prototype across ALL writers "
                         "— fixes PK sampler's 'too few in-batch negatives' failure.")
    ap.add_argument("--hard-neg-margin", type=float, default=0.1,
                    help="Required margin between true-class cosine and hardest "
                         "non-true cosine (default 0.1).")
    ap.add_argument("--init-from", type=str, default=None,
                    help="Path to a pretrain checkpoint (e.g. an IAM-trained "
                         "checkpoint.pt) to initialize the model from. Loads "
                         "everything except ArcFace prototypes and centroid "
                         "buffers (which depend on class count). Use for "
                         "cross-dataset transfer learning: IAM -> HWI etc.")
    ap.add_argument("--inference-only", type=str, default=None,
                    help="Path to a trained checkpoint. Skips training entirely; "
                         "loads the checkpoint into a model with the SAME class "
                         "count, runs final embedding extraction + saves a new "
                         "embeddings_seed*.npz. Used for cheap inference-time "
                         "experiments like --tta-hflip on existing models.")
    ap.add_argument("--tta-hflip", action="store_true",
                    help="Enable horizontal-flip test-time augmentation: each "
                         "patch is forwarded twice (orig + hflip), embeddings "
                         "averaged before page aggregation. Costs ~2× inference "
                         "time. Use with --inference-only on existing checkpoints.")
    ap.add_argument("--page-agg", type=str, default="mean",
                    choices=["mean", "attention"],
                    help="Page-level aggregation strategy. 'mean' (default) averages "
                         "L2-normed patch embeddings — identical to prior behaviour. "
                         "'attention' uses a learned gated attention-MIL module that "
                         "produces a quality-weighted page embedding and trains with "
                         "an auxiliary per-crop ArcFace loss.")
    ap.add_argument("--page-aux-weight", type=float, default=0.1,
                    help="Weight on the auxiliary per-crop ArcFace CE loss used only "
                         "when --page-agg attention is active (default 0.1). "
                         "Set to 0 to disable the auxiliary term.")
    return ap.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    if args.use_skeleton:
        os.environ["INPUT_MODE"] = "skeleton_dt"
        patch_lib._IS_SKELETON_MODE = True
        if "skeleton" not in args.out_dir.lower():
            print(f"WARNING: --use-skeleton active but --out-dir='{args.out_dir}' "
                  f"does not contain 'skeleton'. Consider renaming to avoid "
                  f"clobbering a color-mode run.")

    _DIM_PER_VARIANT = {
        "dinov2_vits14_reg": 384,
        "dinov2_vitb14_reg": 768,
        "dinov2_vitl14_reg": 1024,
        "dinov2_vitg14_reg": 1536,
    }
    cfg = Config(
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_patches_eval=args.n_patches_eval,
        dinov2_variant=args.dinov2_variant,
        dinov2_dim=_DIM_PER_VARIANT[args.dinov2_variant],
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_last_n_blocks=args.lora_blocks,
        vlad_clusters=args.vlad_clusters,
        aggregator=args.aggregator,
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        aug_enabled=not args.no_aug,
        aug_rotation_deg=args.aug_rotation_deg,
        aug_brightness=args.aug_brightness,
        aug_contrast=args.aug_contrast,
        aug_binary_mode=args.aug_binary_mode,
        patch_zoom=args.patch_zoom,
        pk_sampler=args.pk_sampler,
        pk_pages_per_writer=args.pk_pages_per_writer,
        n_crops=args.n_crops,
        lambda_supcon=args.lambda_supcon,
        supcon_tau=args.supcon_tau,
        checkpoint_metric=args.checkpoint_metric,
        multi_proto_k=args.multi_proto_k,
        hard_neg_weight=args.hard_neg_weight,
        hard_neg_margin=args.hard_neg_margin,
        page_agg=args.page_agg,
        page_aux_weight=args.page_aux_weight,
    )
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    input_mode = "skeleton_dt" if args.use_skeleton else "color"
    print(f"Input mode: {input_mode}  (patches._IS_SKELETON_MODE="
          f"{patch_lib._IS_SKELETON_MODE})")
    print(f"Config: epochs={cfg.epochs} bs={cfg.batch_size} lora_blocks={cfg.lora_last_n_blocks}")

    # ── Data ──────────────────────────────────────────────────────────
    splits = Path(cfg.splits_dir)
    df_train = pd.read_csv(splits / "train.csv")
    df_val = pd.read_csv(splits / "val.csv")
    df_test = pd.read_csv(splits / "test.csv")
    # Train must have no -1 rows
    n_train_unk = (df_train["writer_id"].astype(str) == "-1").sum()
    if n_train_unk > 0:
        print(f"WARNING: train.csv has {n_train_unk} '-1' rows; dropping them")
        df_train = df_train[df_train["writer_id"].astype(str) != "-1"].reset_index(drop=True)

    # Force writer_id to string for stable mapping
    for d in (df_train, df_val, df_test):
        d["writer_id"] = d["writer_id"].astype(str)

    writers = sorted(df_train["writer_id"].unique().tolist())
    writer2idx = {w: i for i, w in enumerate(writers)}
    n_writers = len(writers)
    print(f"Known writers: {n_writers}")
    print(f"Train pages: {len(df_train)}  Val pages: {len(df_val)}  Test pages: {len(df_test)}")

    # Resolve binary-mode flag (auto: sample a few train images and check gray levels)
    if cfg.aug_binary_mode == "auto":
        sample_idxs = df_train.sample(min(5, len(df_train)), random_state=cfg.seed).index
        votes = []
        for i in sample_idxs:
            p = Path(cfg.data_dir) / df_train.loc[i, "image_path"]
            try:
                votes.append(patch_lib.is_binarized_image(patch_lib.load_image(p)))
            except Exception:
                pass
        binary_mode = bool(votes) and sum(votes) > len(votes) / 2
        print(f"Binary-mode auto-detect: {binary_mode} (votes={votes})")
    else:
        binary_mode = (cfg.aug_binary_mode == "yes")

    print(f"Augmentation: enabled={cfg.aug_enabled} rotation={cfg.aug_rotation_deg}deg "
          f"brightness={cfg.aug_brightness} contrast={cfg.aug_contrast} binary_mode={binary_mode}")

    # Derive view_size from patch_zoom (None = no resize, else int crop size for 1/zoom of 224)
    view_size = None if cfg.patch_zoom == 1.0 else max(14, int(round(patch_lib.PATCH_SIZE / cfg.patch_zoom)))
    print(f"Patch zoom: {cfg.patch_zoom}x  (view_size={view_size if view_size else patch_lib.PATCH_SIZE}, "
          f"output size=224)")

    train_ds = TrainPatchDataset(
        df_train, cfg.data_dir, writer2idx,
        min_ink_frac=cfg.min_ink_frac,
        aug_enabled=cfg.aug_enabled,
        aug_rotation_deg=cfg.aug_rotation_deg,
        aug_brightness=cfg.aug_brightness,
        aug_contrast=cfg.aug_contrast,
        binary_mode=binary_mode,
        view_size=view_size,
        n_crops=cfg.n_crops,
    )
    val_ds = EvalPagePatchDataset(df_val, cfg.data_dir, n_patches=cfg.n_patches_eval, min_ink_frac=cfg.min_ink_frac, view_size=view_size)
    test_ds = EvalPagePatchDataset(df_test, cfg.data_dir, n_patches=cfg.n_patches_eval, min_ink_frac=cfg.min_ink_frac, view_size=view_size)
    train_eval_ds = EvalPagePatchDataset(df_train, cfg.data_dir, n_patches=cfg.n_patches_eval, min_ink_frac=cfg.min_ink_frac, view_size=view_size)

    
    pages_per_batch = max(1, cfg.batch_size // cfg.n_crops)

    train_collate_fn = (
        train_pageagg_collate if cfg.page_agg == "attention"
        else train_multicrop_collate
    )
    if cfg.page_agg == "attention":
        print(f"Page aggregation: attention-MIL  "
              f"(aux_weight={cfg.page_aux_weight}, "
              f"k_choices={cfg.page_train_k_choices})")

    pk_sampler_obj = None
    if cfg.pk_sampler:
        writers_per_batch = max(1, pages_per_batch // cfg.pk_pages_per_writer)
        pk_sampler_obj = PKBatchSampler(
            labels=[writer2idx[w] for w in df_train["writer_id"].tolist()],
            writers_per_batch=writers_per_batch,
            pages_per_writer=cfg.pk_pages_per_writer,
            seed=cfg.seed,
        )
        print(f"PK sampler: writers/batch={writers_per_batch} × "
              f"pages/writer={cfg.pk_pages_per_writer} × n_crops={cfg.n_crops} "
              f"= {writers_per_batch * cfg.pk_pages_per_writer * cfg.n_crops} items/batch "
              f"(batches/epoch={len(pk_sampler_obj)})")
        train_loader = DataLoader(
            train_ds, batch_sampler=pk_sampler_obj,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            persistent_workers=(cfg.num_workers > 0),
            collate_fn=train_collate_fn,
        )
    else:
        print(f"Shuffle sampler: pages/batch={pages_per_batch} × n_crops={cfg.n_crops} "
              f"= {pages_per_batch * cfg.n_crops} items/batch")
        train_loader = DataLoader(
            train_ds, batch_size=pages_per_batch, shuffle=True, drop_last=True,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            persistent_workers=(cfg.num_workers > 0),
            collate_fn=train_collate_fn,
        )
    eval_kw = dict(batch_size=4, shuffle=False, num_workers=cfg.num_workers,
                   pin_memory=cfg.pin_memory, collate_fn=eval_collate,
                   persistent_workers=(cfg.num_workers > 0))
    val_loader = DataLoader(val_ds, **eval_kw)
    test_loader = DataLoader(test_ds, **eval_kw)
    train_eval_loader = DataLoader(train_eval_ds, **eval_kw)

    # ── Model ─────────────────────────────────────────────────────────
    print(f"\nLoading DINOv2 {cfg.dinov2_variant}...")
    backbone = torch.hub.load(cfg.dinov2_repo, cfg.dinov2_variant, pretrained=True)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone, lora_params = apply_lora(backbone, cfg.lora_rank, cfg.lora_alpha, cfg.lora_last_n_blocks)
    if cfg.grad_checkpointing:
        enable_gradient_checkpointing(backbone, cfg.lora_last_n_blocks)
        print(f"  Gradient checkpointing: enabled for last {cfg.lora_last_n_blocks} blocks")
    print(f"  LoRA params: {sum(p.numel() for p in lora_params):,}")

    model = WriterIDModel(backbone, n_writers, cfg).to(device)

    
    if args.init_from:
        print(f"\nLoading pretrain checkpoint: {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        pretrain_sd = ckpt.get("state_dict", ckpt)
        # Drop class-count-dependent tensors
        SKIP_PREFIXES = ("writer_head.prototypes", "centroid_sum", "centroid_count")
        filtered = {k: v for k, v in pretrain_sd.items()
                    if not any(k.startswith(p) for p in SKIP_PREFIXES)}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        n_loaded = len(filtered)
        n_skip = len(pretrain_sd) - n_loaded
        print(f"  Loaded {n_loaded} tensors. Skipped {n_skip} (class-count "
              f"mismatch). Re-init: {[k for k in missing if 'writer_head' in k or 'centroid' in k][:3]}...")
        if unexpected:
            print(f"  WARNING: unexpected keys in checkpoint: {unexpected[:5]}...")

    # ── Inference-only fast path (skip training, just extract embeddings) ──
    if args.inference_only:
        print(f"\nINFERENCE-ONLY mode: loading {args.inference_only}")
        ckpt = torch.load(args.inference_only, map_location=device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  Loaded checkpoint. missing={len(missing)} unexpected={len(unexpected)}")
        if args.tta_hflip:
            print("  TTA: horizontal flip enabled (2× inference cost)")
        print("\nExtracting embeddings (train, val, test)...")
        train_out = extract_page_embeddings(model, train_eval_loader, device, cfg, n_writers, tta_hflip=args.tta_hflip)
        val_out = extract_page_embeddings(model, val_loader, device, cfg, n_writers, tta_hflip=args.tta_hflip)
        test_out = extract_page_embeddings(model, test_loader, device, cfg, n_writers, tta_hflip=args.tta_hflip)
        train_labels = np.array([writer2idx[w] for w in train_out["writer_id"]], dtype=np.int64)
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_path = out_dir / f"embeddings_seed{cfg.seed}.npz"
        np.savez(
            emb_path,
            train_emb=train_out["emb"].astype(np.float32),
            train_labels=train_labels,
            train_raw_patches=train_out["raw_patches"].astype(np.float32),
            train_quality=train_out["quality"].astype(np.float32),
            val_emb=val_out["emb"].astype(np.float32),
            val_writer_id=np.array(val_out["writer_id"]),
            val_image_id=np.array(val_out["image_id"]),
            val_cosine=val_out["cosine"].astype(np.float32),
            val_raw_patches=val_out["raw_patches"].astype(np.float32),
            val_quality=val_out["quality"].astype(np.float32),
            test_emb=test_out["emb"].astype(np.float32),
            test_writer_id=np.array(test_out["writer_id"]),
            test_image_id=np.array(test_out["image_id"]),
            test_cosine=test_out["cosine"].astype(np.float32),
            test_raw_patches=test_out["raw_patches"].astype(np.float32),
            test_quality=test_out["quality"].astype(np.float32),
            writers=np.array(writers),
        )
        print(f"Saved embeddings -> {emb_path}")
        print("Inference-only run complete. Skip training and exit.")
        return

    init_vlad_kmeans(model, train_ds, device)

    # ── Optimizer ─────────────────────────────────────────────────────
    # page_attn params (if present) are already included in named_parameters()
    # and are not in lora_params, so they fall into head_params automatically.
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not any(p is q for q in lora_params)]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": cfg.lr_lora, "weight_decay": cfg.weight_decay},
        {"params": head_params, "lr": cfg.lr_heads, "weight_decay": cfg.weight_decay},
    ])
    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    cosine_steps = max(1, cfg.epochs * steps_per_epoch - warmup_steps)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
    ema = None

    # ── Training Loop ─────────────────────────────────────────────────
    print(f"\nTraining for {cfg.epochs} epochs (steps/epoch={steps_per_epoch})")
    print(f"Checkpoint metric: {cfg.checkpoint_metric}  "
          f"SupCon: lambda={cfg.lambda_supcon} tau={cfg.supcon_tau}")
    best_metric = -1.0
    best_state = None
    for epoch in range(cfg.epochs):
        t0 = time.time()
        if pk_sampler_obj is not None:
            pk_sampler_obj.set_epoch(epoch)
        loss, acc = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg, epoch)
        if epoch == cfg.ema_start_epoch:
            ema = EMAState(model, decay=cfg.ema_decay)
        if ema is not None:
            ema.update(model)
        dt = time.time() - t0
        msg = f"Epoch {epoch+1:3d}/{cfg.epochs}  loss={loss:.4f}  train_acc={acc:.4f}  time={dt:.1f}s"

        # Light val every few epochs (use EMA if available)
        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            backup = ema.apply_to(model) if ema is not None else None
            val_out = extract_page_embeddings(model, val_loader, device, cfg, n_writers)
            # OOD AUROC: -1 rows as positives (higher max_cosine -> known)
            val_max_cos = val_out["cosine"].max(axis=1)
            val_is_unk = np.array([w == "-1" for w in val_out["writer_id"]])
            auroc = None
            if val_is_unk.sum() > 0 and (~val_is_unk).sum() > 0:
                auroc = roc_auc_score(~val_is_unk, val_max_cos)
                msg += f"  val_AUROC={auroc:.4f}"

            # Centroid-based Known_Top1 (audit finding #3): rank val embeddings
            # against running per-writer train-embedding centroids, not against
            # ArcFace prototypes.
            centroid_known_top1 = None
            centroids = model.get_centroids().detach().cpu().numpy()  # (W, D)
            seen_writers = (model.centroid_count.detach().cpu().numpy() > 0)
            known_mask = ~val_is_unk
            if known_mask.sum() > 0 and seen_writers.any():
                centroid_cos = val_out["emb"] @ centroids.T          # (P, W)
                # Mask unseen writers (their centroid is zeros) — argmax should
                # not pick those by default. Safe-guard by setting their cos to -inf.
                centroid_cos[:, ~seen_writers] = -np.inf
                pred_idx = centroid_cos[known_mask].argmax(axis=1)
                true_writers = [val_out["writer_id"][i] for i in range(len(known_mask)) if known_mask[i]]
                true_idx = np.array([writer2idx[w] for w in true_writers], dtype=np.int64)
                centroid_known_top1 = float((pred_idx == true_idx).mean())
                msg += f"  val_KnownTop1c={centroid_known_top1:.4f}"

            # Choose checkpoint metric
            if cfg.checkpoint_metric == "centroid_known_top1":
                metric_val = centroid_known_top1
            else:
                metric_val = auroc
            if metric_val is not None and metric_val > best_metric:
                best_metric = metric_val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if ema is not None:
                ema.restore(model, backup)
        print(msg)

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ── Final: apply EMA (if any), load best, extract embeddings ─────
    if ema is not None:
        ema.apply_to(model)
    model.load_state_dict(best_state)

    print("\nExtracting final embeddings (train, val, test)...")
    train_out = extract_page_embeddings(model, train_eval_loader, device, cfg, n_writers)
    val_out = extract_page_embeddings(model, val_loader, device, cfg, n_writers)
    test_out = extract_page_embeddings(model, test_loader, device, cfg, n_writers)

    # train labels as int indices
    train_labels = np.array([writer2idx[w] for w in train_out["writer_id"]], dtype=np.int64)

    out_dir = Path(cfg.out_dir)
    ckpt_path = out_dir / f"checkpoint_seed{cfg.seed}.pt"
    emb_path = out_dir / f"embeddings_seed{cfg.seed}.npz"

    torch.save({
        "state_dict": best_state,
        "config": cfg.__dict__,
        "writers": writers,
        "best_metric": best_metric,
        "checkpoint_metric": cfg.checkpoint_metric,
    }, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")

    np.savez(
        emb_path,
        train_emb=train_out["emb"].astype(np.float32),
        train_labels=train_labels,
        train_raw_patches=train_out["raw_patches"].astype(np.float32),
        train_quality=train_out["quality"].astype(np.float32),
        val_emb=val_out["emb"].astype(np.float32),
        val_writer_id=np.array(val_out["writer_id"]),
        val_image_id=np.array(val_out["image_id"]),
        val_cosine=val_out["cosine"].astype(np.float32),
        val_raw_patches=val_out["raw_patches"].astype(np.float32),
        val_quality=val_out["quality"].astype(np.float32),
        test_emb=test_out["emb"].astype(np.float32),
        test_writer_id=np.array(test_out["writer_id"]),
        test_image_id=np.array(test_out["image_id"]),
        test_cosine=test_out["cosine"].astype(np.float32),
        test_raw_patches=test_out["raw_patches"].astype(np.float32),
        test_quality=test_out["quality"].astype(np.float32),
        writers=np.array(writers),
    )
    print(f"Saved embeddings -> {emb_path}")
    print(f"\nBest val metric ({cfg.checkpoint_metric}): {best_metric:.4f}")
    print("Run submit_writerid.py next to generate submissions.")


if __name__ == "__main__":
    main()
