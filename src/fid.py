from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import inception_v3
from torchvision.transforms.functional import resize
from tqdm import tqdm
import numpy as np
from scipy import linalg

from .utils import ensure_dir


class InceptionEmbedder(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        # Be robust to different torchvision versions in Kaggle
        try:
            from torchvision.models import Inception_V3_Weights  # type: ignore
            model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        except Exception:
            # Fallback for older torchvision: uses pretrained flag
            model = inception_v3(pretrained=True, transform_input=False)
        model.fc = nn.Identity()
        model.eval()
        self.model = model.to(device)

    @torch.no_grad()
    def get_activations(self, x: torch.Tensor) -> torch.Tensor:
        x = resize(x, [299, 299])
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)


def _compute_stats(acts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


def _compute_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


@dataclass
class FIDConfig:
    work_dir: str
    num_samples_per_class: int = 1000
    batch_size: int = 64
    device: str = "cuda"
    num_workers: int = 2


class FIDEvaluator:
    def __init__(self, generator, num_classes: int, cfg: FIDConfig):
        self.G = generator
        self.num_classes = num_classes
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.embedder = InceptionEmbedder(self.device)
        ensure_dir(os.path.join(cfg.work_dir, "metrics"))

    @torch.no_grad()
    def _collect_acts_dataset(self, dataset: Dataset, per_class: bool = True) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        acts_by_class: Dict[int, List[np.ndarray]] = {c: [] for c in range(self.num_classes)}
        for x, y in tqdm(loader, desc="Dataset activations"):
            x = x.to(self.device)
            y = y.to(self.device)
            feats = self.embedder.get_activations(x)
            feats = feats.cpu().numpy()
            for i in range(len(y)):
                acts_by_class[int(y[i].item())].append(feats[i:i+1])
        stats = {c: _compute_stats(np.concatenate(v, axis=0)) for c, v in acts_by_class.items() if len(v) > 0}
        return stats

    @torch.no_grad()
    def _collect_acts_generator(self, z_dim: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        self.G.eval()
        per_class = self.cfg.num_samples_per_class
        collected: Dict[int, List[np.ndarray]] = {c: [] for c in range(self.num_classes)}
        for cls in range(self.num_classes):
            remaining = per_class
            while remaining > 0:
                bsz = min(self.cfg.batch_size, remaining)
                z = torch.randn(bsz, z_dim, device=self.device)
                y = torch.full((bsz,), cls, dtype=torch.long, device=self.device)
                x = self.G(z, y)
                feats = self.embedder.get_activations(x)
                collected[cls].append(feats.cpu().numpy())
                remaining -= bsz
        stats = {c: _compute_stats(np.concatenate(v, axis=0)) for c, v in collected.items() if len(v) > 0}
        self.G.train()
        return stats

    def compute_fid(self, dataset: Dataset, z_dim: int) -> Dict[str, float]:
        real_stats = self._collect_acts_dataset(dataset)
        fake_stats = self._collect_acts_generator(z_dim)
        per_class_fid: Dict[str, float] = {}
        vals: List[float] = []
        for c in range(self.num_classes):
            if c not in real_stats or c not in fake_stats:
                continue
            mu_r, sig_r = real_stats[c]
            mu_f, sig_f = fake_stats[c]
            fid = _compute_frechet_distance(mu_r, sig_r, mu_f, sig_f)
            per_class_fid[f"class_{c}"] = float(fid)
            vals.append(fid)
        if len(vals) > 0:
            per_class_fid["overall_mean"] = float(np.mean(vals))
        return per_class_fid


class FIDEvaluatorFromSampler:
    """FID evaluator that uses a provided sampling function instead of a local generator.

    The sampler must accept two tensors `(z, y)` on the current device and return images in `[-1, 1]`.
    """

    def __init__(self, sampler, num_classes: int, cfg: FIDConfig):
        self.sampler = sampler
        self.num_classes = num_classes
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.embedder = InceptionEmbedder(self.device)
        ensure_dir(os.path.join(cfg.work_dir, "metrics"))

    @torch.no_grad()
    def _collect_acts_generator(self, z_dim: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        per_class = self.cfg.num_samples_per_class
        collected: Dict[int, List[np.ndarray]] = {c: [] for c in range(self.num_classes)}
        for cls in range(self.num_classes):
            remaining = per_class
            while remaining > 0:
                bsz = min(self.cfg.batch_size, remaining)
                z = torch.randn(bsz, z_dim, device=self.device)
                y = torch.full((bsz,), cls, dtype=torch.long, device=self.device)
                x = self.sampler(z, y)
                feats = self.embedder.get_activations(x)
                collected[cls].append(feats.cpu().numpy())
                remaining -= bsz
        stats = {c: _compute_stats(np.concatenate(v, axis=0)) for c, v in collected.items() if len(v) > 0}
        return stats

    @torch.no_grad()
    def _collect_acts_dataset(self, dataset: Dataset, per_class: bool = True) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        acts_by_class: Dict[int, List[np.ndarray]] = {c: [] for c in range(self.num_classes)}
        for x, y in tqdm(loader, desc="Dataset activations"):
            x = x.to(self.device)
            y = y.to(self.device)
            feats = self.embedder.get_activations(x)
            feats = feats.cpu().numpy()
            for i in range(len(y)):
                acts_by_class[int(y[i].item())].append(feats[i:i+1])
        stats = {c: _compute_stats(np.concatenate(v, axis=0)) for c, v in acts_by_class.items() if len(v) > 0}
        return stats

    def compute_fid(self, dataset: Dataset, z_dim: int) -> Dict[str, float]:
        real_stats = self._collect_acts_dataset(dataset)
        fake_stats = self._collect_acts_generator(z_dim)
        per_class_fid: Dict[str, float] = {}
        vals: List[float] = []
        for c in range(self.num_classes):
            if c not in real_stats or c not in fake_stats:
                continue
            mu_r, sig_r = real_stats[c]
            mu_f, sig_f = fake_stats[c]
            fid = _compute_frechet_distance(mu_r, sig_r, mu_f, sig_f)
            per_class_fid[f"class_{c}"] = float(fid)
            vals.append(fid)
        if len(vals) > 0:
            per_class_fid["overall_mean"] = float(np.mean(vals))
        return per_class_fid


