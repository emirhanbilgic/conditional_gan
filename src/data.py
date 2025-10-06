from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive
import shutil
import zipfile


def build_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


@dataclass
class ImageNetSubsetConfig:
    data_dir: str  # expects subfolders: train/, val/
    selected_class_names: List[str] | None
    img_size: int = 64
    auto_select_k: int = 10  # used if selected_class_names is None
    val_split: float = 0.2  # used if train/val folders not present
    seed: int = 42


class RemappedSubset(Dataset):
    def __init__(self, base: Dataset, indices: List[int], orig_idx_to_new: Dict[int, int]):
        self.base = base
        self.indices = indices
        self.orig_idx_to_new = orig_idx_to_new

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        img, orig_label = self.base[base_idx]
        new_label = self.orig_idx_to_new[int(orig_label)]
        return img, torch.tensor(new_label, dtype=torch.long)


def _filter_indices_by_orig_indices(dataset: Dataset, keep_orig_indices: List[int]) -> List[int]:
    keep = set(int(c) for c in keep_orig_indices)
    indices: List[int] = []
    for i in range(len(dataset.samples)):
        _, y = dataset.samples[i]
        if int(y) in keep:
            indices.append(i)
    return indices


def load_imagenet_datasets(cfg: ImageNetSubsetConfig) -> Tuple[Dataset, Dataset, Dict[int, int], List[str]]:
    """
    Expects ImageNet-like directory:
      data_dir/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg

    Returns:
      train_subset: labels remapped to [0..k-1]
      val_subset: labels remapped to [0..k-1]
      orig_idx_to_new: mapping original integer label (ImageFolder index) -> new label
      selected_class_names: the ordered list of selected class names
    """
    train_root = os.path.join(cfg.data_dir, "train")
    val_root = os.path.join(cfg.data_dir, "val")
    tfm = build_transforms(cfg.img_size)

    if os.path.isdir(train_root) and os.path.isdir(val_root):
        # Standard ImageNet-style structure
        train_base = datasets.ImageFolder(root=train_root, transform=tfm)
        val_base = datasets.ImageFolder(root=val_root, transform=tfm)

        available_classes = list(train_base.class_to_idx.keys())
        if set(available_classes) != set(val_base.class_to_idx.keys()):
            raise ValueError("Train and val class sets differ; please align your dataset.")

        if cfg.selected_class_names is None or len(cfg.selected_class_names) == 0:
            selected_class_names = sorted(available_classes)[: cfg.auto_select_k]
        else:
            for c in cfg.selected_class_names:
                if c not in available_classes:
                    raise ValueError(f"Class '{c}' not found in dataset. Available example: {available_classes[:5]} ...")
            selected_class_names = list(cfg.selected_class_names)

        orig_indices = [train_base.class_to_idx[c] for c in selected_class_names]
        orig_idx_to_new = {orig: i for i, orig in enumerate(orig_indices)}

        train_idx = _filter_indices_by_orig_indices(train_base, orig_indices)
        val_idx = _filter_indices_by_orig_indices(val_base, orig_indices)

        train_subset = RemappedSubset(train_base, train_idx, orig_idx_to_new)
        val_subset = RemappedSubset(val_base, val_idx, orig_idx_to_new)
        return train_subset, val_subset, orig_idx_to_new, selected_class_names
    else:
        # Single folder with class subdirs; perform stratified split
        base = datasets.ImageFolder(root=cfg.data_dir, transform=tfm)
        available_classes = list(base.class_to_idx.keys())

        if cfg.selected_class_names is None or len(cfg.selected_class_names) == 0:
            selected_class_names = sorted(available_classes)[: cfg.auto_select_k]
        else:
            for c in cfg.selected_class_names:
                if c not in available_classes:
                    raise ValueError(f"Class '{c}' not found in dataset. Available example: {available_classes[:5]} ...")
            selected_class_names = list(cfg.selected_class_names)

        orig_indices = [base.class_to_idx[c] for c in selected_class_names]
        orig_idx_to_new = {orig: i for i, orig in enumerate(orig_indices)}

        # Build stratified indices
        g = torch.Generator()
        g.manual_seed(cfg.seed)
        train_idx: List[int] = []
        val_idx: List[int] = []
        for orig in orig_indices:
            cls_indices = [i for i in range(len(base.samples)) if int(base.samples[i][1]) == int(orig)]
            # deterministic shuffle
            perm = torch.randperm(len(cls_indices), generator=g).tolist()
            cls_indices = [cls_indices[p] for p in perm]
            n_val = max(1, int(round(len(cls_indices) * cfg.val_split))) if len(cls_indices) > 0 else 0
            val_idx.extend(cls_indices[:n_val])
            train_idx.extend(cls_indices[n_val:])

        train_subset = RemappedSubset(base, train_idx, orig_idx_to_new)
        val_subset = RemappedSubset(base, val_idx, orig_idx_to_new)
        return train_subset, val_subset, orig_idx_to_new, selected_class_names


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 4, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)


def filter_dataset_by_new_label(dataset: Dataset, label: int) -> Subset:
    indices = [i for i in range(len(dataset)) if int(dataset[i][1]) == int(label)]
    return Subset(dataset, indices)


def _tiny_imagenet_fix_val_structure(tiny_root: str) -> None:
    """Reorganize Tiny ImageNet val/ into class subfolders based on val_annotations.txt."""
    val_dir = os.path.join(tiny_root, "val")
    images_dir = os.path.join(val_dir, "images")
    ann_path = os.path.join(val_dir, "val_annotations.txt")
    if not os.path.exists(ann_path):
        return
    # Read annotations: filename, class_id, x, y, w, h
    with open(ann_path, "r", encoding="utf-8") as f:
        lines = [l.strip().split("\t") for l in f.readlines() if l.strip()]
    for parts in lines:
        if len(parts) < 2:
            continue
        fname, cls = parts[0], parts[1]
        cls_dir = os.path.join(val_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        src = os.path.join(images_dir, fname)
        dst = os.path.join(cls_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
    # Remove images directory after moving
    if os.path.isdir(images_dir) and len(os.listdir(images_dir)) == 0:
        shutil.rmtree(images_dir, ignore_errors=True)


def download_and_prepare_tiny_imagenet(dest_root: str) -> str:
    """
    Downloads Tiny ImageNet (200 classes, 64x64) and prepares train/val structure.
    Returns the dataset root path that contains train/ and val/.
    """
    os.makedirs(dest_root, exist_ok=True)
    tiny_root = os.path.join(dest_root, "tiny-imagenet-200")
    if os.path.isdir(os.path.join(tiny_root, "train")) and os.path.isdir(os.path.join(tiny_root, "val")):
        _tiny_imagenet_fix_val_structure(tiny_root)
        return tiny_root

    urls = [
        "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        # Fallback mirrors (may not always be available):
        "https://github.com/fastai/imagenette/releases/download/various/tiny-imagenet-200.zip",
    ]

    last_err: Exception | None = None
    for url in urls:
        try:
            download_and_extract_archive(url, download_root=dest_root, extract_root=dest_root, remove_finished=True)
            break
        except Exception as e:
            last_err = e
            continue

    if not os.path.isdir(tiny_root):
        raise RuntimeError(f"Failed to download Tiny ImageNet. Last error: {last_err}")

    _tiny_imagenet_fix_val_structure(tiny_root)
    return tiny_root


