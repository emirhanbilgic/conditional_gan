from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


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
    if not os.path.isdir(train_root) or not os.path.isdir(val_root):
        raise FileNotFoundError(f"Expect ImageNet-like folders at {cfg.data_dir}/train and {cfg.data_dir}/val")

    tfm = build_transforms(cfg.img_size)
    train_base = datasets.ImageFolder(root=train_root, transform=tfm)
    val_base = datasets.ImageFolder(root=val_root, transform=tfm)

    available_classes = list(train_base.class_to_idx.keys())
    # ensure val has same classes
    if set(available_classes) != set(val_base.class_to_idx.keys()):
        raise ValueError("Train and val class sets differ; please align your dataset.")

    if cfg.selected_class_names is None or len(cfg.selected_class_names) == 0:
        selected_class_names = sorted(available_classes)[: cfg.auto_select_k]
    else:
        for c in cfg.selected_class_names:
            if c not in available_classes:
                raise ValueError(f"Class '{c}' not found in dataset. Available example: {available_classes[:5]} ...")
        selected_class_names = list(cfg.selected_class_names)

    # Map original ImageFolder indices to new consecutive indices [0..k-1]
    orig_indices = [train_base.class_to_idx[c] for c in selected_class_names]
    orig_idx_to_new = {orig: i for i, orig in enumerate(orig_indices)}

    train_idx = _filter_indices_by_orig_indices(train_base, orig_indices)
    val_idx = _filter_indices_by_orig_indices(val_base, orig_indices)

    train_subset = RemappedSubset(train_base, train_idx, orig_idx_to_new)
    val_subset = RemappedSubset(val_base, val_idx, orig_idx_to_new)
    return train_subset, val_subset, orig_idx_to_new, selected_class_names


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 4, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)


def filter_dataset_by_new_label(dataset: Dataset, label: int) -> Subset:
    indices = [i for i in range(len(dataset)) if int(dataset[i][1]) == int(label)]
    return Subset(dataset, indices)


