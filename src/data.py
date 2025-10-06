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
class CarsSubsetConfig:
    data_dir: str
    selected_classes: List[int]
    img_size: int = 64
    download: bool = True


class RemappedSubset(Dataset):
    def __init__(self, base: Dataset, indices: List[int], orig_to_new: Dict[int, int]):
        self.base = base
        self.indices = indices
        self.orig_to_new = orig_to_new

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        img, orig_label = self.base[base_idx]
        new_label = self.orig_to_new[int(orig_label)]
        return img, torch.tensor(new_label, dtype=torch.long)


def _filter_indices_by_classes(dataset: Dataset, classes: List[int]) -> List[int]:
    keep = set(int(c) for c in classes)
    indices: List[int] = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        if int(y) in keep:
            indices.append(i)
    return indices


def load_cars_datasets(cfg: CarsSubsetConfig) -> Tuple[Dataset, Dataset, Dict[int, int]]:
    """
    Returns:
      train_subset: labels remapped to [0..k-1]
      test_subset: labels remapped to [0..k-1]
      orig_to_new: mapping original_label -> new_label
    """
    root = os.path.join(cfg.data_dir, "stanford_cars")
    tfm = build_transforms(cfg.img_size)
    train_base = datasets.StanfordCars(root=root, split="train", download=cfg.download, transform=tfm)
    test_base = datasets.StanfordCars(root=root, split="test", download=cfg.download, transform=tfm)

    selected = list(cfg.selected_classes)
    orig_to_new = {orig: i for i, orig in enumerate(selected)}

    train_idx = _filter_indices_by_classes(train_base, selected)
    test_idx = _filter_indices_by_classes(test_base, selected)

    train_subset = RemappedSubset(train_base, train_idx, orig_to_new)
    test_subset = RemappedSubset(test_base, test_idx, orig_to_new)
    return train_subset, test_subset, orig_to_new


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 4, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)


def filter_dataset_by_new_label(dataset: Dataset, label: int) -> Subset:
    indices = [i for i in range(len(dataset)) if int(dataset[i][1]) == int(label)]
    return Subset(dataset, indices)


