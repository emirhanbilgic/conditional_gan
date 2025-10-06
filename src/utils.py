import os
import json
import random
from typing import Dict, Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def sample_noise(batch_size: int, z_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, z_dim, device=device)


def make_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()


def write_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


