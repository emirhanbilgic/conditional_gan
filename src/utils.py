import os
import json
import random
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn



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



class GradCAM:
    """Minimal Grad-CAM for CNN discriminators.

    This implementation attaches hooks to a target convolutional layer,
    records its activations and gradients during a backward pass of a
    scalar score, and computes a class-agnostic heatmap.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device | str = "cpu"):
        self.model = model
        self.target_layer = target_layer
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        self._activations = None
        self._grads = None

        def fwd_hook(module, inp, out):
            self._activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._grads = grad_out[0].detach()

        # register hooks
        self._fwd_handle = self.target_layer.register_forward_hook(fwd_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(bwd_hook)

    def remove_hooks(self) -> None:
        try:
            self._fwd_handle.remove()
        except Exception:
            pass
        try:
            self._bwd_handle.remove()
        except Exception:
            pass

    def compute_heatmap(self, score: torch.Tensor) -> torch.Tensor:
        """Compute Grad-CAM heatmap.

        Args:
            score: scalar tensor to backpropagate (sum over batch for convenience).

        Returns:
            heatmap tensor of shape (B, 1, H, W) normalized to [0, 1].
        """
        if score.dim() > 0:
            score = score.sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        assert self._activations is not None and self._grads is not None, "Hooks did not capture activations/gradients"
        # Global-average-pool gradients to get channel weights
        weights = torch.mean(self._grads, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = torch.sum(weights * self._activations, dim=1, keepdim=True)  # (B, 1, H, W)
        cam = torch.relu(cam)
        # Normalize per-sample
        b, _, h, w = cam.shape
        cam = cam.view(b, -1)
        eps = 1e-8
        cam = (cam - cam.min(dim=1, keepdim=True).values) / (cam.max(dim=1, keepdim=True).values - cam.min(dim=1, keepdim=True).values + eps)
        cam = cam.view(b, 1, h, w)
        return cam


def overlay_heatmap_on_images(images: torch.Tensor, heatmaps: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Overlay single-channel heatmaps onto RGB images.

    Args:
        images: tensor (B, 3, H, W) in [-1, 1] or [0, 1]
        heatmaps: tensor (B, 1, H, W) in [0, 1]
        alpha: blending factor for heatmap overlay

    Returns:
        overlaid images in [0, 1]
    """
    # Bring images to [0,1]
    imgs = images.clone()
    if imgs.min() < 0:
        imgs = (imgs + 1) / 2
    imgs = imgs.clamp(0, 1)

    hm = heatmaps.clamp(0, 1)
    hm_rgb = torch.cat([hm, torch.zeros_like(hm), 1 - hm], dim=1)  # red-blue map
    overlaid = (1 - alpha) * imgs + alpha * hm_rgb
    return overlaid.clamp(0, 1)

