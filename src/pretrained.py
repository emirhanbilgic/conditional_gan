from __future__ import annotations

import torch
from typing import Callable, Dict, List, Tuple
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


def load_pretrained_gan(gan_type: str = "biggan", resolution: int | None = None):
    """Load a pretrained GAN via pytorch-pretrained-gans.

    Returns a callable generator module `G` with methods `sample_class` and `sample_latent` if available.
    """
    try:
        from pytorch_pretrained_gans import make_gan
    except Exception as e:
        raise ImportError(
            "pytorch_pretrained_gans is not installed. Run: \n"
            "pip install --no-cache-dir git+https://github.com/lukemelas/pytorch-pretrained-gans\n"
            "If on Kaggle, prefix with: pip install -q --no-deps --no-cache-dir git+https://github.com/lukemelas/pytorch-pretrained-gans"
        ) from e

    kwargs = {"gan_type": gan_type}
    if resolution is not None:
        kwargs["resolution"] = resolution
    # BigGAN interfaces in some versions don't accept 'resolution' kwarg
    if "biggan" in str(gan_type).lower():
        kwargs.pop("resolution", None)
    try:
        G = make_gan(**kwargs)
    except TypeError as te:
        # Some installed versions do not accept 'resolution'; retry without it
        if "resolution" in kwargs:
            kwargs.pop("resolution", None)
            G = make_gan(**kwargs)
        else:
            raise
    except NotImplementedError as nie:
        if gan_type in {"self_conditioned", "self-conditioned", "scgan"}:
            raise NotImplementedError(
                "The installed pytorch_pretrained_gans does not expose 'self_conditioned' via make_gan. "
                "Please use a supported type like: --pretrained_gan_type biggan"
            ) from nie
        raise
    return G


def make_conditional_sampler(G, device: torch.device) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Wrap a pretrained conditional GAN `G` to a sampler(z, y) -> x in [-1,1].

    - z: (B, z_dim)
    - y: (B,) long class indices or one-hot depending on G.
    """
    G = G.to(device).eval()

    # If model exposes helper methods to sample class/latent, we still accept external z,y
    @torch.no_grad()
    def sampler(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Many pretrained interfaces accept integer class labels for y
        # Convert to the expected format if G expects one-hot
        try:
            x = G(z=z, y=y)
        except Exception:
            # Attempt to convert to one-hot if needed
            num_classes = getattr(G, "num_classes", None)
            if num_classes is None:
                raise
            y_one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes).float().to(z.device)
            x = G(z=z, y=y_one_hot)
        return x

    return sampler


def make_class_latent_samplers(G, z_dim: int, device: torch.device):
    """Return helper callables to sample y and z from the pretrained GAN if exposed; else fall back."""
    @torch.no_grad()
    def sample_y(batch_size: int) -> torch.Tensor:
        if hasattr(G, "sample_class"):
            y = G.sample_class(batch_size=batch_size)
            # Some implementations return one-hot; convert to indices if needed
            if y.dim() == 2:
                y = torch.argmax(y, dim=1)
            return y.to(device)
        # Fallback: uniform classes from [0, num_classes)
        num_classes = getattr(G, "num_classes", 1000)
        return torch.randint(0, num_classes, (batch_size,), device=device)

    @torch.no_grad()
    def sample_z(batch_size: int) -> torch.Tensor:
        if hasattr(G, "sample_latent"):
            z = G.sample_latent(batch_size=batch_size)
            return z.to(device)
        return torch.randn(batch_size, z_dim, device=device)

    return sample_y, sample_z



def _build_imagenet_classifier(device: torch.device) -> nn.Module:
    """Load a pretrained ImageNet classifier for guidance and Grad-CAM.

    Uses torchvision ResNet-50 with pretrained weights.
    """
    try:
        from torchvision.models import ResNet50_Weights  # type: ignore
        clf = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        clf = models.resnet50(pretrained=True)
    clf.eval()
    return clf.to(device)


def _imagenet_preprocess(x: torch.Tensor) -> torch.Tensor:
    """Resize to 224 and normalize to ImageNet stats.

    Expects input in [-1, 1] or [0, 1]; outputs normalized tensor.
    """
    # Bring to [0,1]
    imgs = x
    if imgs.min() < 0:
        imgs = (imgs + 1) / 2
    imgs = imgs.clamp(0, 1)
    # Resize
    imgs = torch.nn.functional.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
    imgs = (imgs - mean) / std
    return imgs


def fisher_prune_generator_with_classifier(
    G: nn.Module,
    forgotten_biggan_label: int,
    retained_biggan_labels: List[int],
    z_dim: int,
    device: torch.device,
    max_batches: int = 100,
    batch_size: int = 32,
    threshold: float = 15.0,
) -> Tuple[int, int]:
    """Unlearn a BigGAN class by Fisher pruning using a pretrained ImageNet classifier.

    Computes generator Fisher diagonals on two sets: forgotten vs retained labels, using
    cross-entropy on a fixed pretrained classifier applied to generated images.
    Prunes parameters where ratio(F_forgotten / F_retained) > threshold.

    Returns: (num_pruned_elements)
    """
    clf = _build_imagenet_classifier(device)
    for p in clf.parameters():
        p.requires_grad_(False)
    G = G.to(device)
    G.train()  # ensure gradients can flow

    def _zeros_like_named_params(module: nn.Module) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, p in module.named_parameters():
            if p.requires_grad:
                out[name] = torch.zeros_like(p.data, device=p.device)
        return out

    def _accumulate(module: nn.Module, accum: Dict[str, torch.Tensor]) -> None:
        for name, p in module.named_parameters():
            if (not p.requires_grad) or (p.grad is None):
                continue
            accum[name] = accum[name] + (p.grad.detach() ** 2)

    def _normalize(accum: Dict[str, torch.Tensor], denom: float) -> Dict[str, torch.Tensor]:
        eps = 1e-8
        return {k: v / max(denom, eps) for k, v in accum.items()}

    def _calc_fisher_for_labels(labels: List[int]) -> Dict[str, torch.Tensor]:
        accum = _zeros_like_named_params(G)
        batches = 0
        if len(labels) == 0:
            return accum
        while batches < max_batches:
            bsz = batch_size
            # Process in micro-batches to reduce peak memory
            mbsz = max(1, min(4, bsz))
            num_micro = (bsz + mbsz - 1) // mbsz
            for m in range(num_micro):
                cur = min(mbsz, bsz - m * mbsz)
                # Sample uniform classes from the given set
                y = torch.tensor([labels[i % len(labels)] for i in range(cur)], device=device, dtype=torch.long)
                z = torch.randn(cur, z_dim, device=device)
                G.zero_grad(set_to_none=True)
                # forward through G and classifier with optional autocast
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    x = G(z=z, y=y)
                    logits = clf(_imagenet_preprocess(x))
                    loss = F.cross_entropy(logits, y)
                loss.backward()
                _accumulate(G, accum)
                # Release activations early
                del x, logits, loss, z, y
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            batches += 1
        return _normalize(accum, float(max(1, batches)))

    fisher_forgot = _calc_fisher_for_labels([forgotten_biggan_label])
    fisher_retain = _calc_fisher_for_labels(retained_biggan_labels)

    # Prune by ratio
    eps = 1e-8
    num_pruned = 0
    for name, p in G.named_parameters():
        if not p.requires_grad:
            continue
        ff = fisher_forgot[name]
        fr = fisher_retain[name]
        ratio = ff / (fr + eps)
        mask = ratio > threshold
        if mask.any():
            with torch.no_grad():
                num_pruned += int(mask.sum().item())
                p.data[mask] = 0.0

    return num_pruned, 0


def ssd_dampen_generator_with_classifier(
    G: nn.Module,
    forgotten_biggan_label: int,
    retained_biggan_labels: List[int],
    z_dim: int,
    device: torch.device,
    *,
    max_batches: int = 100,
    batch_size: int = 32,
    lower_bound: float = 1.0,
    exponent: float = 1.0,
    dampening_constant: float = 0.5,
    selection_weighting: float = 10.0,
) -> int:
    """Selective Synaptic Dampening (SSD) on a pretrained conditional generator using a fixed classifier.

    Computes squared-gradient importances on forgotten vs retained label sets using an ImageNet classifier
    applied to generated images. Multiplies parameters by a dampening factor when forget-importance dominates.
    """
    clf = _build_imagenet_classifier(device)
    for p in clf.parameters():
        p.requires_grad_(False)
    G = G.to(device)
    G.train()

    def _zeros_like_named_params(module: nn.Module) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, p in module.named_parameters():
            if p.requires_grad:
                out[name] = torch.zeros_like(p.data, device=p.device)
        return out

    def _accumulate(module: nn.Module, accum: Dict[str, torch.Tensor]) -> None:
        for name, p in module.named_parameters():
            if (not p.requires_grad) or (p.grad is None):
                continue
            accum[name] = accum[name] + (p.grad.detach() ** 2)

    def _normalize(accum: Dict[str, torch.Tensor], denom: float) -> Dict[str, torch.Tensor]:
        eps = 1e-8
        return {k: v / max(denom, eps) for k, v in accum.items()}

    def _calc_importance_for_labels(labels: List[int]) -> Dict[str, torch.Tensor]:
        accum = _zeros_like_named_params(G)
        batches = 0
        if len(labels) == 0:
            return accum
        while batches < max_batches:
            bsz = batch_size
            # micro-batch
            mbsz = max(1, min(4, bsz))
            num_micro = (bsz + mbsz - 1) // mbsz
            for m in range(num_micro):
                cur = min(mbsz, bsz - m * mbsz)
                y = torch.tensor([labels[i % len(labels)] for i in range(cur)], device=device, dtype=torch.long)
                z = torch.randn(cur, z_dim, device=device)
                G.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    x = G(z=z, y=y)
                    logits = clf(_imagenet_preprocess(x))
                    loss = F.cross_entropy(logits, y)
                loss.backward()
                _accumulate(G, accum)
                del x, logits, loss, z, y
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            batches += 1
        return _normalize(accum, float(max(1, batches)))

    imp_forget = _calc_importance_for_labels([forgotten_biggan_label])
    imp_retain = _calc_importance_for_labels(retained_biggan_labels)

    # Apply dampening
    num_dampened = 0
    with torch.no_grad():
        for name, p in G.named_parameters():
            if not p.requires_grad:
                continue
            oimp = imp_retain.get(name)
            fimp = imp_forget.get(name)
            if oimp is None or fimp is None:
                continue
            mask = fimp > (oimp * selection_weighting)
            if not mask.any():
                continue
            weight = ((oimp * dampening_constant) / (fimp + 1e-9)).pow(exponent)
            update = weight[mask]
            update[update > lower_bound] = lower_bound
            p.data[mask] = p.data[mask] * update
            num_dampened += int(mask.sum().item())

    return num_dampened
