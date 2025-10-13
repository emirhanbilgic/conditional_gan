from __future__ import annotations

import torch
from typing import Callable


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

    try:
        if resolution is None:
            G = make_gan(gan_type=gan_type)
        else:
            G = make_gan(gan_type=gan_type, resolution=resolution)
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


