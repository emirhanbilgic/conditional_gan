from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        nn.init.ones_(self.embed.weight[:, :num_features])
        nn.init.zeros_(self.embed.weight[:, num_features:])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, dim=1)
        gamma = gamma.view(-1, out.size(1), 1, 1)
        beta = beta.view(-1, out.size(1), 1, 1)
        return gamma * out + beta


class Generator(nn.Module):
    def __init__(self, z_dim: int, num_classes: int, img_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(z_dim + num_classes, base_channels * 8 * 4 * 4)

        self.cb1 = ConditionalBatchNorm2d(base_channels * 8, num_classes)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
        )
        self.cb2 = ConditionalBatchNorm2d(base_channels * 4, num_classes)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
        )
        self.cb3 = ConditionalBatchNorm2d(base_channels * 2, num_classes)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
        )
        self.cb4 = ConditionalBatchNorm2d(base_channels, num_classes)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.act = nn.ReLU(True)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()
        h = torch.cat([z, one_hot], dim=1)
        h = self.fc(h)
        h = h.view(h.size(0), -1, 4, 4)
        h = self.act(self.cb1(h, y))
        h = self.up1(h)
        h = self.act(self.cb2(h, y))
        h = self.up2(h)
        h = self.act(self.cb3(h, y))
        h = self.up3(h)
        h = self.act(self.cb4(h, y))
        x = self.up4(h)
        return x


class Discriminator(nn.Module):
    def __init__(self, num_classes: int, img_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.num_classes = num_classes
        # Projection discriminator
        self.features = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.conv_out = nn.Conv2d(base_channels * 8, 1, 4, 1, 0)
        self.embed = nn.Embedding(num_classes, base_channels * 8)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        out = self.conv_out(h).view(x.size(0), -1)
        h_flat = torch.sum(h, dim=(2, 3))
        proj = torch.sum(self.embed(y) * h_flat, dim=1, keepdim=True)
        return out + proj


def weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, 'bias', None) is not None and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        # Only initialize if affine parameters exist
        if m.affine and (m.weight is not None) and (m.bias is not None):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def build_models(z_dim: int, num_classes: int, img_channels: int = 3, base_channels: int = 64) -> Tuple[Generator, Discriminator]:
    G = Generator(z_dim=z_dim, num_classes=num_classes, img_channels=img_channels, base_channels=base_channels)
    D = Discriminator(num_classes=num_classes, img_channels=img_channels, base_channels=base_channels)
    G.apply(weights_init)
    D.apply(weights_init)
    return G, D


