from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from .models import build_models, Generator, Discriminator
from .utils import ensure_dir, save_checkpoint, sample_noise, set_seed


@dataclass
class TrainConfig:
    work_dir: str
    img_size: int = 64
    z_dim: int = 128
    batch_size: int = 128
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    epochs: int = 50
    num_workers: int = 4
    sample_grid: int = 64
    seed: int = 42
    device: str = "cuda"


class CGANTrainer:
    def __init__(self, train_dataset: Dataset, num_classes: int, cfg: TrainConfig):
        set_seed(cfg.seed)
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        self.num_classes = num_classes
        self.G, self.D = build_models(z_dim=cfg.z_dim, num_classes=num_classes)
        self.G.to(self.device)
        self.D.to(self.device)

        self.opt_g = optim.Adam(self.G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
        self.opt_d = optim.Adam(self.D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

        self.criterion = nn.BCEWithLogitsLoss()

        self.ckpt_dir = os.path.join(cfg.work_dir, "checkpoints")
        self.sample_dir = os.path.join(cfg.work_dir, "samples")
        ensure_dir(self.ckpt_dir)
        ensure_dir(self.sample_dir)

    def _save_samples(self, step: int) -> None:
        self.G.eval()
        with torch.no_grad():
            grid_imgs = []
            per_class = min(8, self.cfg.sample_grid // self.num_classes)
            for cls in range(self.num_classes):
                z = sample_noise(per_class, self.cfg.z_dim, self.device)
                y = torch.full((per_class,), cls, dtype=torch.long, device=self.device)
                x = self.G(z, y)
                grid_imgs.append(x.cpu())
            imgs = torch.cat(grid_imgs, dim=0)
            grid = make_grid(imgs, nrow=per_class, normalize=True, value_range=(-1, 1))
            save_image(grid, os.path.join(self.sample_dir, f"step_{step:06d}.png"))
        self.G.train()

    def train(self) -> None:
        step = 0
        for epoch in range(self.cfg.epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs}")
            for real, y in pbar:
                real = real.to(self.device)
                y = y.to(self.device)
                bsz = real.size(0)

                # Train D
                self.opt_d.zero_grad(set_to_none=True)
                z = sample_noise(bsz, self.cfg.z_dim, self.device)
                fake = self.G(z, y).detach()
                pred_real = self.D(real, y)
                pred_fake = self.D(fake, y)
                loss_d = self.criterion(pred_real, torch.ones_like(pred_real)) + \
                         self.criterion(pred_fake, torch.zeros_like(pred_fake))
                loss_d.backward()
                self.opt_d.step()

                # Train G
                self.opt_g.zero_grad(set_to_none=True)
                z = sample_noise(bsz, self.cfg.z_dim, self.device)
                fake = self.G(z, y)
                pred_fake = self.D(fake, y)
                loss_g = self.criterion(pred_fake, torch.ones_like(pred_fake))
                loss_g.backward()
                self.opt_g.step()

                if step % 500 == 0:
                    self._save_samples(step)

                pbar.set_postfix({"loss_d": f"{loss_d.item():.3f}", "loss_g": f"{loss_g.item():.3f}"})
                step += 1

            # Save checkpoint per epoch
            save_checkpoint({
                "G": self.G.state_dict(),
                "D": self.D.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt_d": self.opt_d.state_dict(),
                "epoch": epoch
            }, os.path.join(self.ckpt_dir, f"epoch_{epoch+1:03d}.pt"))

    def finetune_excluding_class(self, dataset_excluding_class: Dataset, epochs: int) -> None:
        """Unlearn a class by fine-tuning on data without that class.
        We continue training but the dataloader contains no samples of that class.
        """
        loader = DataLoader(dataset_excluding_class, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True, drop_last=True)
        step = 0
        for epoch in range(epochs):
            pbar = tqdm(loader, desc=f"Unlearn Epoch {epoch+1}/{epochs}")
            for real, y in pbar:
                real = real.to(self.device)
                y = y.to(self.device)
                bsz = real.size(0)

                # D
                self.opt_d.zero_grad(set_to_none=True)
                z = sample_noise(bsz, self.cfg.z_dim, self.device)
                fake = self.G(z, y).detach()
                pred_real = self.D(real, y)
                pred_fake = self.D(fake, y)
                loss_d = self.criterion(pred_real, torch.ones_like(pred_real)) + \
                         self.criterion(pred_fake, torch.zeros_like(pred_fake))
                loss_d.backward()
                self.opt_d.step()

                # G
                self.opt_g.zero_grad(set_to_none=True)
                z = sample_noise(bsz, self.cfg.z_dim, self.device)
                fake = self.G(z, y)
                pred_fake = self.D(fake, y)
                loss_g = self.criterion(pred_fake, torch.ones_like(pred_fake))
                loss_g.backward()
                self.opt_g.step()

                if step % 500 == 0:
                    self._save_samples(step)

                pbar.set_postfix({"loss_d": f"{loss_d.item():.3f}", "loss_g": f"{loss_g.item():.3f}"})
                step += 1


