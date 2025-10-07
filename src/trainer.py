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
from torch.cuda.amp import GradScaler, autocast

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
    num_workers: int = 2
    pin_memory: bool = True
    sample_grid: int = 32
    sample_interval: int = 2000  # steps; set <=0 to disable
    keep_last: int = 3  # checkpoints to keep; set <=0 to keep all
    save_optimizer: bool = False
    seed: int = 42
    device: str = "cuda"
    amp: bool = True


class CGANTrainer:
    def __init__(self, train_dataset: Dataset, num_classes: int, cfg: TrainConfig):
        set_seed(cfg.seed)
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and self.device.type == "cuda",
            drop_last=True,
        )
        self.num_classes = num_classes
        self.G, self.D = build_models(z_dim=cfg.z_dim, num_classes=num_classes)
        self.G.to(self.device)
        self.D.to(self.device)

        self.opt_g = optim.Adam(self.G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
        self.opt_d = optim.Adam(self.D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

        self.criterion = nn.BCEWithLogitsLoss()

        # Mixed precision scaler (enabled only if CUDA and cfg.amp)
        self.use_amp = bool(cfg.amp and torch.cuda.is_available())
        self.scaler = GradScaler(enabled=self.use_amp)

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
                with autocast(enabled=self.use_amp):
                    z = sample_noise(bsz, self.cfg.z_dim, self.device)
                    fake = self.G(z, y).detach()
                    pred_real = self.D(real, y)
                    pred_fake = self.D(fake, y)
                    loss_d = self.criterion(pred_real, torch.ones_like(pred_real)) + \
                             self.criterion(pred_fake, torch.zeros_like(pred_fake))
                self.scaler.scale(loss_d).backward()
                self.scaler.step(self.opt_d)
                self.scaler.update()

                # Train G
                self.opt_g.zero_grad(set_to_none=True)
                with autocast(enabled=self.use_amp):
                    z = sample_noise(bsz, self.cfg.z_dim, self.device)
                    fake = self.G(z, y)
                    pred_fake = self.D(fake, y)
                    loss_g = self.criterion(pred_fake, torch.ones_like(pred_fake))
                self.scaler.scale(loss_g).backward()
                self.scaler.step(self.opt_g)
                self.scaler.update()

                if self.cfg.sample_interval > 0 and (step % self.cfg.sample_interval == 0):
                    self._save_samples(step)

                pbar.set_postfix({"loss_d": f"{loss_d.item():.3f}", "loss_g": f"{loss_g.item():.3f}"})
                step += 1

            # Save checkpoint per epoch
            state = {
                "G": self.G.state_dict(),
                "D": self.D.state_dict(),
                "epoch": epoch,
            }
            if self.cfg.save_optimizer:
                state["opt_g"] = self.opt_g.state_dict()
                state["opt_d"] = self.opt_d.state_dict()
            ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch+1:03d}.pt")
            save_checkpoint(state, ckpt_path)

            # Rotate old checkpoints to save disk
            if self.cfg.keep_last and self.cfg.keep_last > 0:
                try:
                    files = sorted(
                        [f for f in os.listdir(self.ckpt_dir) if f.startswith("epoch_") and f.endswith(".pt")]
                    )
                    excess = len(files) - self.cfg.keep_last
                    for i in range(max(0, excess)):
                        try:
                            os.remove(os.path.join(self.ckpt_dir, files[i]))
                        except OSError:
                            pass
                except Exception:
                    pass

    def finetune_excluding_class(self, dataset_excluding_class: Dataset, epochs: int) -> None:
        """Unlearn a class by fine-tuning on data without that class.
        We continue training but the dataloader contains no samples of that class.
        """
        loader = DataLoader(
            dataset_excluding_class,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory and self.device.type == "cuda",
            drop_last=True,
        )
        step = 0
        for epoch in range(epochs):
            pbar = tqdm(loader, desc=f"Unlearn Epoch {epoch+1}/{epochs}")
            for real, y in pbar:
                real = real.to(self.device)
                y = y.to(self.device)
                bsz = real.size(0)

                # D
                self.opt_d.zero_grad(set_to_none=True)
                with autocast(enabled=self.use_amp):
                    z = sample_noise(bsz, self.cfg.z_dim, self.device)
                    fake = self.G(z, y).detach()
                    pred_real = self.D(real, y)
                    pred_fake = self.D(fake, y)
                    loss_d = self.criterion(pred_real, torch.ones_like(pred_real)) + \
                             self.criterion(pred_fake, torch.zeros_like(pred_fake))
                self.scaler.scale(loss_d).backward()
                self.scaler.step(self.opt_d)
                self.scaler.update()

                # G
                self.opt_g.zero_grad(set_to_none=True)
                with autocast(enabled=self.use_amp):
                    z = sample_noise(bsz, self.cfg.z_dim, self.device)
                    fake = self.G(z, y)
                    pred_fake = self.D(fake, y)
                    loss_g = self.criterion(pred_fake, torch.ones_like(pred_fake))
                self.scaler.scale(loss_g).backward()
                self.scaler.step(self.opt_g)
                self.scaler.update()

                if self.cfg.sample_interval > 0 and (step % self.cfg.sample_interval == 0):
                    self._save_samples(step)

                pbar.set_postfix({"loss_d": f"{loss_d.item():.3f}", "loss_g": f"{loss_g.item():.3f}"})
                step += 1


    def unlearn_with_fisher(self, train_ds: Dataset, excluded_label: int, z_dim: int, max_batches: int = 100) -> None:
        """Unlearn a class by Fisher pruning without further fine-tuning.

        We compute Fisher information diagonals on two subsets: forgotten (Df) and retained (Dr).
        For each parameter, if Df/Dr > 2, we prune (zero) that weight. Otherwise we leave it untouched.
        If no weights exceed the threshold, we do nothing else.
        """
        device = self.device

        # Build subsets
        forgotten_indices = [i for i in range(len(train_ds)) if int(train_ds[i][1]) == int(excluded_label)]
        retained_indices = [i for i in range(len(train_ds)) if int(train_ds[i][1]) != int(excluded_label)]

        if len(forgotten_indices) == 0 or len(retained_indices) == 0:
            # Nothing to do if a subset is empty
            return

        forgotten_loader = DataLoader(Subset(train_ds, forgotten_indices), batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True, drop_last=True)
        retained_loader = DataLoader(Subset(train_ds, retained_indices), batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True, drop_last=True)

        def _zeros_like_named_params(module: nn.Module) -> Dict[str, torch.Tensor]:
            out: Dict[str, torch.Tensor] = {}
            for name, param in module.named_parameters():
                if param.requires_grad:
                    out[name] = torch.zeros_like(param.data, device=param.device)
            return out

        def _accumulate(module: nn.Module, accum: Dict[str, torch.Tensor]) -> None:
            for name, param in module.named_parameters():
                if (not param.requires_grad) or (param.grad is None):
                    continue
                accum[name] = accum[name] + (param.grad.detach() ** 2)

        def _normalize(accum: Dict[str, torch.Tensor], denom: float) -> Dict[str, torch.Tensor]:
            eps = 1e-8
            return {k: v / max(denom, eps) for k, v in accum.items()}

        def _calculate_fim_diagonal(model: nn.Module, loader: DataLoader, compute_loss_fn) -> Dict[str, torch.Tensor]:
            """Batchwise squared-gradient Fisher approximation for a single model.

            Sets model to eval mode, averages squared grads over batches.
            compute_loss_fn(real, y) should return a scalar loss that depends on `model`.
            """
            model_was_training = model.training
            model.eval()
            fim_diag = _zeros_like_named_params(model)
            batches_processed = 0

            if len(loader.dataset) == 0:
                return fim_diag

            for real, y in loader:
                if batches_processed >= max_batches:
                    break
                real = real.to(device)
                y = y.to(device)

                # zero grads of all involved modules to avoid cross-contamination
                self.G.zero_grad(set_to_none=True)
                self.D.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)

                loss = compute_loss_fn(real, y)
                loss.backward()

                _accumulate(model, fim_diag)
                batches_processed += 1

            denom = float(max(1, batches_processed))
            fim_diag = _normalize(fim_diag, denom)

            if model_was_training:
                model.train()
            return fim_diag

        def _compute_fisher_for_loader(loader: DataLoader) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
            # Discriminator Fisher: BCE on real vs fake, with fake generated without backprop to G
            def d_loss(real: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                bsz = real.size(0)
                with torch.no_grad():
                    z = sample_noise(bsz, self.cfg.z_dim, device)
                    fake = self.G(z, y)
                pred_real = self.D(real, y)
                pred_fake = self.D(fake, y)
                return self.criterion(pred_real, torch.ones_like(pred_real)) + \
                       self.criterion(pred_fake, torch.zeros_like(pred_fake))

            # Generator Fisher: BCE on D(G(z,y)) vs ones; D participates in forward but only G grads are accumulated
            def g_loss(real_unused: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                bsz = y.size(0)
                z = sample_noise(bsz, self.cfg.z_dim, device)
                fake = self.G(z, y)
                pred_fake = self.D(fake, y)
                return self.criterion(pred_fake, torch.ones_like(pred_fake))

            fisher_d = _calculate_fim_diagonal(self.D, loader, d_loss)
            fisher_g = _calculate_fim_diagonal(self.G, loader, g_loss)
            return fisher_g, fisher_d

        fisher_g_forgot, fisher_d_forgot = _compute_fisher_for_loader(forgotten_loader)
        fisher_g_retain, fisher_d_retain = _compute_fisher_for_loader(retained_loader)

        def _prune_by_ratio(module: nn.Module, fisher_forgot: Dict[str, torch.Tensor], fisher_retain: Dict[str, torch.Tensor], threshold: float = 10.0) -> int:
            eps = 1e-8
            num_pruned = 0
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                ff = fisher_forgot[name]
                fr = fisher_retain[name]
                ratio = ff / (fr + eps)
                mask = ratio > threshold
                if mask.any():
                    with torch.no_grad():
                        num_pruned += int(mask.sum().item())
                        param.data[mask] = 0.0
            return num_pruned

        pruned_g = _prune_by_ratio(self.G, fisher_g_forgot, fisher_g_retain)
        pruned_d = _prune_by_ratio(self.D, fisher_d_forgot, fisher_d_retain)

        # No finetuning afterward per requirement.
        # Optional: could log counts
        print(f"Fisher pruning completed: G pruned elements={pruned_g}, D pruned elements={pruned_d}")

