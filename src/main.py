from __future__ import annotations

import os
import argparse
import json
from typing import List

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from datetime import datetime

from .data import CIFAR10Config, load_cifar10_datasets
from .trainer import TrainConfig, CGANTrainer
from .fid import FIDConfig, FIDEvaluator
from .utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conditional GAN on CIFAR-10 with Unlearning")
    p.add_argument("--data_dir", type=str, required=True, help="CIFAR-10 root directory for torchvision cache")
    p.add_argument("--work_dir", type=str, required=True)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--unlearn_epochs", type=int, default=10)
    p.add_argument("--fid_num_samples_per_class", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--unlearning_type", type=str, default="pure_finetuning", choices=["pure_finetuning", "fisher"], help="Unlearning strategy: pure_finetuning or fisher")
    p.add_argument("--samples_per_class", type=int, default=5, help="Number of samples to generate per class for before/after snapshots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.work_dir)

    # 1) Load CIFAR-10 datasets
    os.makedirs(args.data_dir, exist_ok=True)
    c_cfg = CIFAR10Config(data_dir=args.data_dir, img_size=args.img_size, train_download=True, test_download=True, seed=args.seed)
    train_ds, val_ds, _, class_names = load_cifar10_datasets(c_cfg)
    num_classes = 10

    # 2) Train CGAN
    tcfg = TrainConfig(
        work_dir=args.work_dir,
        img_size=args.img_size,
        z_dim=args.z_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )
    trainer = CGANTrainer(train_ds, num_classes=num_classes, cfg=tcfg)
    trainer.train()

    # Create a unique directory per trial to avoid overwrite across runs
    samples_root = os.path.join(args.work_dir, "samples")
    os.makedirs(samples_root, exist_ok=True)
    trial_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    trial_base = f"trial_{trial_tag}_seed{args.seed}_{args.unlearning_type}"
    trial_dir = os.path.join(samples_root, trial_base)
    if os.path.exists(trial_dir):
        # If a run starts within the same second, add a numeric suffix to avoid overwrite
        suffix = 1
        while True:
            candidate = os.path.join(samples_root, f"{trial_base}_{suffix:03d}")
            if not os.path.exists(candidate):
                trial_dir = candidate
                break
            suffix += 1
    os.makedirs(trial_dir, exist_ok=True)

    # Generate samples before unlearning: N samples per class
    trainer.G.eval()
    with torch.no_grad():
        imgs_before = []
        per_class = int(max(1, args.samples_per_class))
        before_dir = os.path.join(trial_dir, "before_unlearning")
        os.makedirs(before_dir, exist_ok=True)
        for cls in range(num_classes):
            z = torch.randn(per_class, args.z_dim, device=trainer.device)
            y = torch.full((per_class,), cls, dtype=torch.long, device=trainer.device)
            x = trainer.G(z, y).cpu()
            imgs_before.append(x)

            # Save individual images per class
            class_dir = os.path.join(before_dir, f"class_{cls}")
            os.makedirs(class_dir, exist_ok=True)
            for i in range(per_class):
                save_image(x[i], os.path.join(class_dir, f"img_{i+1}.png"), normalize=True, value_range=(-1, 1))

        imgs_before = torch.cat(imgs_before, dim=0)
        # Combined grid with nrow = samples per class, rows = classes
        grid_before = make_grid(imgs_before, nrow=per_class, normalize=True, value_range=(-1, 1))
        save_image(grid_before, os.path.join(before_dir, "grid_all_classes.png"))
    trainer.G.train()

    plt.figure(figsize=(12, 3))
    plt.axis('off')
    plt.title(f'Before Unlearning: {per_class} samples per class')
    plt.imshow(grid_before.permute(1, 2, 0).numpy())
    plt.show()

    # 3) FID before unlearning
    fcfg = FIDConfig(
        work_dir=args.work_dir,
        num_samples_per_class=args.fid_num_samples_per_class,
        batch_size=max(32, args.batch_size // 2),
        device=args.device,
    )
    fid_eval = FIDEvaluator(trainer.G, num_classes=num_classes, cfg=fcfg)
    fid_before = fid_eval.compute_fid(val_ds, z_dim=args.z_dim)
    write_json(fid_before, os.path.join(args.work_dir, "metrics", "fid_before.json"))

    # 4) Unlearn the first selected class (c1)
    c1_new = 0  # by construction, the first selected class is mapped to 0
    # filter out all samples equal to c1_new
    indices = [i for i in range(len(train_ds)) if int(train_ds[i][1]) != int(c1_new)]
    from torch.utils.data import Subset
    train_without_c1 = Subset(train_ds, indices)
    if args.unlearning_type == "pure_finetuning":
        trainer.finetune_excluding_class(train_without_c1, epochs=args.unlearn_epochs)
    else:
        # fisher-based unlearning: compute FIM ratio and prune if needed
        # When fisher is selected and Df/Dr < 2, we skip finetuning entirely per requirement
        trainer.unlearn_with_fisher(train_ds=train_ds, excluded_label=c1_new, z_dim=args.z_dim)

    # Generate samples after unlearning: N samples per class
    trainer.G.eval()
    with torch.no_grad():
        imgs_after = []
        per_class = int(max(1, args.samples_per_class))
        after_dir = os.path.join(trial_dir, "after_unlearning")
        os.makedirs(after_dir, exist_ok=True)
        for cls in range(num_classes):
            z = torch.randn(per_class, args.z_dim, device=trainer.device)
            y = torch.full((per_class,), cls, dtype=torch.long, device=trainer.device)
            x = trainer.G(z, y).cpu()
            imgs_after.append(x)

            # Save individual images per class
            class_dir = os.path.join(after_dir, f"class_{cls}")
            os.makedirs(class_dir, exist_ok=True)
            for i in range(per_class):
                save_image(x[i], os.path.join(class_dir, f"img_{i+1}.png"), normalize=True, value_range=(-1, 1))

        imgs_after = torch.cat(imgs_after, dim=0)
        grid_after = make_grid(imgs_after, nrow=per_class, normalize=True, value_range=(-1, 1))
        save_image(grid_after, os.path.join(after_dir, "grid_all_classes.png"))
    trainer.G.train()

    plt.figure(figsize=(12, 3))
    plt.axis('off')
    plt.title(f'After Unlearning: {per_class} samples per class')
    plt.imshow(grid_after.permute(1, 2, 0).numpy())
    plt.show()

    # 5) FID after unlearning
    fid_after = fid_eval.compute_fid(val_ds, z_dim=args.z_dim)
    write_json(fid_after, os.path.join(args.work_dir, "metrics", "fid_after.json"))

    # 6) Print summary
    print("CIFAR-10 classes (fixed order):")
    print(class_names)
    print("FID before unlearning:")
    print(json.dumps(fid_before, indent=2))
    print("FID after unlearning:")
    print(json.dumps(fid_after, indent=2))


if __name__ == "__main__":
    main()


