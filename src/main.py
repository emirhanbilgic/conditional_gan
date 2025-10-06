from __future__ import annotations

import os
import argparse
import json
from typing import List

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

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

    # Generate a grid before unlearning: one example per class and display
    trainer.G.eval()
    with torch.no_grad():
        imgs_before = []
        per_class = 1
        for cls in range(num_classes):
            z = torch.randn(per_class, args.z_dim, device=trainer.device)
            y = torch.full((per_class,), cls, dtype=torch.long, device=trainer.device)
            x = trainer.G(z, y).cpu()
            imgs_before.append(x)
        imgs_before = torch.cat(imgs_before, dim=0)
        grid_before = make_grid(imgs_before, nrow=num_classes, normalize=True, value_range=(-1, 1))
        os.makedirs(os.path.join(args.work_dir, "samples"), exist_ok=True)
        from torchvision.utils import save_image as _save_image
        _save_image(grid_before, os.path.join(args.work_dir, "samples", "before_unlearning.png"))
    trainer.G.train()

    plt.figure(figsize=(12, 3))
    plt.axis('off')
    plt.title('Before Unlearning: 1 sample per class')
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

    # Generate a grid after unlearning: one example per class and display
    trainer.G.eval()
    with torch.no_grad():
        imgs_after = []
        per_class = 1
        for cls in range(num_classes):
            z = torch.randn(per_class, args.z_dim, device=trainer.device)
            y = torch.full((per_class,), cls, dtype=torch.long, device=trainer.device)
            x = trainer.G(z, y).cpu()
            imgs_after.append(x)
        imgs_after = torch.cat(imgs_after, dim=0)
        grid_after = make_grid(imgs_after, nrow=num_classes, normalize=True, value_range=(-1, 1))
        os.makedirs(os.path.join(args.work_dir, "samples"), exist_ok=True)
        from torchvision.utils import save_image as _save_image
        _save_image(grid_after, os.path.join(args.work_dir, "samples", "after_unlearning.png"))
    trainer.G.train()

    plt.figure(figsize=(12, 3))
    plt.axis('off')
    plt.title('After Unlearning: 1 sample per class')
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


