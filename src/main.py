from __future__ import annotations

import os
import argparse
import json
from typing import List

import torch

from .data import CarsSubsetConfig, load_cars_datasets, build_dataloader, filter_dataset_by_new_label
from .trainer import TrainConfig, CGANTrainer
from .fid import FIDConfig, FIDEvaluator
from .utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conditional GAN on Stanford Cars with Unlearning")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--work_dir", type=str, required=True)
    p.add_argument("--classes", type=int, nargs="+", required=True, help="Original class indices to include; first is c1 to unlearn")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--unlearn_epochs", type=int, default=10)
    p.add_argument("--fid_num_samples_per_class", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.work_dir)

    # 1) Load datasets with selected classes
    cars_cfg = CarsSubsetConfig(
        data_dir=args.data_dir,
        selected_classes=args.classes,
        img_size=args.img_size,
        download=True,
    )
    train_ds, test_ds, orig_to_new = load_cars_datasets(cars_cfg)
    num_classes = len(args.classes)

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

    # 3) FID before unlearning
    fcfg = FIDConfig(
        work_dir=args.work_dir,
        num_samples_per_class=args.fid_num_samples_per_class,
        batch_size=max(32, args.batch_size // 2),
        device=args.device,
    )
    fid_eval = FIDEvaluator(trainer.G, num_classes=num_classes, cfg=fcfg)
    fid_before = fid_eval.compute_fid(test_ds, z_dim=args.z_dim)
    write_json(fid_before, os.path.join(args.work_dir, "metrics", "fid_before.json"))

    # 4) Unlearn the first selected class (c1)
    c1_orig = args.classes[0]
    c1_new = orig_to_new[c1_orig]
    # filter out all samples equal to c1_new
    indices = [i for i in range(len(train_ds)) if int(train_ds[i][1]) != int(c1_new)]
    from torch.utils.data import Subset
    train_without_c1 = Subset(train_ds, indices)
    trainer.finetune_excluding_class(train_without_c1, epochs=args.unlearn_epochs)

    # 5) FID after unlearning
    fid_after = fid_eval.compute_fid(test_ds, z_dim=args.z_dim)
    write_json(fid_after, os.path.join(args.work_dir, "metrics", "fid_after.json"))

    # 6) Print summary
    print("FID before unlearning:")
    print(json.dumps(fid_before, indent=2))
    print("FID after unlearning:")
    print(json.dumps(fid_after, indent=2))


if __name__ == "__main__":
    main()


