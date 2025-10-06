from __future__ import annotations

import os
import argparse
import json
from typing import List

import torch

from .data import ImageNetSubsetConfig, load_imagenet_datasets, build_dataloader, filter_dataset_by_new_label
from .data import download_and_prepare_tiny_imagenet
from .trainer import TrainConfig, CGANTrainer
from .fid import FIDConfig, FIDEvaluator
from .utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conditional GAN on ImageNet-10 with Unlearning")
    p.add_argument("--data_dir", type=str, default=None, help="Dataset root. If --tiny_imagenet is set, this is the download/extract root.")
    p.add_argument("--work_dir", type=str, required=True)
    p.add_argument("--classes", type=str, nargs="*", default=None, help="Class names to include; first is c1 to unlearn. If omitted, auto-select 10 alphabetically.")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--unlearn_epochs", type=int, default=10)
    p.add_argument("--fid_num_samples_per_class", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--val_split", type=float, default=0.2, help="If no train/val folders, split ratio for val")
    p.add_argument("--tiny_imagenet", action="store_true", help="Download and use Tiny ImageNet (200 classes)")
    p.add_argument("--num_classes", type=int, default=10, help="Number of classes to use (auto-selected if --classes omitted)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.work_dir)

    # 1) Prepare dataset root
    if args.tiny_imagenet:
        droot = args.data_dir if args.data_dir is not None else "/kaggle/working/data"
        os.makedirs(droot, exist_ok=True)
        data_root = download_and_prepare_tiny_imagenet(droot)
    else:
        if args.data_dir is None:
            raise ValueError("--data_dir must be provided when not using --tiny_imagenet")
        data_root = args.data_dir

    # 2) Load datasets with selected classes (ImageNet-style)
    im_cfg = ImageNetSubsetConfig(
        data_dir=data_root,
        selected_class_names=args.classes if args.classes is not None and len(args.classes) > 0 else None,
        img_size=args.img_size,
        auto_select_k=args.num_classes,
        val_split=args.val_split,
        seed=args.seed,
    )
    train_ds, val_ds, orig_to_new, class_names = load_imagenet_datasets(im_cfg)
    num_classes = len(class_names)

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
    fid_before = fid_eval.compute_fid(val_ds, z_dim=args.z_dim)
    write_json(fid_before, os.path.join(args.work_dir, "metrics", "fid_before.json"))

    # 4) Unlearn the first selected class (c1)
    c1_new = 0  # by construction, the first selected class is mapped to 0
    # filter out all samples equal to c1_new
    indices = [i for i in range(len(train_ds)) if int(train_ds[i][1]) != int(c1_new)]
    from torch.utils.data import Subset
    train_without_c1 = Subset(train_ds, indices)
    trainer.finetune_excluding_class(train_without_c1, epochs=args.unlearn_epochs)

    # 5) FID after unlearning
    fid_after = fid_eval.compute_fid(val_ds, z_dim=args.z_dim)
    write_json(fid_after, os.path.join(args.work_dir, "metrics", "fid_after.json"))

    # 6) Print summary
    print("Selected classes (in order):")
    print(class_names)
    print("FID before unlearning:")
    print(json.dumps(fid_before, indent=2))
    print("FID after unlearning:")
    print(json.dumps(fid_after, indent=2))


if __name__ == "__main__":
    main()


