from __future__ import annotations

import os
import argparse
import json
from typing import List

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from datetime import datetime

from .data import (
    CIFAR10Config,
    load_cifar10_datasets,
    ImageNetSubsetConfig,
    load_imagenet_datasets,
)
from .trainer import TrainConfig, CGANTrainer
from .fid import FIDConfig, FIDEvaluator, FIDEvaluatorFromSampler, InceptionEmbedder
from .utils import ensure_dir, write_json, GradCAM, overlay_heatmap_on_images
from .pretrained import (
    load_pretrained_gan,
    make_conditional_sampler,
    fisher_prune_generator_with_classifier,
    _build_imagenet_classifier,
    _imagenet_preprocess,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conditional GAN on CIFAR-10 with Unlearning")
    p.add_argument("--data_dir", type=str, required=True, help="CIFAR-10 root directory for torchvision cache")
    p.add_argument("--work_dir", type=str, required=True)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--unlearn_epochs", type=int, default=10)
    p.add_argument("--fid_num_samples_per_class", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--unlearning_type", type=str, default="pure_finetuning", choices=["pure_finetuning", "fisher"], help="Unlearning strategy: pure_finetuning or fisher")
    p.add_argument("--samples_per_class", type=int, default=10, help="Number of samples to generate per class for before/after snapshots")
    # Performance/memory toggles
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", type=int, default=1, help="1 to enable pin_memory when CUDA; 0 to disable")
    p.add_argument("--amp", type=int, default=1, help="1 to enable mixed precision when CUDA; 0 to disable")
    p.add_argument("--sample_interval", type=int, default=2000, help="Steps between training sample grids; <=0 disables")
    p.add_argument("--keep_last", type=int, default=3, help="How many checkpoints to keep; <=0 keeps all")
    p.add_argument("--save_optimizer", type=int, default=0, help="1 to include optimizer in checkpoints")
    p.add_argument("--save_individual_samples", type=int, default=0, help="1 to save individual per-class images for snapshots")
    p.add_argument("--show_plots", type=int, default=0, help="1 to display matplotlib figures")
    # Pretrained options
    p.add_argument("--use_pretrained", type=int, default=0, help="1 to use a pretrained GAN instead of training")
    p.add_argument("--pretrained_gan_type", type=str, default="self_conditioned", help="GAN type for pytorch-pretrained-gans, e.g., biggan or self_conditioned")
    p.add_argument("--pretrained_resolution", type=int, default=256, help="Resolution for pretrained GAN, if supported")
    p.add_argument("--pretrained_unlearn", type=int, default=0, help="1 to apply Fisher pruning unlearning on the pretrained GAN itself")
    p.add_argument("--pretrained_unlearn_label", type=int, default=0, help="Label [0..K-1] to unlearn in pretrained mode (maps via --imagenet_biggan_indices if provided)")
    p.add_argument("--pretrained_fisher_batches", type=int, default=100, help="Max batches per side for Fisher in pretrained mode")
    p.add_argument("--pretrained_fisher_batch_size", type=int, default=32, help="Batch size for Fisher accumulation in pretrained mode")
    p.add_argument("--pretrained_fisher_threshold", type=float, default=15.0, help="Threshold on Fisher ratio for pruning in pretrained mode")
    # Analysis: similarity/divergence vs FID delta
    p.add_argument("--analyze_correlation", type=int, default=0, help="1 to compute similarity/divergence vs FID delta and save a plot")
    p.add_argument("--correlation_source", type=str, default="real", choices=["real", "fake_pre", "fake_post"], help="Source features for analysis (before/after unlearning)")
    p.add_argument("--divergence_metric", type=str, default="kl", choices=["cosine", "kl"], help="Use cosine similarity or KL divergence between classes")
    # Dataset switch: CIFAR-10 vs ImageNet-like subset
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet_subset"], help="Select dataset source")
    p.add_argument("--imagenet_classes", type=str, default="", help="Comma-separated class names to select (ImageFolder subdir names). If empty, auto-select first 10.")
    p.add_argument("--auto_select_k", type=int, default=10, help="When using imagenet_subset and no class list provided, pick first K classes deterministically")
    p.add_argument("--unlearn_label", type=int, default=0, help="Which remapped label [0..K-1] to unlearn")
    p.add_argument("--imagenet_biggan_indices", type=str, default="", help="Comma-separated BigGAN class indices aligned with selected class order; used when --use_pretrained=1 with imagenet_subset")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.work_dir)

    # 1) Load dataset (CIFAR-10 or ImageNet-like subset)
    os.makedirs(args.data_dir, exist_ok=True)
    if args.dataset == "cifar10":
        c_cfg = CIFAR10Config(data_dir=args.data_dir, img_size=args.img_size, train_download=True, test_download=True, seed=args.seed)
        train_ds, val_ds, _, class_names = load_cifar10_datasets(c_cfg)
        num_classes = 10
    else:
        selected = [s.strip() for s in args.imagenet_classes.split(",") if len(s.strip()) > 0]
        i_cfg = ImageNetSubsetConfig(
            data_dir=args.data_dir,
            selected_class_names=selected if len(selected) > 0 else None,
            img_size=args.img_size,
            auto_select_k=args.auto_select_k,
            seed=args.seed,
        )
        train_ds, val_ds, orig_idx_to_new, class_names = load_imagenet_datasets(i_cfg)
        num_classes = len(class_names)

    # Optional: Pretrained-only sampling + FID flow
    if int(args.use_pretrained) == 1:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        Gp = load_pretrained_gan(gan_type=args.pretrained_gan_type, resolution=args.pretrained_resolution)
        base_sampler = make_conditional_sampler(Gp, device)

        # Optional: map remapped labels [0..K-1] to BigGAN's global class indices when using imagenet_subset
        mapping_new_to_biggan = None
        if args.dataset == "imagenet_subset" and isinstance(class_names, list):
            if isinstance(args.imagenet_biggan_indices, str) and len(args.imagenet_biggan_indices.strip()) > 0:
                try:
                    mapping_new_to_biggan = [int(x.strip()) for x in args.imagenet_biggan_indices.split(",")]
                except Exception:
                    mapping_new_to_biggan = None
            # Fallback: if not provided, assume identity for first K classes (may mismatch semantics)
            if mapping_new_to_biggan is not None and len(mapping_new_to_biggan) != num_classes:
                raise ValueError("--imagenet_biggan_indices length must equal number of selected classes")

        def sampler(z: torch.Tensor, y_new: torch.Tensor) -> torch.Tensor:
            if mapping_new_to_biggan is None:
                return base_sampler(z, y_new)
            with torch.no_grad():
                y_mapped = torch.tensor([mapping_new_to_biggan[int(yy.item())] for yy in y_new], device=y_new.device, dtype=torch.long)
            return base_sampler(z, y_mapped)

        # Prepare a single classifier instance for Grad-CAM overlays (reuse before/after)
        clf_for_cam = None
        target_layer_for_cam = None
        try:
            clf_for_cam = _build_imagenet_classifier(device)
            try:
                target_layer_for_cam = clf_for_cam.layer4[-1].conv3
            except Exception:
                target_layer_for_cam = clf_for_cam.layer4[2].conv3
        except Exception:
            clf_for_cam = None
            target_layer_for_cam = None

        # Create a unique directory per trial
        samples_root = os.path.join(args.work_dir, "samples")
        os.makedirs(samples_root, exist_ok=True)
        trial_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        trial_base = f"trial_{trial_tag}_seed{args.seed}_pretrained_{args.pretrained_gan_type}"
        trial_dir = os.path.join(samples_root, trial_base)
        if os.path.exists(trial_dir):
            suffix = 1
            while True:
                candidate = os.path.join(samples_root, f"{trial_base}_{suffix:03d}")
                if not os.path.exists(candidate):
                    trial_dir = candidate
                    break
                suffix += 1
        os.makedirs(trial_dir, exist_ok=True)

        # Generate samples: N per selected class index (0..K-1)
        with torch.no_grad():
            imgs_all = []
            per_class = int(max(1, args.samples_per_class))
            before_dir = os.path.join(trial_dir, "pretrained_samples")
            os.makedirs(before_dir, exist_ok=True)
            for cls in range(num_classes):
                z = torch.randn(per_class, args.z_dim, device=device)
                y = torch.full((per_class,), cls, dtype=torch.long, device=device)
                x = sampler(z, y).cpu()
                imgs_all.append(x)
                if args.save_individual_samples:
                    class_dir = os.path.join(before_dir, f"class_{cls}")
                    os.makedirs(class_dir, exist_ok=True)
                    for i in range(per_class):
                        save_image(x[i], os.path.join(class_dir, f"img_{i+1}.png"), normalize=True, value_range=(-1, 1))
            imgs_all = torch.cat(imgs_all, dim=0)
            grid = make_grid(imgs_all, nrow=per_class, normalize=True, value_range=(-1, 1))
            save_image(grid, os.path.join(before_dir, "grid_all_classes.png"))

        # Grad-CAM overlays (before unlearning) using pretrained ImageNet classifier (micro-batched)
        if clf_for_cam is not None and target_layer_for_cam is not None:
            try:
                gradcam = GradCAM(clf_for_cam, target_layer_for_cam, device=device)
                imgs_b_full = imgs_all.to(device)
                labels_list = []
                for new_lbl in range(num_classes):
                    mapped = int(mapping_new_to_biggan[new_lbl]) if mapping_new_to_biggan is not None else int(new_lbl)
                    labels_list.extend([mapped] * per_class)
                labels_b_full = torch.tensor(labels_list, device=device, dtype=torch.long)

                overlays_parts = []
                cam_bsz = min(16, max(4, args.batch_size // 4))
                for start in range(0, imgs_b_full.size(0), cam_bsz):
                    end = min(imgs_b_full.size(0), start + cam_bsz)
                    imgs_b = imgs_b_full[start:end].clone().requires_grad_(True)
                    labels_b = labels_b_full[start:end]
                    logits_b = clf_for_cam(_imagenet_preprocess(imgs_b))
                    score_b = logits_b.gather(1, labels_b.view(-1, 1)).sum()
                    cam_b = gradcam.compute_heatmap(score_b)
                    cam_b_up = torch.nn.functional.interpolate(cam_b, size=imgs_b.shape[-2:], mode="bilinear", align_corners=False)
                    overlays_b = overlay_heatmap_on_images(imgs_b.detach(), cam_b_up.detach(), alpha=0.5).cpu()
                    overlays_parts.append(overlays_b)
                    del imgs_b, labels_b, logits_b, score_b, cam_b, cam_b_up
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                overlays_all = torch.cat(overlays_parts, dim=0)
                grid_cam_b = make_grid(overlays_all, nrow=per_class, normalize=True)
                save_image(grid_cam_b, os.path.join(before_dir, "gradcam_overlay_grid.png"))
                gradcam.remove_hooks()
            except Exception:
                pass

        # FID (per-class and overall) BEFORE unlearning
        fcfg = FIDConfig(
            work_dir=args.work_dir,
            num_samples_per_class=args.fid_num_samples_per_class,
            batch_size=max(32, args.batch_size // 2),
            device=args.device,
            num_workers=max(1, args.num_workers // 2),
        )
        fid_eval = FIDEvaluatorFromSampler(sampler, num_classes=num_classes, cfg=fcfg)
        fid_pre = fid_eval.compute_fid(val_ds, z_dim=args.z_dim)
        write_json(fid_pre, os.path.join(args.work_dir, "metrics", "fid_pretrained_before.json"))

        # Optional: unlearn one label directly on pretrained GAN via Fisher pruning
        if int(args.pretrained_unlearn) == 1:
            # Map the selected new label to BigGAN global label if mapping provided
            target_new = int(args.pretrained_unlearn_label)
            if mapping_new_to_biggan is None:
                forgotten_biggan = target_new
            else:
                forgotten_biggan = int(mapping_new_to_biggan[target_new])
            retained_biggan = []
            for new_lbl in range(num_classes):
                if new_lbl == target_new:
                    continue
                retained_biggan.append(int(mapping_new_to_biggan[new_lbl] if mapping_new_to_biggan is not None else new_lbl))

            pruned_g, _ = fisher_prune_generator_with_classifier(
                G=Gp,
                forgotten_biggan_label=forgotten_biggan,
                retained_biggan_labels=retained_biggan,
                z_dim=args.z_dim,
                device=device,
                max_batches=int(args.pretrained_fisher_batches),
                batch_size=int(args.pretrained_fisher_batch_size),
                threshold=float(args.pretrained_fisher_threshold),
            )
            print(f"Pretrained unlearning: pruned elements in G = {pruned_g}")

            # Re-wrap sampler to use updated Gp
            base_sampler_after = make_conditional_sampler(Gp, device)
            def sampler_after(z: torch.Tensor, y_new: torch.Tensor) -> torch.Tensor:
                if mapping_new_to_biggan is None:
                    return base_sampler_after(z, y_new)
                with torch.no_grad():
                    y_mapped = torch.tensor([mapping_new_to_biggan[int(yy.item())] for yy in y_new], device=y_new.device, dtype=torch.long)
                return base_sampler_after(z, y_mapped)

            # Generate samples after unlearning and save grids + Grad-CAM overlays
            with torch.no_grad():
                imgs_after = []
                per_class = int(max(1, args.samples_per_class))
                after_dir = os.path.join(trial_dir, "pretrained_after_unlearning")
                os.makedirs(after_dir, exist_ok=True)
                for cls in range(num_classes):
                    z = torch.randn(per_class, args.z_dim, device=device)
                    y = torch.full((per_class,), cls, dtype=torch.long, device=device)
                    x = sampler_after(z, y).cpu()
                    imgs_after.append(x)
                    if args.save_individual_samples:
                        class_dir = os.path.join(after_dir, f"class_{cls}")
                        os.makedirs(class_dir, exist_ok=True)
                        for i in range(per_class):
                            save_image(x[i], os.path.join(class_dir, f"img_{i+1}.png"), normalize=True, value_range=(-1, 1))
                imgs_after = torch.cat(imgs_after, dim=0)
                grid_after = make_grid(imgs_after, nrow=per_class, normalize=True, value_range=(-1, 1))
                save_image(grid_after, os.path.join(after_dir, "grid_all_classes.png"))

            # Grad-CAM overlays AFTER unlearning (micro-batched)
            if clf_for_cam is not None and target_layer_for_cam is not None:
                try:
                    gradcam = GradCAM(clf_for_cam, target_layer_for_cam, device=device)
                    imgs_a_full = imgs_after.to(device)
                    labels_list_a = []
                    for new_lbl in range(num_classes):
                        mapped = int(mapping_new_to_biggan[new_lbl]) if mapping_new_to_biggan is not None else int(new_lbl)
                        labels_list_a.extend([mapped] * per_class)
                    labels_a_full = torch.tensor(labels_list_a, device=device, dtype=torch.long)

                    overlays_parts_a = []
                    cam_bsz = min(16, max(4, args.batch_size // 4))
                    for start in range(0, imgs_a_full.size(0), cam_bsz):
                        end = min(imgs_a_full.size(0), start + cam_bsz)
                        imgs_a = imgs_a_full[start:end].clone().requires_grad_(True)
                        labels_a = labels_a_full[start:end]
                        logits_a = clf_for_cam(_imagenet_preprocess(imgs_a))
                        score_a = logits_a.gather(1, labels_a.view(-1, 1)).sum()
                        cam_a = gradcam.compute_heatmap(score_a)
                        cam_a_up = torch.nn.functional.interpolate(cam_a, size=imgs_a.shape[-2:], mode="bilinear", align_corners=False)
                        overlays_a = overlay_heatmap_on_images(imgs_a.detach(), cam_a_up.detach(), alpha=0.5).cpu()
                        overlays_parts_a.append(overlays_a)
                        del imgs_a, labels_a, logits_a, score_a, cam_a, cam_a_up
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    overlays_all_a = torch.cat(overlays_parts_a, dim=0)
                    grid_cam_a = make_grid(overlays_all_a, nrow=per_class, normalize=True)
                    save_image(grid_cam_a, os.path.join(after_dir, "gradcam_overlay_grid.png"))
                    gradcam.remove_hooks()
                except Exception:
                    pass

            # FID AFTER unlearning
            fid_eval_after = FIDEvaluatorFromSampler(sampler_after, num_classes=num_classes, cfg=fcfg)
            fid_post = fid_eval_after.compute_fid(val_ds, z_dim=args.z_dim)
            write_json(fid_post, os.path.join(args.work_dir, "metrics", "fid_pretrained_after.json"))

            # Optional analysis: correlation vs FID delta
            if int(args.analyze_correlation) == 1:
                import numpy as np
                import matplotlib.pyplot as plt
                from collections import defaultdict

                # 1) Build Inception features by class for chosen source
                device_emb = torch.device(args.device if torch.cuda.is_available() else "cpu")
                embedder = InceptionEmbedder(device_emb)
                per_class_feats: dict[int, list[np.ndarray]] = {c: [] for c in range(num_classes)}

                def collect_feats_from_sampler(sampler_fn, z_dim: int, per_class: int) -> dict[int, np.ndarray]:
                    out: dict[int, list[np.ndarray]] = {c: [] for c in range(num_classes)}
                    for cls in range(num_classes):
                        remaining = per_class
                        while remaining > 0:
                            bsz = min(max(16, args.batch_size // 2), remaining)
                            z = torch.randn(bsz, z_dim, device=device_emb)
                            y = torch.full((bsz,), cls, dtype=torch.long, device=device_emb)
                            x = sampler_fn(z, y)
                            feats = embedder.get_activations(x).cpu().numpy()
                            out[cls].append(feats)
                            remaining -= bsz
                    return {c: np.concatenate(v, axis=0) for c, v in out.items() if len(v) > 0}

                def collect_feats_from_dataset(dataset) -> dict[int, np.ndarray]:
                    from torch.utils.data import DataLoader
                    loader = DataLoader(dataset, batch_size=max(32, args.batch_size // 2), shuffle=False, num_workers=max(1, args.num_workers // 2), pin_memory=(device_emb.type == "cuda"))
                    out: dict[int, list[np.ndarray]] = {c: [] for c in range(num_classes)}
                    with torch.no_grad():
                        for x, y in loader:
                            x = x.to(device_emb)
                            y = y.to(device_emb)
                            feats = embedder.get_activations(x).cpu().numpy()
                            for i in range(len(y)):
                                out[int(y[i].item())].append(feats[i:i+1])
                    return {c: np.concatenate(v, axis=0) for c, v in out.items() if len(v) > 0}

                if args.correlation_source == "real":
                    feats_by_class = collect_feats_from_dataset(val_ds)
                elif args.correlation_source == "fake_pre":
                    feats_by_class = collect_feats_from_sampler(sampler, args.z_dim, per_class=max(50, args.fid_num_samples_per_class // 2))
                else:
                    feats_by_class = collect_feats_from_sampler(sampler_after, args.z_dim, per_class=max(50, args.fid_num_samples_per_class // 2))

                # 2) Fit per-class Gaussian stats
                def _mean_cov(v: np.ndarray):
                    mu = v.mean(axis=0)
                    # rowvar=False for features
                    sig = np.cov(v, rowvar=False)
                    return mu, sig
                stats = {c: _mean_cov(v) for c, v in feats_by_class.items()}
                forgotten_new = int(args.pretrained_unlearn_label)
                if forgotten_new not in stats:
                    # If class missing (edge case), skip plotting
                    pass
                else:
                    import numpy as np
                    # Cosine similarity based on means
                    def _cos(a, b):
                        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                        return float(np.dot(a, b) / denom)
                    # KL divergence between Gaussians N0(mu0,S0) || N1(mu1,S1)
                    # KL = 0.5*( tr(S1^{-1} S0) + (mu1-mu0)^T S1^{-1} (mu1-mu0) - k + ln(|S1|/|S0|) )
                    def _kl_gauss(mu0, S0, mu1, S1, eps=1e-6):
                        k = mu0.shape[0]
                        # Stabilize covariances
                        S0s = S0 + np.eye(k) * eps
                        S1s = S1 + np.eye(k) * eps
                        invS1 = np.linalg.inv(S1s)
                        diff = (mu1 - mu0).reshape(-1, 1)
                        term_trace = np.trace(invS1 @ S0s)
                        term_quad = float(diff.T @ invS1 @ diff)
                        sign0, logdet0 = np.linalg.slogdet(S0s)
                        sign1, logdet1 = np.linalg.slogdet(S1s)
                        if sign0 <= 0 or sign1 <= 0:
                            # Fallback if not SPD numerically
                            return float('nan')
                        term_logdet = logdet1 - logdet0
                        return 0.5 * (term_trace + term_quad - k + term_logdet)

                    mu_f, S_f = stats[forgotten_new]
                    if args.divergence_metric == "cosine":
                        # Similarity: cosine between means
                        metric_vals = {c: _cos(mu_f, stats[c][0]) for c in stats.keys()}
                        metric_name = "cosine_similarity"
                    else:
                        # Divergence: KL(N_forgotten || N_c)
                        metric_vals = {c: _kl_gauss(mu_f, S_f, stats[c][0], stats[c][1]) for c in stats.keys()}
                        metric_name = "kl_divergence_forgotten_to_class"

                    # 3) Build FID delta per class (post - pre)
                    fid_delta = {}
                    for c in range(num_classes):
                        key = f"class_{c}"
                        if key in fid_pre and key in fid_post:
                            fid_delta[c] = float(fid_post[key] - fid_pre[key])

                    # 4) Save CSV and plot
                    import csv
                    # Save analysis under the timestamped trial directory for consistency
                    metrics_dir = os.path.join(trial_dir, "metrics")
                    os.makedirs(metrics_dir, exist_ok=True)
                    csv_path = os.path.join(metrics_dir, "divergence_vs_fid_delta.csv")
                    with open(csv_path, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["class_index", "class_name", metric_name, "fid_delta_after_unlearning"])
                        for c in range(num_classes):
                            w.writerow([
                                c,
                                class_names[c] if c < len(class_names) else str(c),
                                metric_vals.get(c, None),
                                fid_delta.get(c, None),
                            ])

                    # Line plot with dual y-axes (left: divergence/similarity, right: FID delta)
                    xs = list(range(num_classes))
                    y1 = [metric_vals.get(c, float("nan")) for c in xs]
                    y2 = [fid_delta.get(c, float("nan")) for c in xs]
                    fig, ax1 = plt.subplots(figsize=(10, 4))
                    color1 = 'tab:blue'
                    color2 = 'tab:orange'
                    l1, = ax1.plot(xs, y1, marker='o', color=color1, label=metric_name.replace('_', ' '))
                    ax1.set_xlabel('class index (remapped order)')
                    ax1.set_ylabel(metric_name.replace('_', ' '), color=color1)
                    ax1.tick_params(axis='y', labelcolor=color1)
                    ax1.set_xticks(xs)
                    ax1.set_xticklabels([str(c) for c in xs], rotation=45)
                    ax1.axvline(forgotten_new, color='red', linestyle='--', alpha=0.5)

                    ax2 = ax1.twinx()
                    l2, = ax2.plot(xs, y2, marker='x', color=color2, label='FID delta (post - pre)')
                    ax2.set_ylabel('FID delta (post - pre)', color=color2)
                    ax2.tick_params(axis='y', labelcolor=color2)

                    lines = [l1, l2]
                    labels = [l.get_label() for l in lines]
                    fig.legend(lines, labels, loc='upper right')
                    fig.suptitle('Divergence/similarity to forgotten vs FID change after unlearning')
                    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
                    plot_path = os.path.join(metrics_dir, "divergence_vs_fid_delta.png")
                    fig.savefig(plot_path)
                    if int(args.show_plots) == 1:
                        plt.show()

            print("Selected classes (fixed order):")
            print(class_names)
            print("FID (pretrained before):")
            print(json.dumps(fid_pre, indent=2))
            print("FID (pretrained after):")
            print(json.dumps(fid_post, indent=2))
            return

        print("Selected classes (fixed order):")
        print(class_names)
        print("FID (pretrained):")
        print(json.dumps(fid_pre, indent=2))
        return

    # 2) Train CGAN
    tcfg = TrainConfig(
        work_dir=args.work_dir,
        img_size=args.img_size,
        z_dim=args.z_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        amp=bool(args.amp),
        sample_grid=32,
        sample_interval=args.sample_interval,
        keep_last=args.keep_last,
        save_optimizer=bool(args.save_optimizer),
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

            # Optional: save individual images per class
            if args.save_individual_samples:
                class_dir = os.path.join(before_dir, f"class_{cls}")
                os.makedirs(class_dir, exist_ok=True)
                for i in range(per_class):
                    save_image(x[i], os.path.join(class_dir, f"img_{i+1}.png"), normalize=True, value_range=(-1, 1))

        imgs_before = torch.cat(imgs_before, dim=0)
        # Combined grid with nrow = samples per class, rows = classes
        grid_before = make_grid(imgs_before, nrow=per_class, normalize=True, value_range=(-1, 1))
        save_image(grid_before, os.path.join(before_dir, "grid_all_classes.png"))
    trainer.G.train()

    # Grad-CAMs before unlearning (through Discriminator)
    # Target the last conv layer in D.features
    # Target the last convolutional feature map layer, not an activation
    # Discriminator.features layout: [conv, lrelu, conv, bn, lrelu, conv, bn, lrelu, conv, lrelu]
    # We want the last conv before projection, which is index 8 (0-based) for conv2d
    d_target_layer = trainer.D.features[8]
    gradcam = GradCAM(trainer.D, d_target_layer, device=trainer.device)
    trainer.D.eval()
    # Compute D scores for generated images and backprop to get CAMs
    with torch.no_grad():
        imgs_b = imgs_before.to(trainer.device)
        labels_b = torch.cat([torch.full((per_class,), c, device=trainer.device, dtype=torch.long) for c in range(num_classes)], dim=0)
    imgs_b.requires_grad_(True)
    pred_b = trainer.D(imgs_b, labels_b)
    cam_b = gradcam.compute_heatmap(pred_b.sum())  # (B,1,H',W') w.r.t last conv spatial size
    # Upsample CAMs to image size and overlay
    cam_b_up = torch.nn.functional.interpolate(cam_b, size=imgs_b.shape[-2:], mode="bilinear", align_corners=False)
    overlays_b = overlay_heatmap_on_images(imgs_b.detach(), cam_b_up.detach(), alpha=0.5).cpu()
    grid_cam_b = make_grid(overlays_b, nrow=per_class, normalize=True)
    save_image(grid_cam_b, os.path.join(before_dir, "gradcam_overlay_grid.png"))
    trainer.D.train()
    gradcam.remove_hooks()

    if args.show_plots:
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
        num_workers=max(1, args.num_workers // 2),
    )
    fid_eval = FIDEvaluator(trainer.G, num_classes=num_classes, cfg=fcfg)
    fid_before = fid_eval.compute_fid(val_ds, z_dim=args.z_dim)
    write_json(fid_before, os.path.join(args.work_dir, "metrics", "fid_before.json"))

    # 4) Unlearn the selected class label
    c1_new = int(args.unlearn_label)
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

            # Optional: save individual images per class
            if args.save_individual_samples:
                class_dir = os.path.join(after_dir, f"class_{cls}")
                os.makedirs(class_dir, exist_ok=True)
                for i in range(per_class):
                    save_image(x[i], os.path.join(class_dir, f"img_{i+1}.png"), normalize=True, value_range=(-1, 1))

        imgs_after = torch.cat(imgs_after, dim=0)
        grid_after = make_grid(imgs_after, nrow=per_class, normalize=True, value_range=(-1, 1))
        save_image(grid_after, os.path.join(after_dir, "grid_all_classes.png"))
    trainer.G.train()

    # Grad-CAMs after unlearning
    d_target_layer = trainer.D.features[8]
    gradcam = GradCAM(trainer.D, d_target_layer, device=trainer.device)
    trainer.D.eval()
    with torch.no_grad():
        imgs_a = imgs_after.to(trainer.device)
        labels_a = torch.cat([torch.full((per_class,), c, device=trainer.device, dtype=torch.long) for c in range(num_classes)], dim=0)
    imgs_a.requires_grad_(True)
    pred_a = trainer.D(imgs_a, labels_a)
    cam_a = gradcam.compute_heatmap(pred_a.sum())
    cam_a_up = torch.nn.functional.interpolate(cam_a, size=imgs_a.shape[-2:], mode="bilinear", align_corners=False)
    overlays_a = overlay_heatmap_on_images(imgs_a.detach(), cam_a_up.detach(), alpha=0.5).cpu()
    grid_cam_a = make_grid(overlays_a, nrow=per_class, normalize=True)
    save_image(grid_cam_a, os.path.join(after_dir, "gradcam_overlay_grid.png"))
    trainer.D.train()
    gradcam.remove_hooks()

    if args.show_plots:
        plt.figure(figsize=(12, 3))
        plt.axis('off')
        plt.title(f'After Unlearning: {per_class} samples per class')
        plt.imshow(grid_after.permute(1, 2, 0).numpy())
        plt.show()

    # 5) FID after unlearning
    fid_after = fid_eval.compute_fid(val_ds, z_dim=args.z_dim)
    write_json(fid_after, os.path.join(args.work_dir, "metrics", "fid_after.json"))

    # 6) Print summary
    print("Selected classes (fixed order):")
    print(class_names)
    print("FID before unlearning:")
    print(json.dumps(fid_before, indent=2))
    print("FID after unlearning:")
    print(json.dumps(fid_after, indent=2))


if __name__ == "__main__":
    main()


