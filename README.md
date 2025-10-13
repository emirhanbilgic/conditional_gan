## Conditional GAN on CIFAR-10 with Unlearning and FID

This project trains a conditional GAN on CIFAR-10, computes per-class and overall FID scores, then "unlearns" one class by fine-tuning on the remaining classes and recomputes FID. After training, a final generated image grid is saved and displayed.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run (CIFAR-10)

```bash
python -m src.main \
  --data_dir ./data \
  --work_dir ./runs/cifar10_cgan \
  --img_size 64 \
  --batch_size 128 \
  --z_dim 128 \
  --epochs 50 \
  --unlearn_epochs 10 \
  --fid_num_samples_per_class 1000
```

Notes:
- CIFAR-10 will be downloaded automatically into `--data_dir` on first run.
- Images are resized to 64×64 for DCGAN-style training.
- FID is computed per-class and overall using Inception-V3 features.
- The first class (index 0) is unlearned during the fine-tuning stage.

### Outputs

- Checkpoints in `work_dir/checkpoints/`
- Samples in `work_dir/samples/`
- FID JSON in `work_dir/metrics/`
- Final grid saved at `work_dir/samples/final_grid.png`; also displayed at the end of training

### Hardware

Training on a GPU is strongly recommended.

### Notes on Environments

- If running in a restricted environment (no Internet), pre-download CIFAR-10 into `--data_dir` using another machine.
- FID requires Inception-V3 weights available via torchvision; ensure Internet at first run or pre-cache models.


### Use a pretrained conditional GAN (no training)

Install the pretrained models package (already added in `requirements.txt`), then run:

```bash
python -m src.main \
  --data_dir ./data \
  --work_dir ./runs/pretrained \
  --img_size 64 \
  --batch_size 64 \
  --z_dim 128 \
  --fid_num_samples_per_class 200 \
  --use_pretrained 1 \
  --pretrained_gan_type self_conditioned \
  --pretrained_resolution 256
```

This will:
- Sample class-conditional images using the pretrained GAN interface from `pytorch-pretrained-gans` ([repo link](https://github.com/lukemelas/pytorch-pretrained-gans/tree/main/pytorch_pretrained_gans/self_conditioned)).
- Save a per-class grid to `work_dir/samples/.../grid_all_classes.png`.
- Compute per-class and overall FID versus CIFAR-10 and write JSON to `work_dir/metrics/fid_pretrained.json`.

If your environment doesn't have the package, install first (e.g., on Kaggle):

```bash
pip install -q --no-deps --no-cache-dir git+https://github.com/lukemelas/pytorch-pretrained-gans
```


