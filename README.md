## Conditional GAN on Stanford Cars with Unlearning and FID

This project trains a conditional GAN on a subset of Stanford Cars classes, computes per-class and overall FID scores, then "unlearns" the first class by fine-tuning on the remaining classes and recomputes FID.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python -m src.main \
  --data_dir /path/to/data \
  --work_dir ./runs/cars_cgan \
  --img_size 64 \
  --batch_size 128 \
  --z_dim 128 \
  --epochs 50 \
  --unlearn_epochs 10 \
  --fid_num_samples_per_class 1000 \
  --classes 0 1 2 3 4 5 6 7 8 9
```

Notes:
- The `--classes` are Stanford Cars class indices (0..195). The first index is treated as `c1` and will be unlearned.
- Images are resized to 64×64 for DCGAN-style training.
- FID is computed per-class (using the test split) and overall.

### Outputs

- Checkpoints in `work_dir/checkpoints/`
- Samples in `work_dir/samples/`
- FID JSON/CSV in `work_dir/metrics/`

### Hardware

Training on a GPU is strongly recommended.


