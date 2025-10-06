## Conditional GAN on ImageNet-10 with Unlearning and FID

This project trains a conditional GAN on a subset of 10 ImageNet-style classes, computes per-class and overall FID scores, then "unlearns" the first class by fine-tuning on the remaining classes and recomputes FID.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python -m src.main \
  --data_dir /path/to/imagenet_like_root \
  --work_dir ./runs/imagenet10_cgan \
  --img_size 64 \
  --batch_size 128 \
  --z_dim 128 \
  --epochs 50 \
  --unlearn_epochs 10 \
  --fid_num_samples_per_class 1000 \
  --classes class_a class_b class_c class_d class_e class_f class_g class_h class_i class_j
```

Notes:
- Expects ImageNet-style folders: `data_dir/train/<class_name>/*.jpg` and `data_dir/val/<class_name>/*.jpg`.
- If you omit `--classes`, the script auto-selects 10 classes alphabetically.
- The first provided/selected class is treated as `c1` and will be unlearned.
- Images are resized to 64×64 for DCGAN-style training.
- FID is computed per-class (using the val split) and overall.

### Outputs

- Checkpoints in `work_dir/checkpoints/`
- Samples in `work_dir/samples/`
- FID JSON/CSV in `work_dir/metrics/`

### Hardware

Training on a GPU is strongly recommended.

### Kaggle Notebook

1) Enable Internet in Notebook Settings (needed to download Stanford Cars and Inception weights).

2) Clone repo and install minimal deps (PyTorch exists on Kaggle already):

```bash
git clone https://github.com/emirhanbilgic/conditional_gan.git
cd conditional_gan
pip install -q tqdm scipy pandas torchmetrics
```

3) Run training + FID + unlearning (writeable paths on Kaggle):

```bash
python -m src.main \
  --data_dir /kaggle/input/your_imagenet_like_dataset \
  --work_dir /kaggle/working/runs/imagenet10_cgan \
  --img_size 64 \
  --batch_size 128 \
  --z_dim 128 \
  --epochs 50 \
  --unlearn_epochs 10 \
  --fid_num_samples_per_class 1000 \
  --classes  tench  goldfish  great_white_shark  tiger_shark  hammerhead  electric_ray  stingray  cock  hen  ostrich
```

If you prefer Internet Off, attach a Kaggle Dataset containing ImageNet-like folders at `/kaggle/input/your_imagenet_like_dataset/{train,val}/<class_name>/*.jpg`. FID also needs Inception weights; pre-bundle them or enable Internet for that step.


