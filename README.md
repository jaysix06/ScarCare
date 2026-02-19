# Acne Scar Classification (ScarNet-style Preprocessing)

Complete PyTorch training project for 3-class acne scar classification:

1. `hypertrophic`
2. `keloid`
3. `atrophic` (icepick + boxcar + rolling grouped)

Primary model: `EfficientNet-B0`  
Baseline model: `ResNet-50`

## Folder Structure

```text
.
├─ data/
│  ├─ hypertrophic/
│  ├─ keloid/
│  └─ atrophic/
├─ requirements.txt
├─ transforms.py
├─ dataset.py
├─ metrics_utils.py
├─ train.py
├─ evaluate.py
├─ infer.py
└─ README.md
```

If your folder is named `dataset/` instead of `data/`, pass `--data_dir dataset`.

## ScarNet Preprocessing Pipeline (implemented in order)

In `transforms.py`, each image is processed as:

1. Resize to `224x224`
2. Guided Filter (OpenCV `ximgproc.guidedFilter` when available; NumPy/OpenCV fallback otherwise)
3. RGB to CIE L\*a\*b\* (`cv2.COLOR_RGB2LAB`)
4. Normalize to float32 in `[0, 1]` per channel using:
   - `L_scaled = L / 100.0`
   - `a_scaled = (a + 128.0) / 255.0`
   - `b_scaled = (b + 128.0) / 255.0`

Train-only augmentation includes:
- Rotation up to ±30 deg
- Horizontal flip
- Shear
- Scale (zoom in/out)
- Translation

Effective train set inflation is controlled by `--augmentation_multiplier` (default `5`).

## Setup (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --data_dir data --model efficientnet_b0
```

ResNet baseline:

```bash
python train.py --data_dir data --model resnet50
```

Useful flags:
- `--loss {weighted_ce,focal}` (default `weighted_ce`)
- `--scheduler {cosine,plateau}` (default `cosine`)
- `--cv_folds 5` for stratified 5-fold CV
- `--augmentation_multiplier 5`
- `--sanity_check` to run preprocessing check on 8 images

## Evaluate

```bash
python evaluate.py --checkpoint outputs/.../best.pt
```

Optional:
- `--data_dir data`
- `--split {train,val,test}` (default `test`)

## Inference

Single image:

```bash
python infer.py --checkpoint outputs/.../best.pt --image path\to\image.jpg
```

Batch folder:

```bash
python infer.py --checkpoint outputs/.../best.pt --input_dir path\to\folder --output_csv predictions.csv
```

## Outputs

Each run writes to `outputs/<timestamp>_<model>/` with:
- `config.json`
- `metrics.json`
- `confusion_matrix.png`
- Fold folder(s), e.g. `single_split/best.pt`, `single_split/metrics.json`

Metrics reported:
- Accuracy
- Macro Precision / Recall / F1
- Per-class Precision / Recall / F1
- Confusion matrix
- ROC-AUC (OvR macro, when feasible)

## Flask Web System (Landing, Analyze, Login/Register, Preprocessing Visualizer)

Added files:
- `app.py`
- `web_model.py`
- `templates/`
- `static/css/style.css`
- `sql/scar_system.sql`
- `.env.example`

### 1) Create database

Run in MySQL:

```sql
SOURCE sql/scar_system.sql;
```

If you already created an older schema with `full_name`, run this migration:

```sql
ALTER TABLE users ADD COLUMN first_name VARCHAR(80) NULL;
ALTER TABLE users ADD COLUMN last_name VARCHAR(80) NULL;
UPDATE users
SET first_name = SUBSTRING_INDEX(full_name, ' ', 1),
    last_name = TRIM(SUBSTRING(full_name, LENGTH(SUBSTRING_INDEX(full_name, ' ', 1)) + 1))
WHERE first_name IS NULL OR last_name IS NULL;
ALTER TABLE users MODIFY first_name VARCHAR(80) NOT NULL;
ALTER TABLE users MODIFY last_name VARCHAR(80) NOT NULL;
```

### 2) Configure environment

Copy `.env.example` to `.env` and set your DB credentials and checkpoint path.

### 3) Run web app

```bash
python app.py
```

Open:
- `http://127.0.0.1:5000/` (Landing)
- `http://127.0.0.1:5000/register`
- `http://127.0.0.1:5000/login`
- `http://127.0.0.1:5000/analyze`
- `http://127.0.0.1:5000/visualizer`
