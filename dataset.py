from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

from transforms import ScarNetTransform


CLASS_NAMES = ["hypertrophic", "keloid", "atrophic"]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Sample:
    path: Path
    label: int


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTS


def collect_samples(data_dir: Path, class_names: Sequence[str] = CLASS_NAMES) -> List[Sample]:
    samples: List[Sample] = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class folder missing: {class_dir}")
        for file_path in class_dir.rglob("*"):
            if _is_image_file(file_path):
                samples.append(Sample(path=file_path, label=class_idx))
    if not samples:
        raise ValueError(f"No image files found in {data_dir}")
    return samples


def stratified_split(
    samples: Sequence[Sample],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, List[Sample]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    labels = np.array([s.label for s in samples], dtype=np.int64)
    idx = np.arange(len(samples))

    train_idx, temp_idx = train_test_split(
        idx,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels,
    )

    temp_labels = labels[temp_idx]
    val_portion_of_temp = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_portion_of_temp),
        random_state=seed,
        stratify=temp_labels,
    )

    return {
        "train": [samples[i] for i in train_idx],
        "val": [samples[i] for i in val_idx],
        "test": [samples[i] for i in test_idx],
    }


def stratified_kfold_train_val_indices(samples: Sequence[Sample], n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    labels = np.array([s.label for s in samples], dtype=np.int64)
    idx = np.arange(len(samples))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(idx, labels))


def compute_class_weights(samples: Sequence[Sample], num_classes: int) -> torch.Tensor:
    labels = np.array([s.label for s in samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    inv = 1.0 / counts
    normalized = inv * (num_classes / inv.sum())
    return torch.tensor(normalized, dtype=torch.float32)


class ScarDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        transform: ScarNetTransform,
        train: bool = False,
        augmentation_multiplier: int = 1,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.train = train
        self.augmentation_multiplier = max(1, int(augmentation_multiplier))

    def __len__(self) -> int:
        base = len(self.samples)
        if self.train:
            return base * self.augmentation_multiplier
        return base

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index % len(self.samples)]
        image = cv2.imread(str(sample.path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {sample.path}")
        tensor = self.transform(image)
        return tensor, sample.label
