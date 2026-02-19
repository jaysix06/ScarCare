from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


IMG_SIZE = 224
LAB_SCALE_DOC = {
    "L": "L/100.0",
    "A": "(a+128.0)/255.0",
    "B": "(b+128.0)/255.0",
}


def _guided_filter_gray(guide_gray: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    k = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide_gray, ddepth=-1, ksize=k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(src, ddepth=-1, ksize=k, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_i = cv2.boxFilter(guide_gray * guide_gray, ddepth=-1, ksize=k, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_ip = cv2.boxFilter(guide_gray * src, ddepth=-1, ksize=k, normalize=True, borderType=cv2.BORDER_REFLECT)

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=k, normalize=True, borderType=cv2.BORDER_REFLECT)

    q = mean_a * guide_gray + mean_b
    return np.clip(q, 0.0, 1.0)


def guided_filter_rgb(image_rgb_float: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        filtered = cv2.ximgproc.guidedFilter(
            guide=image_rgb_float,
            src=image_rgb_float,
            radius=radius,
            eps=eps,
            dDepth=-1,
        )
        return np.clip(filtered, 0.0, 1.0)

    guide_gray = cv2.cvtColor(image_rgb_float, cv2.COLOR_RGB2GRAY)
    channels = []
    for c in range(3):
        channels.append(_guided_filter_gray(guide_gray, image_rgb_float[:, :, c], radius=radius, eps=eps))
    return np.stack(channels, axis=2).astype(np.float32)


def rgb_to_lab_scaled(image_rgb_float: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb_float, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0] / 100.0
    a = (lab[:, :, 1] + 128.0) / 255.0
    b = (lab[:, :, 2] + 128.0) / 255.0
    return np.stack([l, a, b], axis=2).astype(np.float32)


class ScarNetTransform:
    def __init__(
        self,
        train: bool,
        augmentation_multiplier: int = 1,
        guided_radius: int = 8,
        guided_eps: float = 1e-3,
    ) -> None:
        self.train = train
        self.augmentation_multiplier = max(1, int(augmentation_multiplier))
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps
        self.train_aug = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(
                    degrees=30,
                    translate=(0.15, 0.15),
                    scale=(0.8, 1.2),
                    shear=(-20, 20),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                ),
            ]
        )

    def __call__(self, image_bgr_uint8: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        rgb_float = resized.astype(np.float32) / 255.0

        guided = guided_filter_rgb(rgb_float, radius=self.guided_radius, eps=self.guided_eps)
        lab_scaled = rgb_to_lab_scaled(guided)

        tensor = torch.from_numpy(lab_scaled).permute(2, 0, 1).contiguous().float()
        if self.train:
            tensor = self.train_aug(tensor)
            tensor = torch.clamp(tensor, 0.0, 1.0)
        return tensor


def run_preprocessing_sanity_check(
    image_paths: Sequence[Path],
    transform: ScarNetTransform,
    max_images: int = 8,
) -> None:
    subset = list(image_paths)[: max_images]
    if not subset:
        raise ValueError("No images found for sanity check.")

    print(f"Running preprocessing sanity check on {len(subset)} images...")
    batch: List[torch.Tensor] = []
    for path in subset:
        image = cv2.imread(str(path))
        if image is None:
            continue
        tensor = transform(image)
        print(
            f"{path.name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"min={tensor.min().item():.4f} max={tensor.max().item():.4f}"
        )
        batch.append(tensor)
    if not batch:
        raise RuntimeError("Sanity check failed: no readable images.")
    stacked = torch.stack(batch, dim=0)
    print(
        f"Batch summary: shape={tuple(stacked.shape)} dtype={stacked.dtype} "
        f"min={stacked.min().item():.4f} max={stacked.max().item():.4f}"
    )
