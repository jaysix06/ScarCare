import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import timm
import torch

from dataset import CLASS_NAMES, VALID_EXTS
from transforms import ScarNetTransform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for acne scar classifier.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None, help="Single image path.")
    parser.add_argument("--input_dir", type=str, default=None, help="Folder of images for batch inference.")
    parser.add_argument("--output_csv", type=str, default="predictions.csv")
    return parser.parse_args()


def load_model(checkpoint_path: Path) -> Tuple[torch.nn.Module, ScarNetTransform, torch.device]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name = ckpt["model_name"]
    dropout = ckpt.get("dropout", 0.4)
    guided_radius = ckpt.get("guided_radius", 8)
    guided_eps = ckpt.get("guided_eps", 1e-3)
    model = timm.create_model(model_name, pretrained=False, num_classes=len(CLASS_NAMES), drop_rate=dropout)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    transform = ScarNetTransform(train=False, guided_radius=guided_radius, guided_eps=guided_eps)
    return model, transform, device


@torch.no_grad()
def predict_one(model: torch.nn.Module, transform: ScarNetTransform, device: torch.device, image_path: Path) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")
    x = transform(image).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    out = {
        "path": str(image_path),
        "pred_label": CLASS_NAMES[pred_idx],
        "pred_index": pred_idx,
    }
    for i, class_name in enumerate(CLASS_NAMES):
        out[f"prob_{class_name}"] = float(probs[i])
    return out


def collect_images(input_dir: Path) -> List[Path]:
    paths = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            paths.append(p)
    if not paths:
        raise ValueError(f"No images found in {input_dir}")
    return sorted(paths)


def main() -> None:
    args = parse_args()
    if not args.image and not args.input_dir:
        raise ValueError("Provide either --image or --input_dir")

    model, transform, device = load_model(Path(args.checkpoint))

    if args.image:
        result = predict_one(model, transform, device, Path(args.image))
        print(result)
        return

    image_paths = collect_images(Path(args.input_dir))
    rows = [predict_one(model, transform, device, p) for p in image_paths]
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv} ({len(df)} images)")


if __name__ == "__main__":
    main()
