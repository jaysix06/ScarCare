import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, ScarDataset, collect_samples, stratified_split
from metrics_utils import compute_metrics, save_confusion_matrix, save_metrics_json
from transforms import ScarNetTransform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate acne scar classification checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser.parse_args()


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_targets.extend(y.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())
    y_prob = np.concatenate(all_probs, axis=0) if all_probs else None
    return compute_metrics(all_targets, all_preds, class_names=CLASS_NAMES, y_prob=y_prob)


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt["model_name"]
    dropout = ckpt.get("dropout", 0.4)
    guided_radius = ckpt.get("guided_radius", 8)
    guided_eps = ckpt.get("guided_eps", 1e-3)

    model = timm.create_model(model_name, pretrained=False, num_classes=len(CLASS_NAMES), drop_rate=dropout)
    model.load_state_dict(ckpt["state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    samples = collect_samples(Path(args.data_dir), class_names=CLASS_NAMES)
    splits = stratified_split(samples, seed=args.seed)
    split_samples = splits[args.split]

    tf = ScarNetTransform(train=False, guided_radius=guided_radius, guided_eps=guided_eps)
    ds = ScarDataset(split_samples, transform=tf, train=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )

    metrics = run_eval(model, loader, device)

    out_dir = ckpt_path.parent / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_json(metrics, out_dir / "metrics.json")
    save_confusion_matrix(np.array(metrics["confusion_matrix"], dtype=np.int64), CLASS_NAMES, out_dir / "confusion_matrix.png")

    print(f"Evaluation split: {args.split}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    if metrics.get("roc_auc_ovr_macro") is not None:
        print(f"ROC-AUC OVR Macro: {metrics['roc_auc_ovr_macro']:.4f}")
    print(f"Saved: {out_dir / 'metrics.json'}")
    print(f"Saved: {out_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
