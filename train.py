import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    CLASS_NAMES,
    ScarDataset,
    Sample,
    collect_samples,
    compute_class_weights,
    stratified_kfold_train_val_indices,
    stratified_split,
)
from metrics_utils import compute_metrics, save_confusion_matrix, save_metrics_json
from transforms import ScarNetTransform, run_preprocessing_sanity_check


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_name: str, num_classes: int, dropout: float) -> nn.Module:
    if model_name == "efficientnet_b0":
        return timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes, drop_rate=dropout)
    if model_name == "resnet50":
        return timm.create_model("resnet50", pretrained=True, num_classes=num_classes, drop_rate=dropout)
    raise ValueError(f"Unsupported model: {model_name}")


def create_loaders(
    train_samples: List[Sample],
    val_samples: List[Sample],
    test_samples: List[Sample],
    batch_size: int,
    num_workers: int,
    augmentation_multiplier: int,
    guided_radius: int,
    guided_eps: float,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_tf = ScarNetTransform(train=True, augmentation_multiplier=augmentation_multiplier, guided_radius=guided_radius, guided_eps=guided_eps)
    eval_tf = ScarNetTransform(train=False, augmentation_multiplier=1, guided_radius=guided_radius, guided_eps=guided_eps)

    train_ds = ScarDataset(train_samples, transform=train_tf, train=True, augmentation_multiplier=augmentation_multiplier)
    val_ds = ScarDataset(val_samples, transform=eval_tf, train=False)
    test_ds = ScarDataset(test_samples, transform=eval_tf, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict:
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
        all_targets.extend(y.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0) if all_probs else None
    metrics = compute_metrics(all_targets, all_preds, class_names=CLASS_NAMES, y_prob=y_prob)
    metrics["loss"] = float(total_loss / max(1, n))
    return metrics


def train_one_fold(
    run_dir: Path,
    fold_name: str,
    model_name: str,
    train_samples: List[Sample],
    val_samples: List[Sample],
    test_samples: List[Sample],
    args: argparse.Namespace,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = create_loaders(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_multiplier=args.augmentation_multiplier,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
    )

    class_weights = compute_class_weights(train_samples, num_classes=len(CLASS_NAMES)).to(device)
    model = build_model(model_name, num_classes=len(CLASS_NAMES), dropout=args.dropout).to(device)

    if args.loss == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_f1 = -1.0
    patience_counter = 0
    fold_dir = run_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = fold_dir / "best.pt"
    history: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0
        loop = tqdm(train_loader, desc=f"{fold_name} Epoch {epoch}/{args.epochs}", leave=False)
        for x, y in loop:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = x.size(0)
            running_loss += loss.item() * bs
            n += bs
        train_loss = running_loss / max(1, n)

        val_metrics = evaluate_loader(model, val_loader, criterion, device)
        current_f1 = val_metrics["macro_f1"]
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(current_f1)

        epoch_info = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(current_f1),
            "val_macro_precision": float(val_metrics["macro_precision"]),
            "val_macro_recall": float(val_metrics["macro_recall"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_info)
        print(
            f"[{fold_name}] epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={current_f1:.4f}"
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            ckpt_payload = {
                "model_name": model_name,
                "class_names": CLASS_NAMES,
                "state_dict": model.state_dict(),
                "dropout": args.dropout,
                "seed": args.seed,
                "guided_radius": args.guided_radius,
                "guided_eps": args.guided_eps,
            }
            torch.save(ckpt_payload, best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"[{fold_name}] Early stopping at epoch {epoch} (best val macro-F1={best_f1:.4f})")
                break

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate_loader(model, test_loader, criterion, device)
    cm = np.array(test_metrics["confusion_matrix"], dtype=np.int64)
    save_confusion_matrix(cm, CLASS_NAMES, fold_dir / "confusion_matrix.png")

    full_metrics = {
        "fold": fold_name,
        "best_val_macro_f1": float(best_f1),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
        "history": history,
        "test": test_metrics,
    }
    save_metrics_json(full_metrics, fold_dir / "metrics.json")
    return full_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train acne scar classifier with ScarNet preprocessing.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau"])
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--loss", type=str, default="weighted_ce", choices=["weighted_ce", "focal"])
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--augmentation_multiplier", type=int, default=5)
    parser.add_argument("--guided_radius", type=int, default=8)
    parser.add_argument("--guided_eps", type=float, default=1e-3)
    parser.add_argument("--cv_folds", type=int, default=0, help="Set to 5 for 5-fold stratified CV.")
    parser.add_argument("--sanity_check", action="store_true", help="Run preprocessing sanity check on 8 images and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    samples = collect_samples(data_dir, class_names=CLASS_NAMES)

    if args.sanity_check:
        tf = ScarNetTransform(train=False, guided_radius=args.guided_radius, guided_eps=args.guided_eps)
        image_paths = [s.path for s in samples]
        run_preprocessing_sanity_check(image_paths=image_paths, transform=tf, max_images=8)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs") / f"{timestamp}_{args.model}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    if args.cv_folds and args.cv_folds > 1:
        print(f"Running stratified {args.cv_folds}-fold CV")
        fold_indices = stratified_kfold_train_val_indices(samples, n_splits=args.cv_folds, seed=args.seed)
        all_fold_metrics: List[Dict] = []
        for fold_id, (train_idx, val_idx) in enumerate(fold_indices, start=1):
            fold_name = f"fold_{fold_id}"
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]
            test_samples = val_samples
            result = train_one_fold(
                run_dir=run_dir,
                fold_name=fold_name,
                model_name=args.model,
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                args=args,
            )
            all_fold_metrics.append(result)
        macro_f1_scores = [m["test"]["macro_f1"] for m in all_fold_metrics]
        summary = {
            "cv_folds": args.cv_folds,
            "macro_f1_mean": float(np.mean(macro_f1_scores)),
            "macro_f1_std": float(np.std(macro_f1_scores)),
            "folds": all_fold_metrics,
        }
        save_metrics_json(summary, run_dir / "metrics.json")
        print(f"CV finished. macro_f1_mean={summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_std']:.4f}")
    else:
        splits = stratified_split(samples, seed=args.seed)
        print(
            f"Data split sizes: train={len(splits['train'])} "
            f"val={len(splits['val'])} test={len(splits['test'])}"
        )
        result = train_one_fold(
            run_dir=run_dir,
            fold_name="single_split",
            model_name=args.model,
            train_samples=splits["train"],
            val_samples=splits["val"],
            test_samples=splits["test"],
            args=args,
        )
        save_metrics_json(result, run_dir / "metrics.json")
        save_confusion_matrix(
            np.array(result["test"]["confusion_matrix"], dtype=np.int64),
            CLASS_NAMES,
            run_dir / "confusion_matrix.png",
        )
        print(f"Training finished. Best model: {run_dir / 'single_split' / 'best.pt'}")


if __name__ == "__main__":
    main()
