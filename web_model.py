import uuid
from functools import lru_cache
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import timm
import torch

from transforms import ScarNetTransform, guided_filter_rgb


CLASS_NAMES = ["hypertrophic", "keloid", "atrophic"]


class Predictor:
    def __init__(self, checkpoint_path: str):
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        model_name = ckpt["model_name"]
        dropout = ckpt.get("dropout", 0.4)
        self.guided_radius = ckpt.get("guided_radius", 8)
        self.guided_eps = ckpt.get("guided_eps", 1e-3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = timm.create_model(model_name, pretrained=False, num_classes=len(CLASS_NAMES), drop_rate=dropout)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(self.device).eval()
        self.model = model
        self.transform = ScarNetTransform(train=False, guided_radius=self.guided_radius, guided_eps=self.guided_eps)

    @torch.no_grad()
    def predict_bgr(self, image_bgr: np.ndarray) -> Dict:
        x = self.transform(image_bgr).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        return {
            "index": pred_idx,
            "label": CLASS_NAMES[pred_idx],
            "probabilities": {
                CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
            },
        }


@lru_cache(maxsize=2)
def get_predictor(checkpoint_path: str) -> Predictor:
    return Predictor(checkpoint_path=checkpoint_path)


def get_care_suggestions(label: str) -> list[str]:
    suggestions = {
        "hypertrophic": [
            "Discuss silicone gel/sheet therapy with a dermatologist.",
            "Consider pressure therapy and intralesional corticosteroid options.",
            "Avoid friction and repeated trauma to the scar area.",
            "Use strict sun protection to reduce discoloration changes.",
        ],
        "keloid": [
            "Consult a dermatologist for combination therapy planning.",
            "Common options include corticosteroid injections and silicone therapy.",
            "Some cases benefit from laser or cryotherapy under specialist care.",
            "Avoid self-treatment that irritates tissue and worsens growth.",
        ],
        "atrophic": [
            "Discuss procedures such as microneedling, TCA CROSS, or laser resurfacing.",
            "Topical retinoids may help texture over time with proper guidance.",
            "Use sunscreen daily to reduce contrast and post-inflammatory changes.",
            "Keep a gentle skincare routine to avoid further inflammation.",
        ],
    }
    return suggestions.get(label, ["Please consult a dermatologist for personalized care."])


def _write_rgb(path: Path, rgb_img: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def build_preprocessing_visuals(
    image_bgr: np.ndarray,
    output_dir: Path,
    guided_radius: int = 8,
    guided_eps: float = 1e-3,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    rgb_float = resized.astype(np.float32) / 255.0
    guided = guided_filter_rgb(rgb_float, radius=guided_radius, eps=guided_eps)
    guided_u8 = np.clip(guided * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(guided, cv2.COLOR_RGB2LAB).astype(np.float32)

    l = np.clip((lab[:, :, 0] / 100.0) * 255.0, 0, 255).astype(np.uint8)
    a = np.clip((lab[:, :, 1] + 128.0), 0, 255).astype(np.uint8)
    b = np.clip((lab[:, :, 2] + 128.0), 0, 255).astype(np.uint8)
    a_color = cv2.applyColorMap(a, cv2.COLORMAP_TURBO)
    b_color = cv2.applyColorMap(b, cv2.COLORMAP_TURBO)

    paths = {
        "original_resized": output_dir / f"{stem}_original_resized.jpg",
        "guided_filtered": output_dir / f"{stem}_guided_filtered.jpg",
        "lab_l_channel": output_dir / f"{stem}_lab_l_channel.jpg",
        "lab_a_channel": output_dir / f"{stem}_lab_a_channel.jpg",
        "lab_b_channel": output_dir / f"{stem}_lab_b_channel.jpg",
    }
    _write_rgb(paths["original_resized"], resized)
    _write_rgb(paths["guided_filtered"], guided_u8)
    cv2.imwrite(str(paths["lab_l_channel"]), l)
    cv2.imwrite(str(paths["lab_a_channel"]), a_color)
    cv2.imwrite(str(paths["lab_b_channel"]), b_color)
    return paths
