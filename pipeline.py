import os
import json
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Keep native libraries on a single thread for low-memory cloud instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


IMAGE_SIZE = 64
EMBED_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REFERENCE_EMBEDDING_VERSION = 1
DEFAULT_MODEL_ARCH = os.environ.get("EPIGRANET_MODEL_ARCH", "auto").strip().lower() or "auto"

try:
    torch.set_num_threads(int(os.environ.get("EPIGRANET_TORCH_THREADS", "1")))
    torch.set_num_interop_threads(int(os.environ.get("EPIGRANET_TORCH_INTEROP_THREADS", "1")))
except RuntimeError:
    # Thread counts can only be set before work starts; ignore if already initialized.
    pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rotate_nearest(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE,
    )


def correct_skew(image: np.ndarray, delta: int = 1, limit: int = 5) -> Tuple[float, np.ndarray]:
    def determine_score(arr: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
        data = rotate_nearest(arr, angle)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, float(score)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        _, score = determine_score(thresh, float(angle))
        scores.append(score)

    best_angle = float(angles[scores.index(max(scores))])
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return best_angle, rotated


def preprocess_image(input_path: Path, output_path: Path) -> Path:
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError("Unable to read input image.")

    _, rotated = correct_skew(image)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Conservative line-removal: only remove very long ruling lines, not character strokes.
    h, w = gray.shape
    min_hline = max(80, int(w * 0.22))
    min_vline = max(80, int(h * 0.30))
    line_mask = np.zeros_like(gray)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 12), 1))
    horizontal_open = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(horizontal_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw >= min_hline and ch <= 8:
            cv2.rectangle(line_mask, (x, y), (x + cw, y + ch), 255, -1)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 8)))
    vertical_open = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    cnts = cv2.findContours(vertical_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch >= min_vline and cw <= 8:
            cv2.rectangle(line_mask, (x, y), (x + cw, y + ch), 255, -1)

    cleaned = cv2.inpaint(gray, line_mask, 3, cv2.INPAINT_TELEA)
    denoised = cv2.fastNlMeansDenoising(cleaned, None, 8, 7, 15)

    # Normalize uneven illumination to suppress paper texture and scanner shading.
    bg = cv2.GaussianBlur(denoised, (0, 0), sigmaX=15, sigmaY=15)
    normalized = cv2.divide(denoised, bg, scale=255)

    # Build two binarization candidates and pick the one that preserves text strokes better.
    adaptive = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        2,
    )
    _, otsu = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def black_ratio(img: np.ndarray) -> float:
        return float(np.mean(img == 0))

    adaptive_ratio = black_ratio(adaptive)
    otsu_ratio = black_ratio(otsu)

    # Prefer ratios in a sane text range; fallback to whichever keeps more foreground.
    def score_ratio(ratio: float) -> float:
        if 0.02 <= ratio <= 0.35:
            return 1.0 - abs(ratio - 0.12)
        return -abs(ratio - 0.12)

    final_image = adaptive if score_ratio(adaptive_ratio) >= score_ratio(otsu_ratio) else otsu

    # Drop tiny isolated components that usually correspond to threshold noise.
    h, w = final_image.shape
    min_component_area = max(10, int(h * w * 0.00002))
    inv = 255 - final_image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    cleaned_inv = np.zeros_like(inv)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned_inv[labels == label] = 255
    final_image = 255 - cleaned_inv

    # Slight reconnect of broken curves without thickening too much.
    final_image = cv2.morphologyEx(
        final_image,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    cv2.imwrite(str(output_path), final_image)
    return output_path


def segment_characters(
    processed_image_path: Path,
    roi_dir: Path,
    boxed_output_path: Path,
    min_area: int = 200,
) -> List[Path]:
    ensure_dir(roi_dir)
    image = cv2.imread(str(processed_image_path))
    if image is None:
        raise ValueError("Unable to read preprocessed image for segmentation.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilate = cv2.dilate(thresh, None, iterations=2)

    cnts_raw = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts_raw[0] if len(cnts_raw) == 2 else cnts_raw[1]
    sorted_ctrs = sorted(
        cnts,
        key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * image.shape[1],
    )

    boxed = image.copy()
    rois: List[Path] = []
    for index, cnt in enumerate(sorted_ctrs):
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y : y + h, x : x + w]
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_path = roi_dir / f"roi_{index}.png"
        cv2.imwrite(str(roi_path), roi)
        rois.append(roi_path)

    cv2.imwrite(str(boxed_output_path), boxed)
    return rois


class ResNet18EmbeddingNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torchvision import models

        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return F.normalize(x, p=2, dim=1)


class TinyEmbeddingNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(96, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)


def build_model(arch: str) -> nn.Module:
    arch = arch.strip().lower()
    if arch == "resnet18":
        return ResNet18EmbeddingNet()
    if arch == "tiny_cnn":
        return TinyEmbeddingNet()
    raise ValueError(f"Unsupported model architecture: {arch}")


def infer_architecture_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = set(state_dict.keys())
    if any(key.startswith("backbone.layer") or key.startswith("backbone.conv1") for key in keys):
        return "resnet18"
    if any(key.startswith("features.") or key.startswith("head.") for key in keys):
        return "tiny_cnn"
    raise ValueError(
        "Unable to infer model architecture from checkpoint keys. "
        "Set EPIGRANET_MODEL_ARCH or save checkpoints with {'arch': ..., 'state_dict': ...}."
    )


def load_checkpoint_payload(model_path: Path) -> Tuple[str, Dict[str, torch.Tensor]]:
    payload = torch.load(str(model_path), map_location=DEVICE)

    if isinstance(payload, OrderedDict):
        state_dict = dict(payload)
        return infer_architecture_from_state_dict(state_dict), state_dict

    if isinstance(payload, dict) and "state_dict" in payload:
        raw_state_dict = payload["state_dict"]
        if not isinstance(raw_state_dict, (dict, OrderedDict)):
            raise ValueError("Checkpoint 'state_dict' must be a dictionary of tensors.")
        state_dict = dict(raw_state_dict)
        arch = str(payload.get("arch") or infer_architecture_from_state_dict(state_dict)).strip().lower()
        return arch, state_dict

    if isinstance(payload, dict):
        state_dict = dict(payload)
        return infer_architecture_from_state_dict(state_dict), state_dict

    raise ValueError("Unsupported checkpoint format.")


def prepare_image_tensor(image_path: Path, channels: int) -> torch.Tensor:
    img = Image.open(image_path)
    if channels == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0

    if channels == 1:
        arr = np.expand_dims(arr, axis=0)
    else:
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.transpose(arr, (2, 0, 1))

    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor.to(DEVICE)


@dataclass
class PredictionResult:
    text: str
    confidence: float
    tokens: List[Dict[str, object]]


class OCRPredictor:
    def __init__(
        self,
        model_path: Path,
        class_mapping_path: Optional[Path] = None,
        embedding_cache_path: Optional[Path] = None,
        dataset_path: Optional[Path] = None,
    ) -> None:
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.class_mapping_path = class_mapping_path
        self.embedding_cache_path = embedding_cache_path
        self.model_arch = DEFAULT_MODEL_ARCH
        self.input_channels = 3
        self.model: Optional[nn.Module] = None
        self.reference_embeddings: Dict[str, torch.Tensor] = {}
        self.class_mapping: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        checkpoint_arch, state_dict = load_checkpoint_payload(self.model_path)
        if self.model_arch == "auto":
            self.model_arch = checkpoint_arch
        elif self.model_arch != checkpoint_arch:
            raise ValueError(
                f"Configured model architecture '{self.model_arch}' does not match checkpoint architecture "
                f"'{checkpoint_arch}'."
            )

        self.model = build_model(self.model_arch).to(DEVICE)
        self.input_channels = 1 if self.model_arch == "tiny_cnn" else 3
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.class_mapping = self._load_class_mapping()

        if self.embedding_cache_path and self.embedding_cache_path.exists():
            self.reference_embeddings = self._load_reference_embeddings_from_cache(self.embedding_cache_path)
            return

        if not self.dataset_path or not self.dataset_path.exists():
            cache_hint = f" or embedding cache not found: {self.embedding_cache_path}" if self.embedding_cache_path else ""
            raise FileNotFoundError(
                f"Reference dataset not found: {self.dataset_path}{cache_hint}"
            )

        self.reference_embeddings = self._build_reference_embeddings_from_dataset(self.dataset_path)
        if self.embedding_cache_path:
            self._save_reference_embeddings_cache(self.embedding_cache_path, self.reference_embeddings)

    def _load_class_mapping(self) -> Dict[str, str]:
        if not self.class_mapping_path:
            return {}
        if not self.class_mapping_path.exists():
            raise FileNotFoundError(f"Class mapping file not found: {self.class_mapping_path}")
        with self.class_mapping_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Class mapping file must contain a JSON object.")
        return {str(key): str(value) for key, value in data.items()}

    def _build_reference_embeddings_from_dataset(self, dataset_path: Path) -> Dict[str, torch.Tensor]:
        reference_embeddings: Dict[str, torch.Tensor] = {}
        classes = sorted(os.listdir(dataset_path))
        for cls in classes:
            class_path = dataset_path / cls
            if not class_path.is_dir():
                continue
            images = sorted(p for p in class_path.iterdir() if p.is_file())
            if not images:
                continue
            # Preserve the original app behavior by using one representative sample per class.
            emb = self._embed_image(images[0])
            reference_embeddings[cls] = emb

        if not reference_embeddings:
            raise RuntimeError("No reference embeddings could be built from the dataset folder.")
        return reference_embeddings

    def _save_reference_embeddings_cache(
        self,
        cache_path: Path,
        reference_embeddings: Dict[str, torch.Tensor],
    ) -> None:
        ensure_dir(cache_path.parent)
        serializable_embeddings = {
            cls: emb.detach().cpu().squeeze(0) for cls, emb in reference_embeddings.items()
        }
        torch.save(
            {
                "version": REFERENCE_EMBEDDING_VERSION,
                "embedding_dim": EMBED_DIM,
                "reference_embeddings": serializable_embeddings,
            },
            str(cache_path),
        )

    def _load_reference_embeddings_from_cache(self, cache_path: Path) -> Dict[str, torch.Tensor]:
        payload = torch.load(str(cache_path), map_location=DEVICE)
        if not isinstance(payload, dict):
            raise ValueError("Reference embedding cache must contain a dictionary payload.")

        raw_embeddings = payload.get("reference_embeddings")
        if not isinstance(raw_embeddings, dict) or not raw_embeddings:
            raise ValueError("Reference embedding cache does not contain any saved embeddings.")

        reference_embeddings: Dict[str, torch.Tensor] = {}
        for cls, emb in raw_embeddings.items():
            tensor = emb if isinstance(emb, torch.Tensor) else torch.tensor(emb, dtype=torch.float32)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            reference_embeddings[str(cls)] = F.normalize(tensor.to(DEVICE), p=2, dim=1)
        return reference_embeddings

    def _map_label(self, label: str) -> str:
        return self.class_mapping.get(label, label)

    def _embed_image(self, image_path: Path) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model has not been loaded.")
        tensor = prepare_image_tensor(image_path, self.input_channels)
        with torch.inference_mode():
            emb = self.model(tensor)
        return emb

    def predict_char(self, image_path: Path) -> Tuple[str, float]:
        emb = self._embed_image(image_path)
        best_class = None
        best_score = -1.0
        for cls, ref_emb in self.reference_embeddings.items():
            score = float(F.cosine_similarity(emb, ref_emb).item())
            if score > best_score:
                best_score = score
                best_class = cls
        return best_class or "", best_score

    def predict_text(self, roi_paths: List[Path], fallback_image: Path) -> PredictionResult:
        targets = roi_paths if roi_paths else [fallback_image]
        tokens: List[Dict[str, object]] = []
        for roi in targets:
            label, score = self.predict_char(roi)
            mapped_label = self._map_label(label)
            tokens.append({"label": mapped_label, "raw_label": label, "score": score})

        text = "".join([token["label"] for token in tokens])
        confidence = float(np.mean([token["score"] for token in tokens])) if tokens else 0.0
        return PredictionResult(text=text, confidence=confidence, tokens=tokens)


def create_run_id() -> str:
    return uuid.uuid4().hex[:12]


def build_reference_embedding_cache(model_path: Path, dataset_path: Path, output_path: Path) -> Path:
    predictor = OCRPredictor(
        model_path=model_path,
        dataset_path=dataset_path,
        embedding_cache_path=output_path,
    )
    if not output_path.exists():
        raise RuntimeError(f"Failed to create reference embedding cache at: {output_path}")
    return output_path
