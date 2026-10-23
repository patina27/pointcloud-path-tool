"""Image -> metric point map with MoGe (https://github.com/microsoft/MoGe), packed for the web client.

Wire format of `lift()`: uint32 header length | JSON header | uint32 pixel index (N) | float32 points (N, 3) |
uint8 colors (N, 3). Points are OpenCV camera coordinates (x right, y down, z forward, metres) of the valid pixels of
the image resized to at most `max_side` px; the header carries width, height, pixel intrinsics K, fov_x and count.
"""
import io
import json
import struct
import threading

import numpy as np
from PIL import Image

MODELS: dict[str, tuple[str, str]] = {
    "moge-3-vitl": ("v3", "Ruicheng/moge-3-vitl"),
    "moge-3-vitg": ("v3", "Ruicheng/moge-3-vitg"),
    "moge-2-vitl-normal": ("v2", "Ruicheng/moge-2-vitl-normal"),
    "moge-2-vitl": ("v2", "Ruicheng/moge-2-vitl"),
}
DEFAULT_MODEL = "moge-3-vitl"
_loaded = {}
_lock = threading.Lock()


def available() -> list[str]:
    """Models whose MoGe version imports here (MoGe-3 needs the FlexGEMM extension)."""
    names = []
    for name, (version, _) in MODELS.items():
        try:
            __import__(f"moge.model.{version}")
        except ImportError:
            continue
        names.append(name)
    return names


def _model(name: str):
    import importlib

    import torch

    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    if name not in _loaded:
        version, repo = MODELS[name]
        cls = importlib.import_module(f"moge.model.{version}").MoGeModel
        _loaded[name] = cls.from_pretrained(repo).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return _loaded[name]


def health() -> dict:
    import torch

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"ok": True, "models": sorted(_loaded), "available": available(), "default": DEFAULT_MODEL, "device": device}


def lift(image_bytes: bytes, model_name: str = DEFAULT_MODEL, fov_x: float | None = None, max_side: int = 1024) -> bytes:
    """Lift one encoded image; `fov_x` (degrees) fixes the horizontal field of view instead of estimating it."""
    import torch

    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    scale = min(1.0, max_side / max(pil.size))
    if scale < 1:
        pil = pil.resize((round(pil.width * scale), round(pil.height * scale)), Image.BICUBIC)
    rgb = np.asarray(pil)
    h, w = rgb.shape[:2]
    with _lock, torch.no_grad():
        model = _model(model_name)
        device = next(model.parameters()).device
        tensor = torch.tensor(rgb / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        out = model.infer(tensor, fov_x=fov_x) if fov_x else model.infer(tensor)
    points = out["points"].float().cpu().numpy()
    mask = out["mask"].cpu().numpy() > 0.5 if out.get("mask") is not None else np.ones((h, w), bool)
    mask &= np.isfinite(points).all(-1) & (points[..., 2] > 0)
    index = np.flatnonzero(mask.reshape(-1)).astype(np.uint32)
    K = out["intrinsics"].float().cpu().numpy().copy()   # normalized -> pixels
    K[0] *= w
    K[1] *= h
    header = json.dumps(dict(width=w, height=h, K=K.tolist(), model=model_name, count=int(index.size),
                             fov_x=float(np.degrees(2 * np.arctan(w / 2 / K[0, 0]))))).encode()
    return (struct.pack("<I", len(header)) + header + index.tobytes()
            + points.reshape(-1, 3)[index].astype(np.float32).tobytes() + rgb.reshape(-1, 3)[index].tobytes())
