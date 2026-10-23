"""Several photos of one place -> cameras + fused metric points.

VGGT (https://github.com/facebookresearch/vggt) estimates every photo's camera and a depth map up to one global scale.
Each photo is then lifted with MoGe at the field of view VGGT found for it, and the MoGe points are placed at the
VGGT camera with a per-photo scale that matches VGGT's depth. The VGGT unit is converted to metres with the median
MoGe/VGGT depth ratio over all photos, so cameras and points come out in metres in the first photo's camera frame.
"""
import io
import json
import struct
import threading

import numpy as np
from PIL import Image

from pctool import lift as lifting

VGGT_REPO = "facebook/VGGT-1B"
SIZE = 518   # VGGT input: longest side, padded to a square (sides are multiples of its 14 px patch)
_model = None
_lock = threading.Lock()


def _vggt():
    global _model
    if _model is None:
        import torch
        from vggt.models.vggt import VGGT

        _model = VGGT.from_pretrained(VGGT_REPO).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return _model


def _pad_square(pil: Image.Image) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Resize so the longest side is SIZE, pad to SIZE x SIZE (white, as VGGT's own "pad" mode); content box."""
    s = SIZE / max(pil.size)
    w, h = max(1, round(pil.width * s)), max(1, round(pil.height * s))
    canvas = np.ones((SIZE, SIZE, 3), np.float32)
    x0, y0 = (SIZE - w) // 2, (SIZE - h) // 2
    canvas[y0:y0 + h, x0:x0 + w] = np.asarray(pil.resize((w, h), Image.BICUBIC), np.float32) / 255
    return canvas, (x0, y0, w, h)


def reconstruct(images: list[bytes], model_name: str = lifting.DEFAULT_MODEL, max_side: int = 1024) -> bytes:
    """Wire format: uint32 count | per photo: uint32 length | a `pctool.lift` blob whose header also has `c2w`
    (OpenCV camera-to-world, world = first photo's camera, metres), `scale` (multiply the blob's points) and
    `vggt` diagnostics."""
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if len(images) < 2:
        raise ValueError("multi-view reconstruction needs at least 2 photos")
    pils = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
    padded = [_pad_square(p) for p in pils]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = torch.tensor(np.stack([p for p, _ in padded]), device=device).permute(0, 3, 1, 2)
    with _lock, torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        out = _vggt()(batch)
    w2c, K = pose_encoding_to_extri_intri(out["pose_enc"], batch.shape[-2:])
    w2c, K = w2c[0].float().cpu().numpy(), K[0].float().cpu().numpy()
    depth = out["depth"][0, ..., 0].float().cpu().numpy()
    conf = out["depth_conf"][0].float().cpu().numpy()

    blobs, ratios, fovs = [], [], []
    for i, (blob_image, (_, (x0, y0, w, h))) in enumerate(zip(images, padded)):
        fov_x = float(np.degrees(2 * np.arctan(w / 2 / K[i, 0, 0])))
        blob = lifting.lift(blob_image, model_name, fov_x, max_side)
        header, index, points = _unpack(blob)
        # VGGT depth at each MoGe pixel (nearest), confident half of the VGGT pixels only
        wm, hm = header["width"], header["height"]
        u = x0 + ((index % wm) + 0.5) * w / wm
        v = y0 + ((index // wm) + 0.5) * h / hm
        dv = depth[i, np.clip(v.astype(int), 0, SIZE - 1), np.clip(u.astype(int), 0, SIZE - 1)]
        cv = conf[i, np.clip(v.astype(int), 0, SIZE - 1), np.clip(u.astype(int), 0, SIZE - 1)]
        keep = (cv >= np.median(conf[i, y0:y0 + h, x0:x0 + w])) & (dv > 0)
        ratios.append(float(np.median(dv[keep] / points[keep, 2])) if keep.sum() > 100 else float("nan"))
        blobs.append((header, blob))
        fovs.append(fov_x)
    valid = [r for r in ratios if np.isfinite(r)]
    if not valid:
        raise ValueError("could not match MoGe and VGGT depths in any photo")
    metres = float(np.median([1 / r for r in valid]))   # metres per VGGT unit
    result = [struct.pack("<I", len(blobs))]
    for i, (header, blob) in enumerate(blobs):
        c2w = np.eye(4)
        c2w[:3, :3] = w2c[i, :, :3].T
        c2w[:3, 3] = -w2c[i, :, :3].T @ w2c[i, :, 3] * metres
        ratio = ratios[i] if np.isfinite(ratios[i]) else 1 / metres
        header.update(c2w=c2w.tolist(), scale=ratio * metres,
                      vggt=dict(fov_x=fovs[i], depth_ratio=ratio, metres_per_unit=metres))
        body = _repack(header, blob)
        result += [struct.pack("<I", len(body)), body]
    return b"".join(result)


def _unpack(blob: bytes) -> tuple[dict, np.ndarray, np.ndarray]:
    n = struct.unpack_from("<I", blob)[0]
    header = json.loads(blob[4:4 + n])
    count, o = header["count"], 4 + n
    index = np.frombuffer(blob, np.uint32, count, o)
    points = np.frombuffer(blob, np.float32, 3 * count, o + 4 * count).reshape(-1, 3)
    return header, index, points


def _repack(header: dict, blob: bytes) -> bytes:
    n = struct.unpack_from("<I", blob)[0]
    head = json.dumps(header).encode()
    return struct.pack("<I", len(head)) + head + blob[4 + n:]
