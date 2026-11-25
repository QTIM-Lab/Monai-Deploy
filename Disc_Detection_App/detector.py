"""Disc + fovea boxes on an RGB fundus photo with the exported RetinaNet: the workspace prototype.

    import detector
    model = detector.load_model()                # export/retinanet/*.pt, written by export_model.py
    boxes = detector.detect(model, rgb)          # rgb: (H, W, 3) uint8, as pydicom 3 decodes it
    # {"disc": (x0, y0, x1, y1, score), "fovea": (...)}, in the photo's own pixels; a class is absent
    # when the model returned no box for it

The same steps as training (prepare_cache.py + train.py): upstream's fundusprep crop to 512 x 512,
the model on the [0, 1] crop (torchvision normalises inside the model), the top-scoring box per
class, then the box mapped back through the crop's inverse transform. The crop is scale plus
translation only, so boxes stay axis-aligned.

maps/disc_detection/model_spec.py is the promoted copy of this code (plus DICOM decoding,
selection rules and SR codes); change it there too when this changes.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torchvision  # noqa: F401  registers the torchvision:: ops the TorchScript model calls (NMS, box ops)
from rtnls_fundusprep.mask_extraction import get_cfi_bounds

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; read when CUDA starts

# Packaged into the MAP as is (monai-deploy package -m export): one folder per model, one file in it.
MODEL_DIR = Path(__file__).resolve().parent / "export" / "retinanet"
SIZE = 512  # prepare_cache.SIZE
CLASSES = ["background", "disc", "fovea"]  # train.CLASSES: label index order
COORD_DECIMALS = 1  # boxes rounded to 0.1 px and scores to 4 places, so re-runs give identical results
SCORE_DECIMALS = 4


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: torch.device = None):
    files = sorted(MODEL_DIR.glob("*.pt"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one exported model in {MODEL_DIR}, found {files}: run export_model.py")
    return torch.jit.load(str(files[0]), map_location=device or default_device()).eval()


def crop(rgb: np.ndarray):
    """Fundus circle warped to SIZE x SIZE (upstream fundusprep), and the transform that did it."""
    transform, _ = get_cfi_bounds(rgb).crop(SIZE)
    return transform.warp(rgb, (SIZE, SIZE)), transform


def boxes_in_crop(model, cropped: np.ndarray, device: torch.device = None) -> dict:
    """{class: (x0, y0, x1, y1, score)} in crop pixels: the top-scoring box per class."""
    device = device or default_device()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    x = torch.from_numpy(cropped).permute(2, 0, 1).float().div_(255.0).to(device)
    with torch.no_grad():
        output = model([x])
    output = (output[1] if isinstance(output, tuple) else output)[0]  # TorchScript returns (losses, detections)
    found = {}
    for k, name in enumerate(CLASSES[1:], start=1):
        mine = (output["labels"] == k).nonzero().flatten()
        if len(mine):
            best = mine[output["scores"][mine].argmax()]
            found[name] = (*output["boxes"][best].tolist(), float(output["scores"][best]))
    return found


def detect(model, rgb: np.ndarray, device: torch.device = None) -> dict:
    """{class: (x0, y0, x1, y1, score)} in the photo's own pixels, rounded (see COORD_DECIMALS)."""
    cropped, transform = crop(rgb)
    out = {}
    for name, (x0, y0, x1, y1, score) in boxes_in_crop(model, cropped, device).items():
        (ox0, oy0), (ox1, oy1) = transform.apply_inverse(np.array([[x0, y0], [x1, y1]]))
        out[name] = tuple(round(float(v), COORD_DECIMALS) for v in (ox0, oy0, ox1, oy1)) + (round(score, SCORE_DECIMALS),)
    return out
