"""VascX vessel segmentation on an RGB fundus photo: the workspace prototype.

    import vessels
    model = vessels.load_model()
    mask = vessels.segment(model, rgb)          # rgb: (H, W, 3) uint8

Reproduces what upstream's rtnls_inference.SegmentationEnsemble does for vessels_may26.pt,
in a few readable steps. maps/fundus_vessels/model_spec.py is the promoted copy of this
code (plus DICOM decoding, selection rules and SEG codes); change it there too when this
changes, and re-run compare_paper_model.py.
"""

import os
from pathlib import Path

import numpy as np
import torch
from rtnls_fundusprep.mask_extraction import get_cfi_bounds

# Deterministic cuBLAS (ConvNeXt's MLP layers are matrix multiplies). Read when CUDA starts,
# so it has to be set before the model touches the GPU; importing this module does that.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

MODEL_FILE = Path(__file__).resolve().parent / "models" / "vessels" / "vessels_may26.pt"

# From the config embedded in the model file (see README.md for how to read it).
SQUARE_SIZE = 1024  # datamodule.test_transform.square_size; also inference.tracing_input_size
MEAN = [0.485, 0.456, 0.406]  # test_transform.normalize = "imagenet"
STD = [0.229, 0.224, 0.225]
TTA_FLIPS = [[2], [3], [2, 3]]  # inference.tta_flips: vertical, horizontal, both (NCHW dims)
VESSEL_CHANNEL = 1  # lightningmodule.n_class = 2: background, vessel


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path: Path = MODEL_FILE, device: torch.device = None):
    return torch.jit.load(str(path), map_location=device or default_device()).eval()


def crop(rgb: np.ndarray):
    """Find the fundus circle and warp it to SQUARE_SIZE x SQUARE_SIZE (upstream fundusprep).

    Returns the cropped image and the transform, whose warp_inverse maps results back.
    """
    transform, _ = get_cfi_bounds(rgb).crop(SQUARE_SIZE)
    return transform.warp(rgb, (SQUARE_SIZE, SQUARE_SIZE)), transform


def vessel_probability(model, cropped: np.ndarray, device: torch.device = None) -> np.ndarray:
    """(SQUARE_SIZE, SQUARE_SIZE, 3) uint8 -> (SQUARE_SIZE, SQUARE_SIZE) float32 vessel probability.

    Logits averaged over the image and its three flips, then softmax.
    Full precision, deterministic cuDNN and cuBLAS (CUBLAS_WORKSPACE_CONFIG, set at import if
    unset), no TorchScript re-optimisation: the same image gives a bit-identical result on
    every run (on the same GPU type), so re-runs reproduce the same SEG. Upstream uses
    half precision (autocast), which flips a few near-threshold pixels from run to run.
    """
    device = device or default_device()
    x = torch.from_numpy(cropped).permute(2, 0, 1).float().div_(255.0)
    x = ((x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1))[None].to(device)

    def forward(img):
        return model(img)[0].float()  # the file stacks ensemble members first: (M=1, N, C, H, W)

    torch.backends.cudnn.benchmark = False  # benchmark picks kernels per run; not repeatable
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)  # any op without a deterministic kernel raises, not drifts
    # TorchScript re-optimises the graph after its first calls, which changes the numbers
    # slightly: the first image of a run would differ from a re-run. Use the plain graph.
    with torch.no_grad(), torch.jit.optimized_execution(False):
        logits = forward(x)
        for dims in TTA_FLIPS:
            logits += torch.flip(forward(torch.flip(x, dims)), dims)
        logits /= len(TTA_FLIPS) + 1
    return torch.softmax(logits.float(), dim=1)[0, VESSEL_CHANNEL].cpu().numpy()


def segment(model, rgb: np.ndarray, device: torch.device = None) -> np.ndarray:
    """(H, W, 3) uint8 -> (H, W) uint8 mask, 1 = vessel, on the input's own pixel grid."""
    cropped, transform = crop(rgb)
    probability = vessel_probability(model, cropped, device)
    return (transform.warp_inverse(probability, rgb.shape[:2]) > 0.5).astype(np.uint8)
