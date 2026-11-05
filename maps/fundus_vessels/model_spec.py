"""Everything specific to the VascX vessel model. ML engineers: this is your file.

The MAP's pipeline (app.py) and the shared operators (map_ops/) are model-agnostic.
Shipping a different model means revisiting every constant below, in the same change:

  ALGORITHM_VERSION  bump it; it is written into every SEG and is how results from
                     different models are told apart later.
  MODEL_SHA256       the new weights' checksum; the MAP won't run other weights.
  SEGMENT            what the mask means (label and codes).
  segment()          must reproduce the model's own inference exactly. Drift degrades
                     masks silently. VascX_App/compare_paper_model.py compares the App's copy with
                     the VascX reference implementation; run it after any change here.
  SELECTION_RULES    which series the model is allowed to see.

Promoted from the workspace prototype VascX_App/vessels.py: crop / vessel_probability /
segment are the same steps, plus DICOM decoding, selection and SEG codes here. When the
prototype changes, copy the change here, bump ALGORITHM_VERSION and run
tools/check_idempotency.py.

The model is third-party (VascX, AGPL-3.0); see VascX_App/README.md. Its reference
inference is rtnls_inference.SegmentationEnsemble. We reproduce the few steps it takes
for this model rather than install that package (Lightning, Albumentations, ...), but
use upstream's own fundus cropping (retinalysis-fundusprep), the step most likely to
drift if rewritten.

No Holoscan imports here, so this module can be tested on any host.
"""

import json
import os

import numpy as np
import pydicom
import torch
from pydicom.pixels import pixel_array
from pydicom.sr.codedict import codes
from rtnls_fundusprep.mask_extraction import get_cfi_bounds

# --- Identity --------------------------------------------------------------------------

ALGORITHM_NAME = "vascx-vessels"
# Our MAP version. The upstream model file is part of it: vessels_may26.pt, HF commit 962c83a.
ALGORITHM_VERSION = "0.2.0"

# Name of the model under /opt/holoscan/models: the folder in VascX_App/models/.
MODEL_NAME = "vessels"
# SHA-256 of the one .pt file in that folder (vessels_may26.pt). The MAP refuses to start on any
# other file: new weights mean editing this line, and with it ALGORITHM_VERSION.
MODEL_SHA256 = "9f24c987c020718b344f8c8bd5155be38299cc4e2f323efafa72e6bca48cb052"

# --- Output: what the SEG says ---------------------------------------------------------

SEGMENT = dict(
    label="Retinal blood vessels",
    category=codes.SCT.AnatomicalStructure,  # SCT 91723000
    type=codes.SCT.BloodVessel,  # SCT 59820001
    anatomic_region=codes.SCT.Retina,  # SCT 5665001
)

# --- Input: what the model may see -----------------------------------------------------

# Same rules as maps/aireadi_laterality (see its model_spec.py for why): colour fundus
# photographs only. Unlike the laterality model, VascX was trained on many devices and
# fields of view, so no device restriction.
SELECTION_RULES = json.dumps(
    {
        "selections": [
            {
                "name": "colour_fundus_photograph",
                "conditions": {"Modality": "^OP$", "ImageType": ["COLOR"]},
            }
        ]
    }
)

# Multi-field mosaics (e.g. AI-READI's Eidon "Mosaic", ~1.8:1) carry the same tags as single
# fields, but the fundus crop locks onto one circle, so only part of the image would be
# segmented. Single fields in AI-READI are at most 4:3 (1.33). Wider images are skipped.
MAX_ASPECT_RATIO = 1.6

# --- Inference: must match rtnls_inference for this model ------------------------------
# Values from the config embedded in vessels_may26.pt (VascX_App/README.md shows how to read it).

SQUARE_SIZE = 1024  # datamodule.test_transform.square_size; also inference.tracing_input_size
MEAN = [0.485, 0.456, 0.406]  # test_transform.normalize = "imagenet"
STD = [0.229, 0.224, 0.225]
TTA_FLIPS = [[2], [3], [2, 3]]  # inference.tta_flips: vertical, horizontal, both (NCHW dims)
VESSEL_CHANNEL = 1  # lightningmodule.n_class = 2: background, vessel


# Deterministic cuBLAS (ConvNeXt's MLP layers are matrix multiplies). Read when CUDA starts,
# so it has to be set before the model touches the GPU; importing this module does that.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class UnsupportedImage(ValueError):
    """The instance is not something this model can segment. Skipped, not fatal."""


def to_rgb(ds: pydicom.Dataset) -> np.ndarray:
    """Decode an instance to (H, W, 3) uint8 RGB.

    pydicom 3 converts YBR to RGB itself; no second conversion here (unlike the laterality
    model, VascX was trained on correctly decoded images).
    """
    image_type = [str(v).upper() for v in ds.get("ImageType", [])]
    if "COLOR" not in image_type:
        raise UnsupportedImage(f"ImageType={image_type}, not a colour photograph")
    if int(ds.get("SamplesPerPixel", 1)) != 3:
        raise UnsupportedImage(f"SamplesPerPixel={ds.get('SamplesPerPixel')}, expected 3 (colour)")
    if int(ds.get("NumberOfFrames", 1) or 1) != 1:
        raise UnsupportedImage(f"NumberOfFrames={ds.get('NumberOfFrames')}, expected 1")
    if ds.Columns / ds.Rows > MAX_ASPECT_RATIO:
        raise UnsupportedImage(f"{ds.Columns}x{ds.Rows} is wider than one field; likely a mosaic")
    return pixel_array(ds)


def crop(rgb: np.ndarray):
    """Find the fundus circle and warp it to SQUARE_SIZE x SQUARE_SIZE (upstream fundusprep).

    Returns the cropped image and the transform, whose warp_inverse maps results back.
    """
    transform, _ = get_cfi_bounds(rgb).crop(SQUARE_SIZE)
    return transform.warp(rgb, (SQUARE_SIZE, SQUARE_SIZE)), transform


def vessel_probability(model, cropped: np.ndarray, device: torch.device) -> np.ndarray:
    """(SQUARE_SIZE, SQUARE_SIZE, 3) uint8 -> (SQUARE_SIZE, SQUARE_SIZE) float32 vessel probability.

    As SegmentationEnsemble with tta=True: logits averaged over the image and its three
    flips, then softmax. Upstream runs MONAI sliding_window_inference with a window the
    size of the image, which is one plain forward pass, so we call the model directly.
    Full precision, deterministic cuDNN and cuBLAS (CUBLAS_WORKSPACE_CONFIG, set at import if
    unset), no TorchScript re-optimisation: the same image gives a bit-identical result on
    every run (on the same GPU type), so re-runs reproduce the same SEG. Upstream uses
    half precision (autocast), which flips a few near-threshold pixels from run to run.
    """
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


def segment(model, ds: pydicom.Dataset, device: torch.device) -> np.ndarray:
    """Instance -> (Rows, Columns) uint8 mask, 1 = vessel, on the source image's own pixel grid."""
    rgb = to_rgb(ds)
    cropped, transform = crop(rgb)
    probability = vessel_probability(model, cropped, device)
    # Map the probability back, then threshold: same as argmax for two classes.
    original = transform.warp_inverse(probability, rgb.shape[:2])
    return (original > 0.5).astype(np.uint8)
