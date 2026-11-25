"""Everything specific to the disc + fovea detector. ML engineers: this is your file.

The MAP's pipeline (app.py) and the shared operators (map_ops/) are model-agnostic.
Shipping a different model means revisiting every constant below, in the same change:

  ALGORITHM_VERSION  bump it; it is written into every SR and is how results from
                     different models are told apart later.
  MODEL_SHA256       the new weights' checksum; the MAP won't run other weights.
  CLASSES            must be in the model's label order (train.py CLASSES). Nothing inside
                     the model file records this.
  crop/detect        must reproduce training exactly (prepare_cache.py + train.py).
  SELECTION_RULES    which series the model is allowed to see.

Promoted from the App prototype Disc_Detection_App/detector.py: crop / boxes_in_crop / the
inverse mapping are the same steps, plus DICOM decoding, selection and SR codes here. When the
prototype changes, copy the change here, bump ALGORITHM_VERSION and run tools/check_idempotency.py.
The model was chosen and checked in Disc_Detection_App/ (explore_labels.ipynb Part 2, handcheck.py).

No Holoscan imports here, so this module can be tested on any host.
"""

import json
import os

import numpy as np
import pydicom
import torch
import torchvision  # noqa: F401  registers the torchvision:: ops (NMS, box ops) the TorchScript model calls
from pydicom.pixels import pixel_array
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code
from rtnls_fundusprep.mask_extraction import get_cfi_bounds

from map_ops.image_finding import BoxFinding, ImageFinding, NumericMeasurement

# --- Identity --------------------------------------------------------------------------

ALGORITHM_NAME = "disc-fovea-detection"
# Our MAP version. The model file is part of it: torchvision RetinaNet (ResNet-50 FPN v2), run
# retinanet_20261105-151050, exported to TorchScript by Disc_Detection_App/export_model.py.
ALGORITHM_VERSION = "0.1.0"

# Name of the model under /opt/holoscan/models: the folder in Disc_Detection_App/export/.
MODEL_NAME = "retinanet"
# SHA-256 of the one .pt file in that folder. The MAP refuses to start on any other file: new
# weights mean editing this line, and with it ALGORITHM_VERSION.
MODEL_SHA256 = "a25f87a10367598bb176ec5d6925447df626a637d1a78af7e995620078fc1e64"

# --- Output: what the SR says ----------------------------------------------------------

# Model label index -> the structure it locates. Training: CLASSES = ["background", "disc", "fovea"].
CLASSES = {
    1: codes.SCT.OpticDiscStructure,  # SCT 81016008
    2: codes.SCT.StructureOfFoveaCentralis,  # SCT 67046006
}
# The detector's score for the box (0..1), not a calibrated probability.
SCORE = codes.DCM.CertaintyOfFinding  # DCM 111012
UNITY = Code("1", "UCUM", "no units")

# No verified standard code for this procedure is in pydicom's dictionary; private scheme.
PROCEDURE = Code("AI-FUNDUS-DISC-FOVEA", "99QTIM", "AI optic disc and fovea detection on fundus photograph")

# --- Input: what the model may see -----------------------------------------------------

# Same rules as maps/aireadi_laterality (see its model_spec.py for why): colour fundus photographs.
# Trained on AI-READI standard fields from four cameras (Aurora; Eidon, Maestro2, Triton macula;
# Triton disc). Peripheral fields are not excluded: the disc may be in view, and the score says
# how sure the model is. Mosaics are skipped (below).
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

# Multi-field mosaics (e.g. Eidon "Mosaic", ~1.8:1): the fundus crop locks onto one circle. Same
# cut-off as maps/fundus_vessels.
MAX_ASPECT_RATIO = 1.6

# --- Inference: must match prepare_cache.py + train.py ---------------------------------

SIZE = 512  # prepare_cache.SIZE: the fundusprep crop the model was trained on
# Rounding makes re-runs identical: GPU results can differ in the last bits between runs or batch
# shapes, and the SR must not. 0.1 px and 4 places are far below what the model can resolve.
COORD_DECIMALS = 1
SCORE_DECIMALS = 4

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; read when CUDA starts


class UnsupportedImage(ValueError):
    """The instance is not something this model can process. Skipped, not fatal."""


def to_rgb(ds: pydicom.Dataset) -> np.ndarray:
    """Decode an instance to (H, W, 3) uint8 RGB, as prepare_cache.py did (pydicom 3, no extra conversion)."""
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
    """Fundus circle warped to SIZE x SIZE (upstream fundusprep), and the transform that did it."""
    transform, _ = get_cfi_bounds(rgb).crop(SIZE)
    return transform.warp(rgb, (SIZE, SIZE)), transform


def boxes_in_crop(model, cropped: np.ndarray, device: torch.device) -> dict:
    """{label: (x0, y0, x1, y1, score)} in crop pixels: the top-scoring box per class.

    torchvision keeps boxes scoring >= 0.05; a class with none is reported as not found.
    """
    torch.backends.cudnn.benchmark = False  # benchmark picks kernels per run; not repeatable
    torch.backends.cudnn.deterministic = True
    x = torch.from_numpy(cropped).permute(2, 0, 1).float().div_(255.0).to(device)  # the model normalises
    with torch.no_grad():
        output = model([x])
    output = (output[1] if isinstance(output, tuple) else output)[0]  # TorchScript returns (losses, detections)
    found = {}
    for label in CLASSES:
        mine = (output["labels"] == label).nonzero().flatten()
        if len(mine):
            best = mine[output["scores"][mine].argmax()]
            found[label] = (*output["boxes"][best].tolist(), float(output["scores"][best]))
    return found


def detect(model, ds: pydicom.Dataset, device: torch.device) -> ImageFinding:
    """Instance -> ImageFinding: a box per structure found, in the source image's own pixels, or 'absent'."""
    rgb = to_rgb(ds)
    cropped, transform = crop(rgb)
    found = boxes_in_crop(model, cropped, device)
    boxes, not_found = [], []
    for label, structure in CLASSES.items():
        if label not in found:
            not_found.append(structure)
            continue
        x0, y0, x1, y1, score = found[label]
        (ox0, oy0), (ox1, oy1) = transform.apply_inverse(np.array([[x0, y0], [x1, y1]]))
        # Clipped to the image: a box can poke past the edge of a photo whose fundus circle is cut off.
        ox0, ox1 = np.clip([ox0, ox1], 0, ds.Columns)
        oy0, oy1 = np.clip([oy0, oy1], 0, ds.Rows)
        box = tuple(round(float(v), COORD_DECIMALS) for v in (ox0, oy0, ox1, oy1))
        boxes.append(BoxFinding(structure, box, [NumericMeasurement(SCORE, round(score, SCORE_DECIMALS), UNITY)]))
    return ImageFinding(source=ds, boxes=boxes, not_found=not_found)
