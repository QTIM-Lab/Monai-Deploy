"""Everything specific to the laterality model. ML engineers: this is your file.

The MAP's pipeline (app.py) and the shared operators (map_ops/) are model-agnostic.
Shipping a new model.ts means revisiting every constant below, in the same commit:

  ALGORITHM_VERSION  bump it; it is written into every SR and is how results from
                     different models are told apart later.
  CLASSES            must be in the model's output index order. Nothing inside
                     model.ts records this, so a mismatch gives confident wrong
                     answers, not an error.
  to_rgb/preprocess  must reproduce training exactly. Drift degrades accuracy
                     silently. tools/check_preprocessing.py compares against the
                     training cache bit-for-bit; run it after any change here.
  SELECTION_RULES    which series the model is allowed to see.

No Holoscan imports here, so this module can be tested on any host.
"""

import json

import numpy as np
import pydicom
import torch
from PIL import Image
from pydicom.pixel_data_handlers.util import convert_color_space
from pydicom.pixels import pixel_array
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code

# --- Identity --------------------------------------------------------------------------

ALGORITHM_NAME = "aireadi-laterality"
ALGORITHM_VERSION = "0.1.0"

# Name of the model under /opt/holoscan/models. "" means the only model packaged.
MODEL_NAME = ""

# --- Output: what the SR says ----------------------------------------------------------

# Finding: SCT 272741003 "Laterality". Values: CID 244 laterality qualifiers.
FINDING = codes.SCT.Laterality

# Index order of the model's logits. Training used CLASSES = ["L", "R"] (prepare_data.py).
CLASSES = [
    codes.SCT.Left,  # 0 -> SCT 7771000
    codes.SCT.Right,  # 1 -> SCT 24028007
]

CONFIDENCE = codes.DCM.Probability  # DCM 122157, softmax probability of the chosen class
UNITY = Code("1", "UCUM", "no units")

# No verified standard code for this procedure is in pydicom's dictionary; private scheme.
PROCEDURE = Code("AI-FUNDUS-LATERALITY", "99QTIM", "AI laterality of fundus photograph")

# --- Input: what the model may see -----------------------------------------------------

# DICOMSeriesSelectorOperator rules. Anchor regexes: the selector falls back to an
# unanchored re.search, so "OP" would also match OCT's "OPT".
# The model was trained on Topcon Maestro2 macula-centred colour photographs only. To
# restrict to that domain, add  "ManufacturerModelName": "Maestro2"  to the conditions
# (matched against the first instance of each series).
SELECTION_RULES = json.dumps(
    {
        "selections": [
            {
                "name": "colour_fundus_photograph",
                "conditions": {"Modality": "^OP$"},
            }
        ]
    }
)

# --- Preprocessing: must match prepare_data.py + train.py ------------------------------

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# KNOWN TRAINING DEFECT, reproduced on purpose. pydicom 3 already returns RGB for YBR
# images; prepare_data.py then converts YBR->RGB a second time (its comment describes
# pydicom 2). The model learned from those miscoloured images, so inference must do the
# same until the model is retrained with correct decoding. When retraining: fix
# prepare_data.py, set this False, bump ALGORITHM_VERSION, all in one commit.
# This also pins the MAP to pydicom 3.x (requirements.txt); pydicom 2 decodes differently.
REPRODUCE_TRAINING_DOUBLE_YBR_CONVERSION = True


class UnsupportedImage(ValueError):
    """The instance is not something this model can score. Skipped, not fatal."""


def to_rgb(ds: pydicom.Dataset) -> np.ndarray:
    """Decode an instance to (H, W, 3) uint8 exactly as prepare_data.load_one did.

    Uses pydicom's functional pixel_array, which does not cache the decoded array on the
    dataset; ds.pixel_array would keep ~15 MB per image alive for the whole batch.
    """
    if int(ds.get("SamplesPerPixel", 1)) != 3:
        raise UnsupportedImage(f"SamplesPerPixel={ds.get('SamplesPerPixel')}, expected 3 (colour)")
    if int(ds.get("NumberOfFrames", 1) or 1) != 1:
        raise UnsupportedImage(f"NumberOfFrames={ds.get('NumberOfFrames')}, expected 1")

    arr = pixel_array(ds)
    if REPRODUCE_TRAINING_DOUBLE_YBR_CONVERSION and ds.PhotometricInterpretation.startswith("YBR"):
        arr = convert_color_space(arr, ds.PhotometricInterpretation, "RGB")
    return arr


def resize(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (IMAGE_SIZE, IMAGE_SIZE, 3) uint8. The training cache holds this."""
    img = Image.fromarray(rgb).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def preprocess(ds: pydicom.Dataset) -> torch.Tensor:
    """Instance -> normalised (3, IMAGE_SIZE, IMAGE_SIZE) float32, as train.load_split."""
    x = torch.from_numpy(resize(to_rgb(ds))).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x - mean) / std
