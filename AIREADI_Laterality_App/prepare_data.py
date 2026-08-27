# Reads the AI-READI retinal photography manifest, picks one device and one anatomic
# region, splits by PATIENT, decodes the DICOMs once, and caches downsized RGB arrays.
#
# Decoding is ~400 ms per image, so doing it every epoch would dominate training time.
# This runs once and writes .npy files that load into RAM instantly.

import os
import csv
import random
import shutil
from multiprocessing import Pool

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import convert_color_space

# - Configuration - #
# Maestro2 + Macula is one device with one framing, so laterality is always read from the
# same cue (disc position relative to the fovea). Mixing in Optic Disc or Temporal
# Periphery views would make the task inconsistent for no benefit.
DATA_ROOT = os.environ.get("AIREADI_ROOT", "/persist/public_datasets/ai_readi")
MANIFEST = os.path.join(DATA_ROOT, "retinal_photography", "manifest.tsv")

DEVICE = "Maestro2"
REGION = "Macula"
IMAGING = "Color Photography"

IMAGE_SIZE = 224  # ImageNet-pretrained backbones expect roughly this
CLASSES = ["L", "R"]
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
N_DEMO = 3  # DICOMs held out untouched, for the live demo
SEED = 0
WORKERS = 12

CACHE_DIR = os.path.join(os.path.curdir, "cache")
DEMO_DIR = os.path.join(os.path.curdir, "demo_input")


def read_manifest():
    """Return the manifest rows matching the configured device / region / imaging type."""
    with open(MANIFEST) as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t")]

    rows = [
        r
        for r in rows
        if r["manufacturers_model_name"] == DEVICE
        and r["anatomic_region"] == REGION
        and r["imaging"] == IMAGING
        and r["laterality"] in CLASSES
    ]
    print(f"{len(rows)} images match {DEVICE} / {REGION} / {IMAGING}")
    return rows


def split_by_patient(rows):
    """Assign every row a split, keeping all of a patient's images on the same side.

    Splitting by image would put the same eye in train and test, since each patient
    contributes both eyes and sometimes repeat captures. The accuracy number would
    then be measuring memorisation.
    """
    patients = sorted({r["participant_id"] for r in rows})
    random.Random(SEED).shuffle(patients)

    n_train = int(len(patients) * SPLITS["train"])
    n_val = int(len(patients) * SPLITS["val"])
    assignment = {}
    for i, p in enumerate(patients):
        if i < n_train:
            assignment[p] = "train"
        elif i < n_train + n_val:
            assignment[p] = "val"
        else:
            assignment[p] = "test"

    for r in rows:
        r["split"] = assignment[r["participant_id"]]

    print(f"{len(patients)} patients ->", {s: sum(a == s for a in assignment.values()) for s in SPLITS})
    return rows


def load_one(row):
    """Decode a single DICOM to an (IMAGE_SIZE, IMAGE_SIZE, 3) uint8 RGB array."""
    ds = pydicom.dcmread(os.path.join(DATA_ROOT + row["filepath"]))
    arr = ds.pixel_array

    # pydicom hands back the stored colour space, it does not convert. Fundus DICOMs in
    # this dataset are often YBR_FULL_422, which looks wrong if treated as RGB.
    if ds.PhotometricInterpretation.startswith("YBR"):
        arr = convert_color_space(arr, ds.PhotometricInterpretation, "RGB")

    img = Image.fromarray(arr).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def build_cache(rows, split):
    subset = [r for r in rows if r["split"] == split]
    print(f"decoding {len(subset)} images for '{split}' with {WORKERS} workers...")

    with Pool(WORKERS) as pool:
        images = pool.map(load_one, subset)

    x = np.stack(images)
    y = np.array([CLASSES.index(r["laterality"]) for r in subset], dtype=np.int64)

    np.save(os.path.join(CACHE_DIR, f"{split}_x.npy"), x)
    np.save(os.path.join(CACHE_DIR, f"{split}_y.npy"), y)
    print(f"  {split}: {x.shape} {x.dtype}  L={int((y == 0).sum())} R={int((y == 1).sum())}")


def copy_demo_files(rows):
    """Copy a few untouched test-split DICOMs aside to run through the MAP live."""
    picks = [r for r in rows if r["split"] == "test"][:N_DEMO]
    for r in picks:
        src = os.path.join(DATA_ROOT + r["filepath"])
        shutil.copy(src, os.path.join(DEMO_DIR, os.path.basename(src)))
        print(f"  demo: {os.path.basename(src)}  (truth = {r['laterality']})")


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DEMO_DIR, exist_ok=True)

    rows = split_by_patient(read_manifest())

    # Keep a record of exactly which file went where, so results stay reproducible.
    with open(os.path.join(CACHE_DIR, "split.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "laterality", "split", "filepath"])
        for r in rows:
            w.writerow([r["participant_id"], r["laterality"], r["split"], r["filepath"]])

    for split in SPLITS:
        build_cache(rows, split)

    copy_demo_files(rows)
    print(f"\nCache written to {CACHE_DIR}, demo DICOMs in {DEMO_DIR}")
