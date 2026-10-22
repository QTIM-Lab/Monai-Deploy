"""Assemble a batch test input from held-out AI-READI patients, plus distractors.

    python maps/aireadi_laterality/tools/make_test_batch.py OUT_DIR [--patients 10]

Layout mirrors what MONAI Deploy's Task Manager stages: <study>/<series>/<sop>.dcm with
a .dcm.json sidecar next to each file. Contents:

  - every Maestro2 macula colour photo of N test-split patients (should be scored);
  - one OCT volume, Modality OPT (must be skipped: guards the ^OP$ selection rule);
  - a stray non-DICOM file (must be ignored by the loader).

Test-split patients only, so accuracy on this batch is not inflated by training data.
Ground truth stays in each file's ImageLaterality tag; tools/check_batch.py reads it.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import pydicom

REPO = Path(__file__).resolve().parents[3]
SPLIT = REPO / "AIREADI_Laterality_App" / "cache" / "split.csv"
DATA_ROOT = os.environ.get("AIREADI_ROOT", "/persist/public_datasets/ai_readi")
OCT_MANIFEST = Path(DATA_ROOT) / "retinal_oct" / "manifest.tsv"


def stage(src: Path, out: Path) -> Path:
    ds = pydicom.dcmread(src, stop_before_pixels=True)
    dest_dir = out / ds.StudyInstanceUID / ds.SeriesInstanceUID
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{ds.SOPInstanceUID}.dcm"
    shutil.copy(src, dest)
    # Task Manager writes a JSON sidecar per instance; the loader must tolerate it.
    (dest_dir / f"{ds.SOPInstanceUID}.dcm.json").write_text(json.dumps({"SOPInstanceUID": ds.SOPInstanceUID}))
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--patients", type=int, default=10)
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        print(f"{args.out_dir} is not empty; refusing to mix batches", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    test_rows = [r for r in csv.DictReader(open(SPLIT)) if r["split"] == "test"]
    patients = sorted({r["participant_id"] for r in test_rows})[: args.patients]
    photos = [r for r in test_rows if r["participant_id"] in patients]
    for row in photos:
        stage(Path(DATA_ROOT + row["filepath"]), args.out_dir)

    with open(OCT_MANIFEST) as f:
        oct_row = next(r for r in csv.DictReader(f, delimiter="\t") if r["participant_id"] in patients)
    stage(Path(DATA_ROOT + oct_row["filepath"]), args.out_dir)

    (args.out_dir / "README.txt").write_text("not DICOM; the loader must skip this\n")

    left = sum(r["laterality"] == "L" for r in photos)
    print(f"staged {len(photos)} photos (L={left}, R={len(photos) - left}) from {len(patients)} patients")
    print(f"       + 1 OCT ({oct_row['manufacturers_model_name']}, must be skipped) + 1 non-DICOM file")
    print(f"       into {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
