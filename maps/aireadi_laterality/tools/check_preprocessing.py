"""Prove model_spec decodes and resizes exactly as training did. Run after any change
to model_spec.py's preprocessing, or to pydicom/Pillow versions.

    python maps/aireadi_laterality/tools/check_preprocessing.py [--n 20]

Compares model_spec.resize(model_spec.to_rgb(ds)) against the arrays prepare_data.py
cached for training, byte for byte. Needs AIREADI_Laterality_App/cache/ (run
prepare_data.py first) and the AI-READI dataset. No Holoscan, no GPU.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pydicom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import model_spec  # noqa: E402

REPO = HERE.parents[2]
CACHE = REPO / "AIREADI_Laterality_App" / "cache"
DATA_ROOT = os.environ.get("AIREADI_ROOT", "/persist/public_datasets/ai_readi")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=20, help="test-split images to compare")
    args = parser.parse_args()

    rows = [r for r in csv.DictReader(open(CACHE / "split.csv")) if r["split"] == "test"]
    cached = np.load(CACHE / "test_x.npy", mmap_mode="r")
    assert len(rows) == len(cached), "split.csv and test_x.npy disagree; rerun prepare_data.py"

    mismatches = 0
    for i, row in enumerate(rows[: args.n]):
        ds = pydicom.dcmread(DATA_ROOT + row["filepath"])
        ours = model_spec.resize(model_spec.to_rgb(ds))
        if not np.array_equal(ours, cached[i]):
            mismatches += 1
            diff = np.abs(ours.astype(int) - cached[i].astype(int))
            print(f"MISMATCH {Path(row['filepath']).name}: max |diff| {diff.max()}, mean {diff.mean():.2f}")

    n = min(args.n, len(rows))
    print(f"{n - mismatches}/{n} identical to the training cache")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
