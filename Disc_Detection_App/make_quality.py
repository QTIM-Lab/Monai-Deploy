"""Run VascX's image-quality model on the cohort, to drop ungradable photos before training.

    python make_quality.py --per-device 20     # a quick sample
    python make_quality.py                     # the whole cohort (resumes; done images are skipped)

Writes $DISC_DATA/quality.csv, one row per image (sop_instance_uid):
    q_good, q_usable, q_bad   class probabilities (softmax of the 5 members' mean logits)
    quality                   the most likely class: good / usable / bad

The model is VascX's EyePACS-trained classifier (trained on EyeQ, classes Good / Usable / Reject).
VascX's docs recommend dropping `bad` before analysis, or `bad` and `usable` for stricter work.
Preprocessing as embedded in the model file: fundusprep crop to 1024, resize to 224, ImageNet
normalisation, horizontal-flip TTA.

$AIREADI_ROOT, $DISC_DATA: as in build_cohort.py.
"""

import argparse
import csv
from collections import defaultdict

import cv2
import numpy as np
import pydicom
import torch
from rtnls_fundusprep.mask_extraction import get_cfi_bounds
from tqdm import tqdm

from make_labels import AIREADI_ROOT, DISC_DATA, HERE, as_is, to_tensor  # also sets deterministic cuBLAS

QUALITY_MODEL = HERE / "models" / "quality" / "quality.pt"
SQUARE, RESIZE = 1024, 224  # test_transform: square_size 1024, resize 224
CLASSES = ["good", "usable", "bad"]  # EyeQ order: Good, Usable, Reject
COLUMNS = ["sop_instance_uid", "q_good", "q_usable", "q_bad", "quality", "error"]


class Photos(torch.utils.data.Dataset):
    """Decodes and crops in DataLoader workers, one photo per item."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        try:
            rgb = pydicom.dcmread(AIREADI_ROOT / row["filepath"].lstrip("/")).pixel_array  # pydicom 3: YBR -> RGB
            transform, _ = get_cfi_bounds(rgb).crop(SQUARE)
            square = transform.warp(rgb, (SQUARE, SQUARE))
            return {"row": row, "image": cv2.resize(square, (RESIZE, RESIZE), interpolation=cv2.INTER_LINEAR)}
        except Exception as e:  # unreadable file or no fundus circle found: record it, keep going
            return {"row": row, "error": f"{type(e).__name__}: {e}"}


def quality(model, image, device) -> dict:
    x = to_tensor(image, device)
    with torch.no_grad(), torch.jit.optimized_execution(False):
        logits = model(x).float() + model(torch.flip(x, [3])).float()  # (M=5, N, 3), plus the flipped image
    probabilities = torch.softmax((logits / 2).mean(0)[0], dim=0).cpu().numpy()
    return {"q_good": round(float(probabilities[0]), 4), "q_usable": round(float(probabilities[1]), 4),
            "q_bad": round(float(probabilities[2]), 4), "quality": CLASSES[int(np.argmax(probabilities))]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-device", type=int, help="only the first N images of each device / region (a sample)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers decoding DICOMs")
    args = parser.parse_args()

    with open(DISC_DATA / "cohort.csv", newline="") as f:
        cohort = list(csv.DictReader(f))
    if args.per_device:
        by_field = defaultdict(list)
        for r in cohort:
            by_field[(r["device"], r["region"])].append(r)
        cohort = [r for rows in by_field.values() for r in rows[: args.per_device]]

    quality_csv = DISC_DATA / "quality.csv"
    done = set()
    if quality_csv.exists():
        with open(quality_csv, newline="") as f:
            done = {r["sop_instance_uid"] for r in csv.DictReader(f)}
    todo = [r for r in cohort if r["sop_instance_uid"] not in done]
    print(f"{len(cohort)} images selected, {len(cohort) - len(todo)} already rated, {len(todo)} to do -> {quality_csv}")
    if not todo:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model = torch.jit.load(str(QUALITY_MODEL), map_location=device).eval()

    loader = torch.utils.data.DataLoader(Photos(todo), batch_size=None, num_workers=args.workers, collate_fn=as_is)
    new_file = not quality_csv.exists()
    with open(quality_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for item in tqdm(loader, total=len(todo), unit="img"):
            row = {"sop_instance_uid": item["row"]["sop_instance_uid"]}
            row.update({"error": item["error"]} if "error" in item else quality(model, item["image"], device))
            writer.writerow(row)
            f.flush()  # every row on disk at once: an interrupted run resumes where it stopped


if __name__ == "__main__":
    main()
