"""Decode and crop every cohort photo once, so training reads ready arrays instead of DICOMs.

    python prepare_cache.py --limit 50     # a quick test: the first 50 photos
    python prepare_cache.py                # the whole cohort (resumes; done photos are skipped)

Writes $DISC_DATA/cache_512/:
    images.npy   (N, 512, 512, 3) uint8, row i = row i of cohort.csv: the fundusprep crop, as VascX sees it
    index.csv    one row per cached photo: row, sop_instance_uid, the crop matrix (to map results back
                 to the original photo), and the disc and fovea boxes in crop pixels

Decoding is ~0.2 s per photo; doing it every epoch would dominate training. The crop is scale plus
translation only, so boxes stay axis-aligned. Every photo is cached (also ones without a disc);
train.py chooses which to use. Delete the folder to rebuild, e.g. after changing the cohort or labels.

$AIREADI_ROOT, $DISC_DATA: as in build_cohort.py.
"""

import argparse
import csv

import numpy as np
import pydicom
import torch
from rtnls_fundusprep.mask_extraction import get_cfi_bounds
from tqdm import tqdm

from make_labels import AIREADI_ROOT, DISC_DATA, as_is

SIZE = 512
CACHE = DISC_DATA / f"cache_{SIZE}"
BOXES = ["disc", "fovea"]
COLUMNS = ["row", "sop_instance_uid"] + [f"m{i}" for i in range(6)] + [
    f"{b}_{c}" for b in BOXES for c in ("x0", "y0", "x1", "y1")] + ["error"]


class Photos(torch.utils.data.Dataset):
    """Decodes and crops in DataLoader workers, one photo per item."""

    def __init__(self, rows):
        self.rows = rows  # (cohort row number, cohort row, boxes.csv row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        n, photo, boxes = self.rows[i]
        out = {"row": n, "sop_instance_uid": photo["sop_instance_uid"]}
        try:
            rgb = pydicom.dcmread(AIREADI_ROOT / photo["filepath"].lstrip("/")).pixel_array  # pydicom 3: YBR -> RGB
            transform, _ = get_cfi_bounds(rgb).crop(SIZE)
            out["image"] = transform.warp(rgb, (SIZE, SIZE))
            out.update({f"m{i}": round(float(v), 6) for i, v in enumerate(transform.M[:2].flatten())})
            for b in BOXES:
                if boxes.get(f"{b}_x0"):
                    corners = np.array([[float(boxes[f"{b}_x0"]), float(boxes[f"{b}_y0"])],
                                        [float(boxes[f"{b}_x1"]), float(boxes[f"{b}_y1"])]])
                    (x0, y0), (x1, y1) = transform.apply(corners)
                    out.update({f"{b}_x0": round(x0, 2), f"{b}_y0": round(y0, 2),
                                f"{b}_x1": round(x1, 2), f"{b}_y1": round(y1, 2)})
        except Exception as e:  # unreadable file or no fundus circle found: record it, keep going
            out["error"] = f"{type(e).__name__}: {e}"
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="only the first N cohort photos (a test)")
    parser.add_argument("--workers", type=int, default=16, help="DataLoader workers decoding DICOMs")
    args = parser.parse_args()

    with open(DISC_DATA / "cohort.csv", newline="") as f:
        cohort = list(csv.DictReader(f))
    with open(DISC_DATA / "boxes.csv", newline="") as f:
        boxes = {r["sop_instance_uid"]: r for r in csv.DictReader(f)}

    CACHE.mkdir(parents=True, exist_ok=True)
    images_file, index_file = CACHE / "images.npy", CACHE / "index.csv"
    if images_file.exists():
        images = np.load(images_file, mmap_mode="r+")
        if len(images) != len(cohort):
            raise SystemExit(f"{images_file} has {len(images)} rows but cohort.csv has {len(cohort)}: delete {CACHE} and rebuild")
    else:
        images = np.lib.format.open_memmap(images_file, mode="w+", dtype=np.uint8, shape=(len(cohort), SIZE, SIZE, 3))

    done = set()
    if index_file.exists():
        with open(index_file, newline="") as f:
            done = {int(r["row"]) for r in csv.DictReader(f)}
    selected = range(min(args.limit, len(cohort))) if args.limit else range(len(cohort))
    todo = [(n, cohort[n], boxes.get(cohort[n]["sop_instance_uid"], {})) for n in selected if n not in done]
    print(f"{len(selected)} photos selected, {len(selected) - len(todo)} already cached, {len(todo)} to do -> {CACHE}")
    if not todo:
        return

    loader = torch.utils.data.DataLoader(Photos(todo), batch_size=None, num_workers=args.workers, collate_fn=as_is)
    new_file = not index_file.exists()
    with open(index_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for i, item in enumerate(tqdm(loader, total=len(todo), unit="img")):
            if "image" in item:
                images[item["row"]] = item["image"]
            writer.writerow(item)
            if i % 200 == 0:
                images.flush()  # images first, then the index row that says they're there
                f.flush()
        images.flush()


if __name__ == "__main__":
    main()
