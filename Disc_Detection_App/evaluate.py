"""Run trained detectors on every cached photo of a split and save each photo's top boxes, for the notebook.

    python evaluate.py                                   # every run in Models/, test split
    python evaluate.py --runs fasterrcnn_20261105-151050 --split val

train.py's test.csv only holds summaries over the photos that pass the filter. This keeps one row per
photo, for *all* photos of the split (the filtered-out ones too, flagged by `kept`), so the notebook can
look at error distributions, worst cases, and what the detector says on photos VascX couldn't label.

Writes $DISC_DATA/../Stats/<run>/<split>_predictions.csv, one row per photo:
    sop_instance_uid, row (in cache_512/images.npy), field, kept, q_bad, disc_found, disc_parts,
    gt_<disc|fovea>_x0..y1    VascX box in crop pixels (empty when VascX found no disc)
    <disc|fovea>_x0..y1       top-scoring predicted box in crop pixels (empty when nothing scored >= 0.05)
    <disc|fovea>_score        its score
    <disc|fovea>_n            boxes of that class with score >= 0.5 (more than 1 = the model saw two candidates)
and <split>_speed.json (photos per second at batch 16, on the GPU used). Existing files are kept unless --force.

$DISC_DATA: as in build_cohort.py.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from prepare_cache import CACHE
from train import CLASSES, all_photos, build_model
from make_labels import DISC_DATA

MODELS = DISC_DATA.parent / "Models"
STATS = DISC_DATA.parent / "Stats"


class Images(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows, self.images = rows, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.images is None:
            self.images = np.load(CACHE / "images.npy", mmap_mode="r")
        return torch.from_numpy(np.array(self.images[int(self.rows[i])])).permute(2, 0, 1).float().div_(255)


@torch.no_grad()
def predict(run, photos: pd.DataFrame, device, workers: int):
    checkpoint = torch.load(run / "best.pt", map_location="cpu")
    model = build_model(checkpoint["arch"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    loader = torch.utils.data.DataLoader(Images(photos.row.tolist()), batch_size=16, num_workers=workers)
    rows, seconds = [], 0.0
    for images in tqdm(loader, desc=run.name, unit="batch", leave=False):
        images = images.to(device)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        start = time.perf_counter()
        outputs = model(list(images))
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        seconds += time.perf_counter() - start
        for output in outputs:
            row = {}
            for k, name in enumerate(CLASSES[1:], start=1):
                mine = (output["labels"] == k).nonzero().flatten()
                row[f"{name}_n"] = int((output["scores"][mine] >= 0.5).sum())
                if len(mine) == 0:
                    continue
                best = mine[output["scores"][mine].argmax()]
                row |= dict(zip([f"{name}_{c}" for c in ("x0", "y0", "x1", "y1")], output["boxes"][best].tolist()))
                row[f"{name}_score"] = float(output["scores"][best])
            rows.append(row)
    return pd.DataFrame(rows), {"photos": len(photos), "seconds": round(seconds, 2),
                                "photos_per_s": round(len(photos) / seconds, 1), "gpu": torch.cuda.get_device_name(device),
                                "epoch": checkpoint["epoch"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="*", help="run folder names in Models/ (default: all)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="overwrite existing predictions")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    runs = [MODELS / r for r in args.runs] if args.runs else sorted(p for p in MODELS.iterdir() if (p / "best.pt").exists())
    for run in runs:
        q_bad_max = json.loads((run / "config.json").read_text())["q_bad_max"]
        photos = all_photos(q_bad_max)
        photos = photos[photos.split == args.split].reset_index(drop=True)
        out = STATS / run.name / f"{args.split}_predictions.csv"
        if out.exists() and not args.force:
            print(f"{out} exists, skipping (--force to redo)")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        predictions, speed = predict(run, photos, device, args.workers)
        gt = photos[[f"{b}_{c}" for b in ("disc", "fovea") for c in ("x0", "y0", "x1", "y1")]]
        gt.columns = [f"gt_{c}" for c in gt.columns]
        gt = gt.where(photos.disc_found.eq(1), axis=0)  # no VascX disc -> no ground truth
        keep = ["sop_instance_uid", "row", "field", "kept", "q_bad", "disc_found", "disc_parts"]
        pd.concat([photos[keep], gt, predictions], axis=1).to_csv(out, index=False, float_format="%.4g")
        (out.parent / f"{args.split}_speed.json").write_text(json.dumps(speed, indent=2))
        print(f"{run.name}: {len(photos)} photos ({photos.kept.sum()} kept), {speed['photos_per_s']} photos/s -> {out}")


if __name__ == "__main__":
    main()
