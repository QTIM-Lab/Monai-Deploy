"""Run VascX disc and fovea on the cohort and write the detector's labels to the data area.

    python make_labels.py --per-device 20      # a quick sample: 20 images of each device / region
    python make_labels.py                      # the whole cohort (resumes; done images are skipped)

Per image (sop_instance_uid), in $DISC_DATA:
    disc_masks/<uid>.png   VascX disc mask at the photo's own size, 0 / 255
    boxes.csv              one row: disc box (largest part of the mask), fovea point, fovea box

Boxes are pixel coordinates in the original photo, x0 y0 inclusive, x1 y1 exclusive.
The fovea box is a square centred on the fovea point with the disc's diameter (mean of its box
width and height) as its side, so it scales with the eye; it's empty when no disc is found.

Same preprocessing as the configs embedded in the model files (and as upstream's
rtnls_inference): the fundusprep crop, ImageNet normalisation, softmax / heatmap argmax.
Deterministic GPU settings, so a re-run gives the same labels on the same GPU type.

$AIREADI_ROOT, $DISC_DATA: as in build_cohort.py.
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; set before CUDA starts

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pydicom  # noqa: E402
import torch  # noqa: E402
from rtnls_fundusprep.mask_extraction import get_cfi_bounds  # noqa: E402
from scipy import ndimage  # noqa: E402
from tqdm import tqdm  # noqa: E402

AIREADI_ROOT = Path(os.environ.get("AIREADI_ROOT", "/persist/public_datasets/ai_readi"))
DISC_DATA = Path(os.environ.get("DISC_DATA", "/scratch90/QTIM/Active/Non-IRB/VascX_disc_detection_and_segmentation/Data"))
HERE = Path(__file__).resolve().parent

DISC_MODEL = HERE / "models" / "disc" / "disc_may26.pt"
FOVEA_MODEL = HERE / "models" / "fovea" / "fovea_may26.pt"

# From the configs embedded in the model files.
DISC_SQUARE = 512  # disc: square_size 512, no TTA
FOVEA_SQUARE, FOVEA_RESIZE = 1024, 512  # fovea: square_size 1024 then resize 512, horizontal-flip TTA
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # normalize: imagenet
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COLUMNS = [
    "sop_instance_uid", "height", "width",
    "disc_found", "disc_x0", "disc_y0", "disc_x1", "disc_y1", "disc_pixels", "disc_parts",
    "fovea_x", "fovea_y", "fovea_peak", "fovea_x0", "fovea_y0", "fovea_x1", "fovea_y1",
    "error",
]


class Photos(torch.utils.data.Dataset):
    """Decodes and crops in DataLoader workers (decoding is the slow part), one photo per item."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        try:
            rgb = pydicom.dcmread(AIREADI_ROOT / row["filepath"].lstrip("/")).pixel_array  # pydicom 3: YBR -> RGB
            bounds = get_cfi_bounds(rgb)
            disc_t, _ = bounds.crop(DISC_SQUARE)
            fovea_t, _ = bounds.crop(FOVEA_SQUARE)
            fovea_in = cv2.resize(fovea_t.warp(rgb, (FOVEA_SQUARE, FOVEA_SQUARE)), (FOVEA_RESIZE, FOVEA_RESIZE),
                                  interpolation=cv2.INTER_LINEAR)
            return {"row": row, "shape": rgb.shape[:2], "disc_t": disc_t, "fovea_t": fovea_t,
                    "disc_in": disc_t.warp(rgb, (DISC_SQUARE, DISC_SQUARE)), "fovea_in": fovea_in}
        except Exception as e:  # unreadable file or no fundus circle found: record it, keep going
            return {"row": row, "error": f"{type(e).__name__}: {e}"}


def as_is(item):
    """DataLoader collate that leaves items alone (the default turns numpy arrays into tensors)."""
    return item


def to_tensor(rgb: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy((rgb.astype(np.float32) / 255.0 - MEAN) / STD).permute(2, 0, 1)[None].to(device)


def disc_label(model, item, device) -> dict:
    with torch.no_grad(), torch.jit.optimized_execution(False):
        logits = model(to_tensor(item["disc_in"], device))[0].float()  # (M=1, N, C, H, W) -> (N, C, H, W)
    probability = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
    mask = (item["disc_t"].warp_inverse(probability, item["shape"]) > 0.5).astype(np.uint8)
    cv2.imwrite(str(DISC_DATA / "disc_masks" / f"{item['row']['sop_instance_uid']}.png"), mask * 255)

    parts, n_parts = ndimage.label(mask)
    if n_parts == 0:
        return {"disc_found": 0, "disc_parts": 0}
    largest = 1 + int(np.argmax(ndimage.sum(mask, parts, range(1, n_parts + 1))))
    ys, xs = np.nonzero(parts == largest)
    return {"disc_found": 1, "disc_x0": int(xs.min()), "disc_y0": int(ys.min()),
            "disc_x1": int(xs.max()) + 1, "disc_y1": int(ys.max()) + 1,
            "disc_pixels": int(mask.sum()), "disc_parts": n_parts}


def fovea_label(model, item, device) -> dict:
    x = to_tensor(item["fovea_in"], device)
    with torch.no_grad(), torch.jit.optimized_execution(False):
        heatmap = model(x)[0].float() + torch.flip(model(torch.flip(x, [3]))[0].float(), [3])
    heatmap = (heatmap / 2)[0, 0]  # (N, K, H, W) -> (H, W)
    peak, index = heatmap.flatten().max(0)
    row, col = divmod(int(index), heatmap.shape[1])
    scale = FOVEA_SQUARE / FOVEA_RESIZE  # back to the 1024 crop, then to the photo
    (fx, fy), = item["fovea_t"].apply_inverse(np.array([[(col + 0.5) * scale, (row + 0.5) * scale]]))
    return {"fovea_x": round(float(fx), 1), "fovea_y": round(float(fy), 1), "fovea_peak": round(float(peak), 4)}


def fovea_box(label: dict, height: int, width: int) -> dict:
    if not label.get("disc_found"):
        return {}
    side = ((label["disc_x1"] - label["disc_x0"]) + (label["disc_y1"] - label["disc_y0"])) / 2
    x, y = label["fovea_x"], label["fovea_y"]
    return {"fovea_x0": max(0, round(x - side / 2)), "fovea_y0": max(0, round(y - side / 2)),
            "fovea_x1": min(width, round(x + side / 2)), "fovea_y1": min(height, round(y + side / 2))}


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

    boxes_csv = DISC_DATA / "boxes.csv"
    done = set()
    if boxes_csv.exists():
        with open(boxes_csv, newline="") as f:
            done = {r["sop_instance_uid"] for r in csv.DictReader(f)}
    todo = [r for r in cohort if r["sop_instance_uid"] not in done]
    print(f"{len(cohort)} images selected, {len(cohort) - len(todo)} already labelled, {len(todo)} to do -> {DISC_DATA}")
    if not todo:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    disc_model = torch.jit.load(str(DISC_MODEL), map_location=device).eval()
    fovea_model = torch.jit.load(str(FOVEA_MODEL), map_location=device).eval()

    (DISC_DATA / "disc_masks").mkdir(parents=True, exist_ok=True)
    loader = torch.utils.data.DataLoader(Photos(todo), batch_size=None, num_workers=args.workers, collate_fn=as_is)
    new_file = not boxes_csv.exists()
    with open(boxes_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for item in tqdm(loader, total=len(todo), unit="img"):
            label = {"sop_instance_uid": item["row"]["sop_instance_uid"]}
            if "error" in item:
                label["error"] = item["error"]
            else:
                height, width = item["shape"]
                label.update(height=height, width=width)
                label.update(disc_label(disc_model, item, device))
                label.update(fovea_label(fovea_model, item, device))
                label.update(fovea_box(label, height, width))
            writer.writerow(label)
            f.flush()  # every row on disk at once: an interrupted run resumes where it stopped


if __name__ == "__main__":
    main()
