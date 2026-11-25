"""Export a trained run's best.pt to TorchScript, the form a MAP loads (the App SDK only loads TorchScript).

    python export_model.py                                  # the chosen run, RetinaNet
    python export_model.py --run fasterrcnn_20261105-151050 # another run

Writes export/retinanet/<run>.pt (gitignored, like the VascX weights) and prints its SHA-256, which
goes into maps/disc_detection/model_spec.py as MODEL_SHA256. Then checks the exported model gives the
same boxes as the original on a few cached test photos. export/ is what `monai-deploy package -m` takes:
the App SDK accepts only folders in it, each holding exactly one file, so nothing else goes there.

The exported model still needs torchvision at run time (its NMS and box ops are torchvision kernels).
$DISC_DATA: as in build_cohort.py.
"""

import argparse
import hashlib
import shutil

import numpy as np
import torch

from detector import MODEL_DIR, boxes_in_crop
from make_labels import DISC_DATA
from prepare_cache import CACHE
from train import build_model

CHOSEN_RUN = "retinanet_20261105-151050"  # RetinaNet: same accuracy as Faster R-CNN, 2.2x faster (explore_labels.ipynb Part 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=CHOSEN_RUN)
    args = parser.parse_args()

    run = DISC_DATA.parent / "Models" / args.run
    checkpoint = torch.load(run / "best.pt", map_location="cpu")
    model = build_model(checkpoint["arch"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scripted = torch.jit.script(model)

    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)  # one model per folder: the MAP refuses a folder with two
    MODEL_DIR.mkdir(parents=True)
    out = MODEL_DIR / f"{args.run}.pt"
    scripted.save(str(out))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{out}\nSHA-256 {sha}")

    # Same boxes from the exported model as from the original, on cached photos.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = np.load(CACHE / "images.npy", mmap_mode="r")
    loaded = torch.jit.load(str(out), map_location=device).eval()
    model.to(device)
    worst = 0.0
    for row in range(0, 2000, 250):
        a, b = boxes_in_crop(model, np.array(images[row]), device), boxes_in_crop(loaded, np.array(images[row]), device)
        assert a.keys() == b.keys(), (row, a, b)
        worst = max([worst] + [abs(x - y) for k in a for x, y in zip(a[k], b[k])])
    print(f"exported vs original on 8 cached photos: largest difference {worst:.2e}")


if __name__ == "__main__":
    main()
