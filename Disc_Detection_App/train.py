"""Train a disc + fovea box detector on the cached AI-READI photos, with VascX's boxes as labels.

    python train.py --arch fasterrcnn               # torchvision Faster R-CNN (two-stage)
    python train.py --arch retinanet                # torchvision RetinaNet (one-stage)
    python train.py --arch fasterrcnn --limit 64 --epochs 1   # a quick test

Both are torchvision detectors with the same input and output format, so only the model line differs.
COCO-pretrained weights, new heads for 3 classes (background, disc, fovea). Input: the 512 x 512
fundusprep crop from prepare_cache.py.

Which photos: AI-READI's split (splits.csv), keeping photos where VascX found a disc in one piece
and the quality model's q_bad <= --q-bad-max (0.99: above it, label failures jump from <1 % to 3-20 %;
see explore_labels.ipynb). The same filter applies to train, val and test.

Metrics: each photo has exactly one disc and one fovea, so per class we take the top-scoring box
and report IoU with the VascX box, the share with IoU >= 0.5, and the centre error in disc diameters.

Output, per run ($DISC_DATA/../Models/<arch>_<YYYYMMDD-HHMMSS>/, never reused): best.pt (best val), config.json, log.csv
(per epoch), test.csv (best model on the test split, overall and per device).

$DISC_DATA: as in build_cohort.py.
"""

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from tqdm import tqdm

from make_labels import DISC_DATA
from prepare_cache import CACHE, SIZE

CLASSES = ["background", "disc", "fovea"]


def all_photos(q_bad_max: float) -> pd.DataFrame:
    """Every cached photo with its labels, quality, device and split; `kept` = passes the training filter."""
    index = pd.read_csv(CACHE / "index.csv")
    index = index[index.error.isna()].drop(columns="error")
    labels = pd.read_csv(DISC_DATA / "boxes.csv")[["sop_instance_uid", "disc_found", "disc_parts"]]
    quality = pd.read_csv(DISC_DATA / "quality.csv")[["sop_instance_uid", "q_bad"]]
    cohort = pd.read_csv(DISC_DATA / "cohort.csv")[["sop_instance_uid", "participant_id", "device", "region"]]
    photos = (index.merge(labels, on="sop_instance_uid").merge(quality, on="sop_instance_uid")
              .merge(cohort, on="sop_instance_uid").merge(pd.read_csv(DISC_DATA / "splits.csv"), on="participant_id"))
    photos["field"] = photos.device + " / " + photos.region
    photos["kept"] = (photos.disc_found == 1) & (photos.disc_parts == 1) & (photos.q_bad <= q_bad_max)
    return photos


def load_photos(q_bad_max: float, limit: int = None) -> dict:
    """Train / val / test tables of cached photos that pass the filter."""
    photos = all_photos(q_bad_max)
    print(f"{len(photos)} cached photos, {photos.kept.sum()} pass the filter (disc found in one piece, q_bad <= {q_bad_max})")
    photos = photos[photos.kept].drop(columns="kept")
    return {s: photos[photos.split == s].head(limit).reset_index(drop=True) for s in ("train", "val", "test")}


class Crops(torch.utils.data.Dataset):
    """Cached crops with their two boxes; flips and mild colour changes when training."""

    def __init__(self, photos: pd.DataFrame, train: bool):
        self.photos, self.train = photos, train
        self.images = None  # opened in each worker: a memory map isn't shared across processes

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, i):
        if self.images is None:
            self.images = np.load(CACHE / "images.npy", mmap_mode="r")
        p = self.photos.iloc[i]
        image = torch.from_numpy(np.array(self.images[int(p.row)])).permute(2, 0, 1).float().div_(255)
        boxes = torch.tensor([[p.disc_x0, p.disc_y0, p.disc_x1, p.disc_y1],
                              [p.fovea_x0, p.fovea_y0, p.fovea_x1, p.fovea_y1]], dtype=torch.float32).clamp_(0, SIZE)
        if self.train:
            if random.random() < 0.5:  # a mirrored left eye looks like a right eye: free variety
                image = image.flip(-1)
                boxes[:, [0, 2]] = SIZE - boxes[:, [2, 0]]
            image = (image * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08)).clamp_(0, 1)
        return image, {"boxes": boxes, "labels": torch.tensor([1, 2])}


def collate(batch):
    return tuple(zip(*batch))


def build_model(arch: str):
    if arch == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", min_size=SIZE, max_size=SIZE)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, len(CLASSES))
    else:
        model = retinanet_resnet50_fpn_v2(weights="DEFAULT", min_size=SIZE, max_size=SIZE)
        head = model.head.classification_head
        model.head.classification_head = RetinaNetClassificationHead(
            head.conv[0][0].in_channels, head.num_anchors, len(CLASSES))
    return model


def box_iou(a, b) -> float:
    return float(torchvision.ops.box_iou(a[None], b[None])[0, 0])


@torch.no_grad()
def evaluate(model, photos: pd.DataFrame, device, workers: int) -> pd.DataFrame:
    """One row per photo: IoU and centre error (disc diameters) of the top box per class."""
    model.eval()
    loader = torch.utils.data.DataLoader(Crops(photos, train=False), batch_size=16, num_workers=workers, collate_fn=collate)
    rows = []
    for images, targets in loader:
        for output, target in zip(model([im.to(device) for im in images]), targets):
            gt = target["boxes"]
            disc_diameter = float(((gt[0, 2] - gt[0, 0]) + (gt[0, 3] - gt[0, 1])) / 2)
            row = {}
            for k, name in enumerate(CLASSES[1:]):
                mine = (output["labels"] == k + 1).nonzero().flatten()
                if len(mine) == 0:
                    row.update({f"{name}_iou": 0.0, f"{name}_centre_dd": np.nan})
                    continue
                box = output["boxes"][mine[output["scores"][mine].argmax()]].cpu()
                centre_error = float(torch.dist((box[:2] + box[2:]) / 2, (gt[k, :2] + gt[k, 2:]) / 2))
                row.update({f"{name}_iou": box_iou(box, gt[k]), f"{name}_centre_dd": centre_error / disc_diameter})
            rows.append(row)
    return pd.concat([photos[["sop_instance_uid", "field"]], pd.DataFrame(rows)], axis=1)


def summarise(per_photo: pd.DataFrame) -> dict:
    out = {}
    for name in CLASSES[1:]:
        out[f"{name}_iou"] = per_photo[f"{name}_iou"].mean()
        out[f"{name}_hit50"] = (per_photo[f"{name}_iou"] >= 0.5).mean()
        out[f"{name}_centre_dd"] = per_photo[f"{name}_centre_dd"].median()
    return {k: round(float(v), 4) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", choices=["fasterrcnn", "retinanet"], default="fasterrcnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--q-bad-max", type=float, default=0.99)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="at most N photos per split (a quick test)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    photos = load_photos(args.q_bad_max, args.limit)
    print({s: len(p) for s, p in photos.items()})

    run = DISC_DATA.parent / "Models" / f"{args.arch}_{time.strftime('%Y%m%d-%H%M%S')}"
    run.mkdir(parents=True)  # never reuse a folder: an existing one is another run's results
    (run / "config.json").write_text(json.dumps(vars(args) | {"size": SIZE, "classes": CLASSES,
                                                              "photos": {s: len(p) for s, p in photos.items()}}, indent=2))

    model = build_model(args.arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = torch.utils.data.DataLoader(Crops(photos["train"], train=True), batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.workers, collate_fn=collate, drop_last=True)
    log, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, targets in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", unit="batch", leave=False):
            loss = sum(model([im.to(device) for im in images],
                             [{k: v.to(device) for k, v in t.items()} for t in targets]).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        val = summarise(evaluate(model, photos["val"], device, args.workers))
        log.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4)} | val)
        pd.DataFrame(log).to_csv(run / "log.csv", index=False)
        score = (val["disc_iou"] + val["fovea_iou"]) / 2
        if score > best:
            best = score
            torch.save({"arch": args.arch, "classes": CLASSES, "size": SIZE, "epoch": epoch,
                        "model": model.state_dict()}, run / "best.pt")
        print(f"epoch {epoch}: loss {log[-1]['train_loss']}  val {val}{'  <- best' if score == best else ''}")

    model.load_state_dict(torch.load(run / "best.pt", map_location=device)["model"])
    per_photo = evaluate(model, photos["test"], device, args.workers)
    test = pd.DataFrame([{"field": "all", "photos": len(per_photo)} | summarise(per_photo)] + [
        {"field": f, "photos": len(g)} | summarise(g) for f, g in per_photo.groupby("field")])
    test.to_csv(run / "test.csv", index=False)
    print(f"\ntest, best epoch ({run}):\n{test.to_string(index=False)}")


if __name__ == "__main__":
    main()
