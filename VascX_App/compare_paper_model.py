"""The paper's VascX model vs the current one, on the same photos from our PACS.

    pyenv activate vascx
    python VascX_App/compare_paper_model.py --patient 1001          # or --study <StudyInstanceUID>

  PAPER model   vessels_july24.pt: the 5-model ensemble evaluated in Vargas Quiros et al.,
                TVST 2025, doi:10.1167/tvst.14.7.19. Run with the AUTHORS' code, exactly as
                their published notebooks do (huggingface.co/Eyened/vascx, notebooks/):
                  0_preprocess.ipynb           rtnls_fundusprep parallel_preprocess -> RGB + contrast-enhanced
                  1_segment_preprocessed.ipynb SegmentationEnsemble ... predict_preprocessed
  NEW model     vessels_may26.pt: what the MAP deploys. Run with OUR code, vessels.py.
  CHECK         the new model run with the authors' code too. It should equal ours (Dice ~0.999;
                they use half precision, we use full), which shows vessels.py runs it faithfully.

The question this answers: on our images, does the new model behave like the published one?
Two different models, so "roughly match" (Dice ~0.8-0.95) is the expectation, not 1.0.
All three are compared on the same 1024x1024 fundus crop (the authors' preprocessed RGB).
Writes results/paper_vs_new_<date>.csv. Weights: bash VascX_App/fetch_model.sh.
"""

import argparse
import datetime
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from pydicom.errors import InvalidDicomError
from rtnls_fundusprep.preprocessor import parallel_preprocess
from rtnls_inference import SegmentationEnsemble

import vessels
from orthanc import fetch

HERE = Path(__file__).resolve().parent
PAPER_MODEL = HERE / "reference_models" / "vessels_july24.pt"


def colour_photos(folder: Path):
    """(dataset, RGB) for every single-field colour fundus photo in a folder of DICOMs.

    Same rule as the MAP's model_spec: Modality OP, ImageType COLOR, 3 samples, one frame,
    no wider than 1.6:1 (wider are multi-field mosaics the crop can't handle).
    """
    photos = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        try:
            ds = pydicom.dcmread(path)
        except (InvalidDicomError, OSError):
            continue
        if (
            ds.get("Modality") == "OP"
            and "COLOR" in [str(v).upper() for v in ds.get("ImageType", [])]
            and int(ds.get("SamplesPerPixel", 1)) == 3
            and int(ds.get("NumberOfFrames", 1) or 1) == 1
            and ds.Columns / ds.Rows <= 1.6
        ):
            photos.append((ds, ds.pixel_array))
    return photos


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def compare(photos, device: torch.device, workdir: Path) -> pd.DataFrame:
    """Run all three on the photos; one row per photo. Also leaves the masks in workdir/<model>/."""
    original = workdir / "original"
    original.mkdir()
    for i, (_, rgb) in enumerate(photos):
        Image.fromarray(rgb).save(original / f"{i}.png")
    files = sorted(original.glob("*.png"), key=lambda p: int(p.stem))

    # Authors' notebook 0: crop to the fundus and make the contrast-enhanced copy the paper model needs.
    parallel_preprocess(files, rgb_path=workdir / "rgb", ce_path=workdir / "ce", n_jobs=4)
    data = [{"id": f.stem, "image": str(workdir / "rgb" / f.name), "contrast_enhanced": str(workdir / "ce" / f.name)}
            for f in files]

    # Authors' notebook 1, with the paper's model; then the same call with the new model.
    paper = SegmentationEnsemble.from_torchscript(str(PAPER_MODEL)).to(device).eval()
    paper.predict_preprocessed(data, dest_path=workdir / "paper", num_workers=0)
    new_by_authors = SegmentationEnsemble.from_torchscript(str(vessels.MODEL_FILE)).to(device).eval()
    new_by_authors.predict_preprocessed([{"id": d["id"], "image": d["image"]} for d in data],
                                        dest_path=workdir / "new_authors_code", num_workers=0)

    # Ours: vessels.py on the same crop.
    model = vessels.load_model(device=device)
    (workdir / "new").mkdir()
    rows = []
    for i, (ds, _) in enumerate(photos):
        crop = np.asarray(Image.open(workdir / "rgb" / f"{i}.png"))
        new = vessels.vessel_probability(model, crop, device) > 0.5
        Image.fromarray(new.astype(np.uint8)).save(workdir / "new" / f"{i}.png")
        old = np.asarray(Image.open(workdir / "paper" / f"{i}.png")) > 0
        theirs = np.asarray(Image.open(workdir / "new_authors_code" / f"{i}.png")) > 0
        rows.append(dict(
            image=i,
            camera=str(ds.get("ManufacturerModelName", "?")),
            sop_instance_uid=ds.SOPInstanceUID,
            dice_paper_vs_new=round(dice(old, new), 4),
            vessel_pct_paper=round(100 * old.mean(), 2),
            vessel_pct_new=round(100 * new.mean(), 2),
            dice_new_authors_vs_ours=round(dice(theirs, new), 4),
        ))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--patient", help="PatientID in Orthanc")
    which.add_argument("--study", help="StudyInstanceUID in Orthanc")
    args = parser.parse_args()

    device = vessels.default_device()
    with fetch(patient=args.patient, study=args.study) as folder:
        photos = colour_photos(folder)
    if not photos:
        print("no single-field colour fundus photos found", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as workdir:
        table = compare(photos, device, Path(workdir))

    pd.set_option("display.width", 200)
    print(table.drop(columns="sop_instance_uid").to_string(index=False))
    print("\nby camera (median):")
    print(table.groupby("camera")[["dice_paper_vs_new", "vessel_pct_paper", "vessel_pct_new",
                                   "dice_new_authors_vs_ours"]].median().round(4).to_string())

    out = HERE / "results" / f"paper_vs_new_{datetime.date.today():%Y-%m-%d}.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n{len(table)} photo(s), on {device}; written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
