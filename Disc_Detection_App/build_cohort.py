"""Pick the AI-READI colour fundus photos for disc detection and write the cohort and splits.

    python build_cohort.py

Writes, to the data area and to dataset/ here (same files):
    cohort.csv   one row per image: sop_instance_uid, participant, device, region, laterality, size, filepath
    splits.csv   one row per participant: AI-READI's own recommended_split (train / val / test)

The split is AI-READI's official one, per participant, so both eyes of a person stay together and
results are comparable with other AI-READI work. Re-running gives the same files.

$AIREADI_ROOT  the AI-READI v2.0.0 release (default /persist/public_datasets/ai_readi)
$DISC_DATA     the data area's Data/ folder (default the scratch working copy)
"""

import csv
import os
from collections import Counter
from pathlib import Path

AIREADI_ROOT = Path(os.environ.get("AIREADI_ROOT", "/persist/public_datasets/ai_readi"))
DISC_DATA = Path(os.environ.get("DISC_DATA", "/scratch90/QTIM/Active/Non-IRB/VascX_disc_detection_and_segmentation/Data"))
HERE = Path(__file__).resolve().parent

# Standard-field colour photos, the kind VascX was trained on ("not trained on WF or UWF").
# Left out: Eidon Mosaic, Nasal, Temporal Periphery and Maestro2 Wide Field (wide or off-centre fields).
FIELDS = {
    ("Aurora", "Macula or Optic Disc"),
    ("Eidon", "Macula"),
    ("Maestro2", "Macula"),
    ("Triton", "Macula, 6 x 6"),
    ("Triton", "Macula, 12 x 12"),
    ("Triton", "Optic Disc"),
}

COHORT_COLUMNS = ["sop_instance_uid", "participant_id", "device", "region", "laterality", "height", "width", "filepath"]


def read_tsv(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_csv(rows, columns, name: str) -> None:
    for folder in (DISC_DATA, HERE / "dataset"):
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    manifest = read_tsv(AIREADI_ROOT / "retinal_photography" / "manifest.tsv")
    split_of = {p["participant_id"]: p["recommended_split"] for p in read_tsv(AIREADI_ROOT / "participants.tsv")}

    cohort = [
        {
            "sop_instance_uid": r["sop_instance_uid"],
            "participant_id": r["participant_id"],
            "device": r["manufacturers_model_name"],
            "region": r["anatomic_region"],
            "laterality": r["laterality"],
            "height": r["height"],
            "width": r["width"],
            "filepath": r["filepath"],  # relative to AIREADI_ROOT, as in the manifest
        }
        for r in manifest
        if r["imaging"] == "Color Photography" and (r["manufacturers_model_name"], r["anatomic_region"]) in FIELDS
    ]
    cohort.sort(key=lambda r: (r["participant_id"], r["device"], r["region"], r["laterality"], r["sop_instance_uid"]))
    participants = sorted({r["participant_id"] for r in cohort})
    splits = [{"participant_id": p, "split": split_of[p]} for p in participants]

    write_csv(cohort, COHORT_COLUMNS, "cohort.csv")
    write_csv(splits, ["participant_id", "split"], "splits.csv")

    split_by_participant = {s["participant_id"]: s["split"] for s in splits}
    images_per_split = Counter(split_by_participant[r["participant_id"]] for r in cohort)
    print(f"{len(cohort)} images, {len(participants)} participants -> {DISC_DATA} and {HERE / 'dataset'}")
    for split in ("train", "val", "test"):
        n_participants = sum(s["split"] == split for s in splits)
        print(f"  {split:5s}  {n_participants:4d} participants  {images_per_split[split]:6d} images")
    for (device, region), n in sorted(Counter((r["device"], r["region"]) for r in cohort).items()):
        print(f"  {n:6d}  {device} / {region}")


if __name__ == "__main__":
    main()
