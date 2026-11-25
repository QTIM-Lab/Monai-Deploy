#!/usr/bin/env python3
"""Per patient: colour photos and how many have each model's result in the practice Orthanc.

    python3 LOCAL_SCRIPTS/results.py 1019 7115 7034            # table
    python3 LOCAL_SCRIPTS/results.py 1019 7115 7034 --clear    # delete their AI results first (photos stay)
    python3 LOCAL_SCRIPTS/results.py 7115 --delete             # delete the whole patient (to re-upload with auto-run on)

"expected" = colour photos a MAP can process: not Optomed Aurora (the SDK loader drops them: empty
SeriesNumber) and not a mosaic (wider than 1.6:1). A complete run gives expected == each model's count.
--clear deletes only series whose SeriesDescription starts with "AI " (our MAP outputs), so the next run's
results can be watched arriving; Orthanc never overwrites, so without it a re-run changes nothing visible.
"""
import argparse
import json
import urllib.request

ORTHANC = "http://127.0.0.1:18042"
MODELS = {"laterality SR": "AI aireadi-laterality", "vessels SEG": "AI vascx-vessels", "disc/fovea SR": "AI disc-fovea-detection"}


def call(path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(urllib.request.Request(ORTHANC + path, data, method=method)) as r:
        return json.loads(r.read() or b"null")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("patients", nargs="+")
    ap.add_argument("--clear", action="store_true", help="delete these patients' AI result series first")
    ap.add_argument("--delete", action="store_true", help="delete these patients entirely (photos and results)")
    args = ap.parse_args()

    print(f"{'patient':>9} {'colour':>7} {'expected':>9} " + " ".join(f"{m:>14}" for m in MODELS))
    for pid in args.patients:
        if args.delete:
            for orthanc_id in call("/tools/find", {"Level": "Patient", "Query": {"PatientID": pid}}):
                call(f"/patients/{orthanc_id}", method="DELETE")
        series = call("/tools/find", {"Level": "Series", "Query": {"PatientID": pid}, "Expand": True})
        if args.clear:
            for s in series:
                if s["MainDicomTags"].get("SeriesDescription", "").startswith("AI "):
                    call(f"/series/{s['ID']}", method="DELETE")
            series = call("/tools/find", {"Level": "Series", "Query": {"PatientID": pid}, "Expand": True})
        colour = expected = 0
        counts = dict.fromkeys(MODELS, 0)
        for s in series:
            desc = s["MainDicomTags"].get("SeriesDescription", "")
            for model, prefix in MODELS.items():
                if desc.startswith(prefix):
                    counts[model] += len(s["Instances"])
            if s["MainDicomTags"].get("Modality") == "OP":
                for i in s["Instances"]:
                    t = call(f"/instances/{i}/simplified-tags")
                    if "COLOR" in t.get("ImageType", "").upper():
                        colour += 1
                        aurora = not t.get("SeriesNumber")
                        mosaic = int(t.get("Columns", 1)) / int(t.get("Rows", 1)) > 1.6
                        expected += not (aurora or mosaic)
        print(f"{pid:>9} {colour:>7} {expected:>9} " + " ".join(f"{counts[m]:>14}" for m in MODELS))


if __name__ == "__main__":
    main()
