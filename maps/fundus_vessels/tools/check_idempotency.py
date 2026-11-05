"""Idempotency check: re-running the MAP reproduces the same SEGs, never duplicates.

Runs inside the MAP image itself (nothing to install), on a patient or study from Orthanc:

    docker run --rm --gpus device=0 --network host --entrypoint python3 \
      fundus_vessels-x64-workstation-dgpu-linux-amd64:0.2.0 \
      /opt/holoscan/app/tools/check_idempotency.py --patient 1001

Run it after changing model_spec.py (inference) or map_ops/seg_builder.py (SEG output).
Uses model_spec + map_ops.seg_builder exactly as the MAP does. Checks:

  1. same input twice           -> same SOP UIDs, byte-identical files
  2. alone vs in a batch        -> each image's SEG doesn't depend on what else was run
  3. version bump               -> all-new UIDs (old results are kept, not overwritten)
  4. re-run after a partial run -> only the missing SEGs are new; the rest are identical
  5. against Orthanc            -> SEGs already stored under these UIDs have the same content as
                                   now. Catches drift checks 1-4 can't see: code changed without a
                                   version bump, another host/GPU/driver, a rebuilt image. Orthanc
                                   keeps the first copy (OverwriteInstances false), so drift would
                                   otherwise go unnoticed. SKIP if nothing is stored yet.

Orthanc URL: $ORTHANC_URL, default the practice instance (hence --network host).
"""

import argparse
import io
import json
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pydicom
import torch
from pydicom.errors import InvalidDicomError
from pydicom.filebase import DicomBytesIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_spec  # noqa: E402
from map_ops.instance_segmenter_operator import checked_model_file  # noqa: E402
from map_ops.seg_builder import build_segs  # noqa: E402

ORTHANC_URL = os.environ.get("ORTHANC_URL", "http://127.0.0.1:18042")
# In the image: the packaged model. In the repo: the workspace's copy.
MODELS = Path(os.environ.get("HOLOSCAN_MODEL_PATH", Path(__file__).resolve().parents[3] / "VascX_App" / "models"))


def fetch_from_orthanc(level: str, key: str, value: str, folder: Path) -> None:
    """Every DICOM file of one patient or study, from Orthanc's REST API into folder."""
    request = urllib.request.Request(f"{ORTHANC_URL}/tools/find", json.dumps({"Level": level, "Query": {key: value}}).encode())
    ids = json.loads(urllib.request.urlopen(request).read())
    if not ids:
        raise LookupError(f"no {level.lower()} {value} in Orthanc at {ORTHANC_URL}")
    for orthanc_id in ids:
        archive = urllib.request.urlopen(f"{ORTHANC_URL}/{level.lower()}s/{orthanc_id}/archive").read()
        zipfile.ZipFile(io.BytesIO(archive)).extractall(folder)


def stored_in_orthanc(sop_instance_uid: str):
    """The dataset Orthanc holds under this SOPInstanceUID, or None."""
    request = urllib.request.Request(f"{ORTHANC_URL}/tools/find",
                                     json.dumps({"Level": "Instance", "Query": {"SOPInstanceUID": sop_instance_uid}}).encode())
    ids = json.loads(urllib.request.urlopen(request).read())
    if not ids:
        return None
    return pydicom.dcmread(io.BytesIO(urllib.request.urlopen(f"{ORTHANC_URL}/instances/{ids[0]}/file").read()))


def to_bytes(ds) -> bytes:
    buffer = DicomBytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def run(results, version=model_spec.ALGORITHM_VERSION):
    segs = build_segs(
        results,
        segment=model_spec.SEGMENT,
        algorithm_name=model_spec.ALGORITHM_NAME,
        algorithm_version=version,
        manufacturer="QTIM",
        series_description=f"AI {model_spec.ALGORITHM_NAME} {version} - research use only",
    )
    return {seg.SOPInstanceUID: to_bytes(seg) for seg in segs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--patient", help="PatientID in Orthanc")
    which.add_argument("--study", help="StudyInstanceUID in Orthanc")
    args = parser.parse_args()

    sources = []
    with tempfile.TemporaryDirectory() as folder:
        if args.patient:
            fetch_from_orthanc("Patient", "PatientID", args.patient, Path(folder))
        else:
            fetch_from_orthanc("Study", "StudyInstanceUID", args.study, Path(folder))
        for path in sorted(p for p in Path(folder).rglob("*") if p.is_file()):
            try:
                ds = pydicom.dcmread(path)
                model_spec.to_rgb(ds)
            except (InvalidDicomError, OSError, AttributeError, model_spec.UnsupportedImage):
                continue
            sources.append(ds)
    if len(sources) < 2:
        print(f"need at least 2 colour fundus photos, found {len(sources)}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file = checked_model_file(MODELS / model_spec.MODEL_NAME, model_spec.MODEL_SHA256)
    model = torch.jit.load(str(model_file), map_location=device).eval()

    def infer():  # fresh inference each time: inference determinism is part of the claim
        return [(ds, model_spec.segment(model, ds, device)) for ds in sources]

    checks = []
    first, second = run(infer()), run(infer())
    checks.append(("same input twice: same UIDs, identical bytes", first == second))

    alone = {}
    for result in infer():
        alone.update(run([result]))
    checks.append(("each image alone == all in one batch", alone == first))

    bumped = run(infer(), version=model_spec.ALGORITHM_VERSION + "-test")
    checks.append(("version bump: all-new UIDs", not (bumped.keys() & first.keys()) and len(bumped) == len(first)))

    partial = run(infer()[:-1])
    resumed = run(infer())
    checks.append(("re-run after a partial run: only the missing SEG is new",
                   len(resumed.keys() - partial.keys()) == 1 and all(resumed[uid] == partial[uid] for uid in partial)))

    # Compare content, not file bytes: a C-STORE transfer rewrites the file meta header.
    stored = {uid: stored_in_orthanc(uid) for uid in first}
    present = {uid: ds for uid, ds in stored.items() if ds is not None}
    drifted = [uid for uid, ds in present.items() if ds != pydicom.dcmread(io.BytesIO(first[uid]))]
    name = f"matches Orthanc: {len(present) - len(drifted)}/{len(present)} stored SEGs identical"
    if not present:
        checks.append(("matches Orthanc: nothing stored yet under these UIDs", None))
    else:
        checks.append((name + (f"; DRIFT in {len(drifted)}: same version, different output. "
                               "Bump ALGORITHM_VERSION or find the drift." if drifted else ""), not drifted))

    for name, ok in checks:
        print(f"{'SKIP' if ok is None else 'PASS' if ok else 'FAIL'}  {name}")
    print(f"{len(sources)} photo(s) on {device}, {model_spec.ALGORITHM_NAME} {model_spec.ALGORITHM_VERSION}")
    return 1 if any(ok is False for _, ok in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
