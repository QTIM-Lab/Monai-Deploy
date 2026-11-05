#!/usr/bin/env python3
"""Send series from the practice Orthanc to a MONAI AE, ONE AT A TIME, waiting for each run to finish.

    python3 LOCAL_SCRIPTS/send.py --patient 1015
    python3 LOCAL_SCRIPTS/send.py --study 2.16.840.1.114517.10.5.1.2.907063120230821141408
    python3 LOCAL_SCRIPTS/send.py --all --modality OP --skip-done "aireadi-laterality 0.2.0"

This is the sender side, standing in for a PACS operator pressing "send" in a loop. It only chooses
what to send and when; which AI runs is still decided by the AE (--ae, default LATERALITY) and the
workflow's router.

Why one at a time: overlapping runs of one workflow make the IG delay an export ~30 min (upstream bug),
and the Task Manager would start one GPU container per series at once. So: send a series, wait until
the Workflow Manager has a new run and no run is still in progress, then send the next.

--skip-done "<algorithm> <version>" makes re-runs idempotent: a series is skipped when its study
already holds an SR or SEG whose SeriesDescription starts with "AI <algorithm> <version>" and which
references that series. Laterality: "aireadi-laterality 0.2.0"; VascX: "vascx-vessels 0.2.0". --modality is only a send-side pre-filter to avoid shipping irrelevant series.
"""

import argparse
import json
import sys
import time
import urllib.request

ORTHANC = "http://127.0.0.1:18042"
WM = "http://127.0.0.1:5001"
AI_OUTPUT_MODALITIES = {"SR", "SEG"}  # never send results back in


def get(url):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode())


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        text = resp.read().decode()
        return json.loads(text) if text else None


def runs(status=None):
    q = f"?pageSize=1&status={status}" if status else "?pageSize=1"
    return get(f"{WM}/workflowinstances{q}")["totalRecords"]


def find_series(args):
    query = {}
    if args.patient:
        query["PatientID"] = args.patient
    if args.study:
        query["StudyInstanceUID"] = args.study
    if args.modality:
        query["Modality"] = args.modality
    found = post(f"{ORTHANC}/tools/find", {"Level": "Series", "Query": query, "Expand": True})
    return [s for s in found if s["MainDicomTags"].get("Modality") not in AI_OUTPUT_MODALITIES]


def done_series(study_id, algorithm, cache={}):
    """Series UIDs in this study already referenced by an SR or SEG from `algorithm` ("name version")."""
    if study_id in cache:
        return cache[study_id]
    refs = set()
    for series in get(f"{ORTHANC}/studies/{study_id}/series"):
        tags = series["MainDicomTags"]
        if tags.get("Modality") not in AI_OUTPUT_MODALITIES or not tags.get("SeriesDescription", "").startswith(f"AI {algorithm}"):
            continue
        for instance in series["Instances"]:
            result = get(f"{ORTHANC}/instances/{instance}/tags?simplify")
            for study in result.get("CurrentRequestedProcedureEvidenceSequence", []):  # SR
                for ref in study.get("ReferencedSeriesSequence", []):
                    refs.add(ref.get("SeriesInstanceUID"))
            for ref in result.get("ReferencedSeriesSequence", []):  # SEG
                refs.add(ref.get("SeriesInstanceUID"))
    cache[study_id] = refs
    return refs


def wait_for_run(before_total, appear_timeout=90, finish_timeout=1800):
    """Wait for a new run to appear, then for nothing to be in progress. Returns a short outcome."""
    deadline = time.time() + appear_timeout
    while runs() <= before_total:
        if time.time() > deadline:
            return "no run started (IG ignored it, or no workflow for this AE)"
        time.sleep(2)
    deadline = time.time() + finish_timeout
    while runs("Created") > 0:
        if time.time() > deadline:
            return "TIMEOUT waiting for the run to finish"
        time.sleep(3)
    return "done"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--patient", help="PatientID")
    target.add_argument("--study", help="StudyInstanceUID")
    target.add_argument("--all", action="store_true", help="every series in Orthanc")
    ap.add_argument("--ae", default="LATERALITY", help="Orthanc modality to send to (default LATERALITY)")
    ap.add_argument("--modality", help="only send series with this Modality, e.g. OP")
    ap.add_argument("--skip-done", metavar="'ALGORITHM VERSION'", help="skip series that already have this SR/SEG")
    ap.add_argument("--dry-run", action="store_true", help="list what would be sent")
    args = ap.parse_args()

    if runs("Created") > 0:
        sys.exit("A run is still in progress; wait for it (or check LOCAL_SETUP 'What ran') before sending.")

    series_list = find_series(args)
    print(f"{len(series_list)} series matched")
    counts = {"sent": 0, "skipped": 0, "problems": 0}
    for i, series in enumerate(series_list, 1):
        tags = series["MainDicomTags"]
        label = f"[{i}/{len(series_list)}] {tags.get('Modality', '?')} series {tags.get('SeriesInstanceUID', '?')}"
        if args.skip_done and tags.get("SeriesInstanceUID") in done_series(series["ParentStudy"], args.skip_done):
            print(f"{label}: already has an SR from {args.skip_done}, skipped")
            counts["skipped"] += 1
            continue
        if args.dry_run:
            print(f"{label}: would send to {args.ae}")
            continue
        before, failed_before, start = runs(), runs("Failed"), time.time()
        post(f"{ORTHANC}/modalities/{args.ae}/store", {"Resources": [series["ID"]], "Synchronous": True})
        outcome = wait_for_run(before)
        if outcome == "done" and runs("Failed") > failed_before:
            outcome = "run FAILED (see LOCAL_SETUP 'What ran')"
        counts["sent"] += 1
        counts["problems"] += outcome != "done"
        print(f"{label}: {outcome} in {time.time() - start:.0f}s")

    print(f"sent {counts['sent']}, skipped {counts['skipped']}, problems {counts['problems']}")
    return 1 if counts["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
