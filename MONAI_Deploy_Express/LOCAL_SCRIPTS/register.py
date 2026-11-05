#!/usr/bin/env python3
"""Register our workflows and per-MAP AEs with the running stack. Safe to re-run.

    python3 LOCAL_SCRIPTS/register.py

1. Every workflows/*.json is created, or updated in place if a workflow with that name exists
   (skipped when unchanged, so re-runs don't pile up revisions).
2. Every per-MAP AE in MAP_AES is (re)created on the Informatics Gateway with `workflows` set to that
   MAP's workflow id, so a study sent to that AE runs that one workflow and nothing else. Grouping and
   ignored SOP classes are copied from the MONAI-DEPLOY AE (configs/config-ig.sh is their source).
   Each AE also needs a matching DicomModalities entry in configs/orthanc.json.

Workflow ids change when MongoDB is wiped, which is why the AEs are made here and not in config-ig.sh.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

IG = "http://127.0.0.1:5000"
WM = "http://127.0.0.1:5001"
WORKFLOWS = Path(__file__).resolve().parent.parent / "workflows"

# AE title (max 16 characters) -> workflow name
MAP_AES = {"LATERALITY": "aireadi-laterality", "VESSELS": "fundus-vessels"}


def call(method, url, body=None, ok=(200, 201)):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode()
            return resp.status, (json.loads(text) if text else None)
    except urllib.error.HTTPError as exc:
        if exc.code in ok:
            return exc.code, None
        return exc.code, exc.read().decode()[:300]


def wait_for_services(timeout=180):
    """Until the WM answers and configure-ig has created the MONAI-DEPLOY AE (a few seconds after start)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if call("GET", f"{WM}/workflows?pageSize=1")[0] == 200 and call("GET", f"{IG}/config/ae/MONAI-DEPLOY")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(3)
    sys.exit("Workflow Manager / MONAI-DEPLOY AE not ready; is the stack up? (docker compose ps)")


def same(stored, ours):
    """True if `stored` holds everything in `ours`. The WM adds defaults and changes key case on save."""
    if isinstance(ours, dict):
        if not isinstance(stored, dict):
            return False
        stored = {k.lower(): v for k, v in stored.items()}
        return all(k.lower() in stored and same(stored[k.lower()], v) for k, v in ours.items())
    if isinstance(ours, list):
        return isinstance(stored, list) and len(stored) == len(ours) and all(map(same, stored, ours))
    return stored == ours


def register_workflows():
    _, listing = call("GET", f"{WM}/workflows?pageSize=1000")
    existing = {w["workflow"]["name"]: w for w in listing["data"]}
    ids = {}
    for path in sorted(WORKFLOWS.glob("*.json")):
        wf = json.loads(path.read_text())
        name = wf["name"]
        if name in existing:
            current = existing[name]
            ids[name] = current["workflow_id"]
            if same(current["workflow"], wf):
                print(f"workflow {name} {wf['version']}: unchanged")
                continue
            status, resp = call("PUT", f"{WM}/workflows/{current['workflow_id']}",
                                {"original_workflow_name": name, "workflow": wf})
            print(f"workflow {name} {wf['version']}: updated ({status})")
        else:
            status, resp = call("POST", f"{WM}/workflows", wf)
            if status not in (200, 201):
                sys.exit(f"workflow {name}: register failed ({status}): {resp}")
            ids[name] = resp["workflow_id"]
            print(f"workflow {name} {wf['version']}: created")
    return ids


def register_map_aes(ids):
    status, template = call("GET", f"{IG}/config/ae/MONAI-DEPLOY")
    if status != 200:
        sys.exit("MONAI-DEPLOY AE missing; configure-ig should create it at stack start")
    for ae, workflow in MAP_AES.items():
        if workflow not in ids:
            sys.exit(f"AE {ae}: workflow {workflow!r} not found in workflows/")
        call("DELETE", f"{IG}/config/ae/{ae}", ok=(200, 404))
        body = {
            "name": ae,
            "aeTitle": ae,
            "workflows": [ids[workflow]],
            "grouping": template["grouping"],
            "timeout": template["timeout"],
            "ignoredSopClasses": template["ignoredSopClasses"],
        }
        status, resp = call("POST", f"{IG}/config/ae", body)
        if status not in (200, 201):
            sys.exit(f"AE {ae}: create failed ({status}): {resp}")
        print(f"AE {ae} -> {workflow} ({ids[workflow][:8]}), grouping {template['grouping']}")


if __name__ == "__main__":
    wait_for_services()
    register_map_aes(register_workflows())
