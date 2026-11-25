#!/usr/bin/env python3
"""Recent workflow runs, classified, so the two concurrency bugs show up as counts.

    python3 LOCAL_SCRIPTS/runs.py              # runs started in the last 30 minutes
    python3 LOCAL_SCRIPTS/runs.py --minutes 5 --list

Outcome per run (one run = one series x one workflow):
    succeeded        MAP ran and its export finished
    export stuck     export task still "Dispatched": the IG refused it ("already queued") and holds the
                     RabbitMQ message unacked; it is redelivered ~30 min later (stock IG; fixed by PR #544)
    router only      run "Succeeded" but only the router ran: no MAP, no result. Either the series isn't a
                     colour photo (normal) or the WM parser race hit it (stock WM; see FailedToParseCondition)
    MAP failed       the MAP task failed (e.g. no output for Aurora / mosaic photos, or out of GPU memory)
    running          not finished yet
Also counts the two bugs' log lines in the same window.
"""
import argparse
import datetime as dt
import json
import subprocess
import urllib.request

WM = "http://127.0.0.1:5001"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--list", action="store_true", help="also list each run")
    args = ap.parse_args()

    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=args.minutes)
    runs = json.load(urllib.request.urlopen(f"{WM}/workflowinstances?pageSize=100000"))["data"]
    names = {w["workflow_id"]: w["workflow"]["name"] for w in json.load(urllib.request.urlopen(f"{WM}/workflows?pageSize=1000"))["data"]}
    runs = [r for r in runs if dt.datetime.fromisoformat(r["start_time"][:19]).replace(tzinfo=dt.timezone.utc) >= since]

    table = {}
    for r in sorted(runs, key=lambda r: r["start_time"]):
        tasks = {t["task_id"]: t["status"] for t in r["tasks"]}
        export = next((s for k, s in tasks.items() if k.startswith("export")), None)
        mapped = [s for k, s in tasks.items() if k != "router" and not k.startswith("export")]
        if r["status"] == "Created" and export == "Dispatched":
            outcome = "export stuck"
        elif r["status"] == "Created":
            outcome = "running"
        elif list(tasks) == ["router"]:
            outcome = "router only"
        elif "Failed" in mapped or r["status"] == "Failed":
            outcome = "MAP failed"
        else:
            outcome = "succeeded"
        wf = names.get(r["workflow_id"], r["workflow_id"][:8])
        table.setdefault(wf, {}).setdefault(outcome, 0)
        table[wf][outcome] += 1
        if args.list:
            print(f"{r['start_time'][11:19]}  {wf:20} {outcome:13} {tasks}")

    columns = ["succeeded", "export stuck", "router only", "MAP failed", "running"]
    print(f"\nruns started in the last {args.minutes} min ({len(runs)}):")
    print(f"  {'workflow':20} " + " ".join(f"{c:>13}" for c in columns))
    for wf, counts in sorted(table.items()):
        print(f"  {wf:20} " + " ".join(f"{counts.get(c, 0):>13}" for c in columns))
    window = f"{args.minutes}m"
    for container, needle, what in (("mdl-ig", "already queued", "IG 'already queued' (export bug)"),
                                    ("mdl-wm", "FailedToParseCondition", "WM 'FailedToParseCondition' (router race)")):
        out = subprocess.run(["docker", "logs", container, "--since", window], capture_output=True, text=True)
        print(f"  {what}: {(out.stdout + out.stderr).count(needle)}")


if __name__ == "__main__":
    main()
