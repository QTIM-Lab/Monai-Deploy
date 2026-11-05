# Workflow Manager (WM)

Container `mdl-wm` (0.2.1-2), REST on 5001, config `configs/workflow-manager.json`, state in MongoDB.
Turns each IG payload into a run of every workflow whose `informatics_gateway.ae_title` matches the AE
the data was sent to.

## Workflows

Ours live in `workflows/` (upstream examples in `sample-workflows/`). Format: tasks of type `router`,
`docker` (runs a MAP via the Task Manager) or `export` (IG C-STOREs a task's output).

```bash
WM=http://127.0.0.1:5001
curl -s "$WM/workflowinstances?pageSize=1&status=Created"   # runs in progress (lists page at 10 by default)
curl -s $WM/workflows | python3 -m json.tool | grep -E '"workflow_id"|"name"|"version"'
curl -X POST $WM/workflows -H 'Content-Type: application/json' -d @workflows/aireadi-laterality.json   # register
# update: PUT with the original name wrapped around the new definition
python3 -c "import json;print(json.dumps({'original_workflow_name':'aireadi-laterality','workflow':json.load(open('workflows/aireadi-laterality.json'))}))" \
  | curl -X PUT $WM/workflows/<workflow_id> -H 'Content-Type: application/json' -d @-
curl -X DELETE $WM/workflows/<workflow_id>
```

Router conditions read DICOM tags, e.g. `"{{ context.dicom.series.any('0008','0060') }} == 'OP'"`.
A payload that matches no condition ends with `router Succeeded` and nothing else.

**`series.any(tag)` returns the first file's value**, not "does any file have it"; `series.all(tag)` returns
the value only if every file agrees (`DicomService.cs`). Hence the IG groups per series: one payload = one
series. Multi-value tags come back as a list, e.g. ImageType `[ORIGINAL, PRIMARY, , COLOR]`.

Operators (`Resovler/Conditional.cs`): `== != > < >= <=` (case-insensitive), `CONTAINS` / `NOT_CONTAINS`
(list membership, **case-sensitive**: give a list on the right, e.g. `CONTAINS ['COLOR', 'Color']`), `AND`/`OR`.
The WM log shows the resolved condition (`resolvedConditional=...`).

Give each workflow's export task its own id (`export-sr`, `export-seg`): the IG de-duplicates exports by task id,
so two workflows both using `export-sr` would collide on every image.

Keep MAPs from being started on series they will skip: a docker task's output can be optional, but an
`export` task with no files always fails (`ExportFilesNotFound`, even with `"mandatory": false`), turning
the run Failed. So filter in the router, not only inside the MAP.

`python3 LOCAL_SCRIPTS/register.py` creates or updates every `workflows/*.json` (skips unchanged) and
rebuilds the per-MAP AEs; prefer it to the raw API below.

Run the MAP as `python3 -u /opt/holoscan/app` (entrypoint `/bin/bash,-c`), not via the image's default
launcher: the launcher swallows the app's exit code, so a crashed MAP would report Succeeded.

## Known bug: overlapping runs delay exports (~30 min)

IG log `The export request export-sr is already queued for export`. Cause: the WM names export requests
after the task (`export-sr`) and the IG refuses a request whose name is already queued. Two studies
finishing together (e.g. both eyes of one visit) are enough. The refused message is never acknowledged,
so RabbitMQ re-delivers it after its consumer timeout: observed 2026-10-16, refused 21:17:18, delivered
21:49:55. Until then the run shows `export-sr Dispatched` and the SR waits in MinIO. Not lost, but refused
requests are redelivered **together**, so they collide again: one gets through per ~30-min cycle
(observed 2026-10-26: 2 refused at 19:21, 1 delivered 19:52, the other refused again). N stranded
exports take ~N×30 min. A parallel bulk send can stall results for hours. Fix exists upstream as IG PR #544 (open since 2026-03,
unmerged). **Pace bulk sends.**

## Database view

The WM API returns snake_case (`start_time`, `status: "Succeeded"`); MongoDB stores the C# names and enum
numbers (`StartTime`, `Status: 1`). Workflow status 1 = Succeeded, 2 = Failed; task status 3 = Succeeded,
4 = Failed (others not yet observed). See [storage.md](storage.md).

## Other 0.2.1 quirks

- An input taken from `context.executions.<task>.artifacts.<name>` keeps the producer's artifact name.
- A mandatory input whose name matches a previous task's output becomes a typed dependency; docker tasks
  never report types, so the next task is **silently skipped** and the run still says Succeeded. Avoid
  docker → docker hand-offs, or mark such inputs `"mandatory": false`.
- MinIO retention comes from the workflow's `dataRetentionDays` (default 3), not the app setting ([storage.md](storage.md)).
