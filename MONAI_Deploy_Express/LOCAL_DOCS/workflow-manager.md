# Workflow Manager (WM)

Container `mdl-wm` (0.2.1-2), REST on 5001, config `configs/workflow-manager.json`, state in MongoDB.
Turns each IG payload into a run of every workflow whose `informatics_gateway.ae_title` matches the AE
the data was sent to.

## Workflows

Ours live in `workflows/` (upstream examples in `sample-workflows/`). Format: tasks of type `router`,
`docker` (runs a MAP via the Task Manager) or `export` (IG C-STOREs a task's output).

```bash
WM=http://127.0.0.1:5001
curl -s $WM/workflows | python3 -m json.tool | grep -E '"workflow_id"|"name"|"version"'
curl -X POST $WM/workflows -H 'Content-Type: application/json' -d @workflows/aireadi-laterality.json   # register
# update: PUT with the original name wrapped around the new definition
python3 -c "import json;print(json.dumps({'original_workflow_name':'aireadi-laterality','workflow':json.load(open('workflows/aireadi-laterality.json'))}))" \
  | curl -X PUT $WM/workflows/<workflow_id> -H 'Content-Type: application/json' -d @-
curl -X DELETE $WM/workflows/<workflow_id>
```

Router conditions read DICOM tags, e.g. `"{{ context.dicom.series.any('0008','0060') }} == 'OP'"`.
A study that matches no condition ends with `router Succeeded` and nothing else.

Run the MAP as `python3 -u /opt/holoscan/app` (entrypoint `/bin/bash,-c`), not via the image's default
launcher: the launcher swallows the app's exit code, so a crashed MAP would report Succeeded.

## Known bug: overlapping runs delay exports (~30 min)

IG log `The export request export-sr is already queued for export`. Cause: the WM names export requests
after the task (`export-sr`) and the IG refuses a request whose name is already queued. Two studies
finishing together (e.g. both eyes of one visit) are enough. The refused message is never acknowledged,
so RabbitMQ re-delivers it after its consumer timeout: observed 2026-10-16, refused 21:17:18, delivered
21:49:55. Until then the run shows `export-sr Dispatched` and the SR waits in MinIO. Not lost, but under
a bulk send every collision is a 30-min stall. Fix exists upstream as IG PR #544 (open since 2026-03,
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
- `dataRetentionDays: 10` also governs MinIO cleanup ([storage.md](storage.md)).
