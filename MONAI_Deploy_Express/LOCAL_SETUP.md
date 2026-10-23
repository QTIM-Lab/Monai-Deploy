# Local setup — control sheet

Upstream MONAI Deploy Express 0.6.0, running on this host (copied 2026-10-07). `README.md` and
`docker-compose.yml` are upstream and unmodified. Everything we added or changed is listed at the
bottom. Per-service detail: [LOCAL_DOCS/](LOCAL_DOCS/).

## Run

```bash
cd MONAI_Deploy_Express                                   # .env uses ${PWD}
mkdir -p .md/{minio,rabbitmq,orthanc,mongodb,mdtm,mdig}  # first time only
docker compose up -d        # start (state persists in .md/)
docker compose ps           # all healthy?
docker compose down         # stop
```

| Service | URL (localhost only) | Login |
|---|---|---|
| Orthanc UI/REST | http://127.0.0.1:18042 (DICOM 14242) | none |
| Informatics Gateway API | http://127.0.0.1:5000 (DICOM 104) | none |
| Workflow Manager API | http://127.0.0.1:5001 | none |
| RabbitMQ UI | http://127.0.0.1:15672 | admin / admin |
| MinIO console | http://127.0.0.1:9001 | minioadmin / minioadmin |

From another machine: `ssh -L 18042:127.0.0.1:18042 -L 15672:127.0.0.1:15672 ophlapps04`.

## How a scan flows, and where it is filtered

```
new series in Orthanc ─(Lua)─▶ IG AE MONAI-DEPLOY ─▶ Workflow Manager ─▶ router ─▶ MAP ─▶ export ─▶ Orthanc
```

| Filter | Where | Rule |
|---|---|---|
| 1. Send | `configs/orthanc-autoroute.lua` | every new **series**, unless it came from `MONAISCU` (the IG's export AE) |
| 2. Accept | IG AE `MONAI-DEPLOY` (`configs/config-ig.sh`) | drop SR and SEG SOP classes (loop guard) |
| 3. Choose | workflow `router` task | e.g. `Modality == 'OP'` → laterality MAP |
| 4. See | MAP's `model_spec.SELECTION_RULES` | which series the model accepts |

**The Lua** is not MONAI. It stands in for the routing rule a PACS admin would set ("send new images to
the AI gateway"); the IG can't watch Orthanc itself. It is deliberately dumb: it decides nothing about
which AI runs (filter 3) or what MONAI accepts (filter 2). Its one rule is the one only the sender can
apply: don't send back what MONAI just sent. It runs on Orthanc's `OnStableSeries`, i.e. `StableAge`
(15 s) after a series' last instance. To pause auto-run, remove `LuaScripts` from `configs/orthanc.json`
and `docker compose restart orthanc`. More: [LOCAL_DOCS/orthanc.md](LOCAL_DOCS/orthanc.md).

## Everyday commands

```bash
# What ran (the useful "queue" view)
curl -s http://127.0.0.1:5001/workflowinstances | python3 -c "
import sys,json; d=json.load(sys.stdin); d=d.get('data',d) if isinstance(d,dict) else d
for w in sorted(d, key=lambda w: w.get('start_time') or '')[-10:]:
    print(w['start_time'][:19], w['status'], [(t['task_id'], t['status']) for t in w['tasks']])"

# Send one study by hand (or Orthanc UI: study → Send to DICOM modality)
curl -X POST http://127.0.0.1:18042/modalities/MONAI-DEPLOY/store -d '"<orthanc-study-id>"'

# Register / update a workflow: LOCAL_DOCS/workflow-manager.md
# Logs
docker logs mdl-orthanc 2>&1 | grep autoroute     # what the Lua sent / skipped
docker logs mdl-ig --since 10m                    # receive, ignore, export
# Remove the exited MAP containers the Task Manager leaves behind
docker container prune -f --filter label=org.monai.deploy.app-sdk
```

## Gotchas

- **Overlapping runs delay exports ~30 min.** If two runs of one workflow export together, one is refused
  and RabbitMQ re-delivers it ~30 min later (upstream bug, fix unmerged). Meanwhile the run shows
  `export-sr Dispatched`. Pace bulk sends. [workflow-manager.md](LOCAL_DOCS/workflow-manager.md)
- **The Task Manager does not serialize jobs** (`concurrency: 1` only limits launches) and **never removes
  task containers**. [task-manager.md](LOCAL_DOCS/task-manager.md)
- **IG rejects every association above the disk watermark** (`/scratch90` ~97% full; set to 99).
- **MinIO is scratch:** inputs and outputs per run, deleted after `dataRetentionDays` (10). Orthanc is the archive.
- MAP images and workflows are not in git: build MAPs per `../maps/`, and re-POST `workflows/*.json`
  after wiping `.md/`.

## Local changes

| File | Change | Why |
|---|---|---|
| `.env` | all host ports on `127.0.0.1`; Orthanc on 14242/18042; MinIO image swap | default creds; dashboard Orthanc owns 4242/8042; `minio/minio` gone from Docker Hub |
| `docker-compose.override.yml` | run MinIO binary as root; mount the Lua | Bitnami image differs; auto-run |
| `configs/orthanc.json` | `LuaScripts`, `StableAge` 60 → 15 | auto-run on upload |
| `configs/orthanc-autoroute.lua` | **new**: filter 1 | stand-in for PACS routing |
| `configs/informatics-gateway.json` | `watermarkPercent` 95 → 99 | full disk |
| `configs/config-ig.sh` | AE created with `ignoredSopClasses` | filter 2, loop guard |
| `workflows/aireadi-laterality.json` | **new**: router → MAP → export | the laterality pipeline |
| `LOCAL_SETUP.md`, `LOCAL_DOCS/` | **new** | this |
