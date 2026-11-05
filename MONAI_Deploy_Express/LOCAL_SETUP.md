# Local setup — control sheet

Upstream MONAI Deploy Express 0.6.0 on this host. `README.md` and `docker-compose.yml` are upstream and
unmodified; our changes are listed at the bottom. Detail per service: [LOCAL_DOCS/](LOCAL_DOCS/).

## Run

```bash
cd MONAI_Deploy_Express                                   # .env uses ${PWD}
mkdir -p .md/{minio,rabbitmq,orthanc,mongodb,mdtm,mdig}  # first time only
docker compose up -d   |   docker compose ps   |   docker compose down
```

| Service | URL (localhost only; forward the port) | Login |
|---|---|---|
| Orthanc | http://127.0.0.1:18042 (DICOM 14242) | none |
| Informatics Gateway API | http://127.0.0.1:5000 (DICOM 104) | none |
| Workflow Manager API | http://127.0.0.1:5001 | none |
| RabbitMQ / MinIO | http://127.0.0.1:15672 / :9001 | admin/admin, minioadmin/minioadmin |

## How a scan flows

```
send.py ─▶ IG AE MONAI-DEPLOY ─▶ every workflow ─▶ router ─▶ MAP ─▶ export ─▶ Orthanc (same study)
send.py ─▶ IG AE LATERALITY / VESSELS ─▶ only that model's workflow
```

| Filter | Where | Rule |
|---|---|---|
| 1. Send | `send.py`, or the auto-run Lua (**off by default**) | chosen series; Lua skips anything from `MONAISCU` |
| 2. Accept | IG AEs (`configs/config-ig.sh`) | per **series**; SR and SEG dropped (loop guard) |
| 3. Choose | workflow `router` | `Modality == 'OP' AND ImageType CONTAINS ['COLOR','Color']` |
| 4. See | MAP `model_spec.SELECTION_RULES` | same rule, owned by the model |

Results are deterministic per (image, model version) and Orthanc never overwrites, so re-sends add nothing.
Auto-run: `ORTHANC_AUTOROUTE=on docker compose up -d orthanc` (why it's off: [orthanc.md](LOCAL_DOCS/orthanc.md)).

## Scripts (`LOCAL_SCRIPTS/`, safe to re-run; system python is fine)

```bash
LOCAL_SCRIPTS/reset.sh orthanc|monai|all     # clear images | MONAI state | everything; re-registers
python3 LOCAL_SCRIPTS/register.py            # after editing workflows/*.json or adding a MAP AE
LOCAL_SCRIPTS/upload.sh <dicom folder>       # load DICOM (refuses while auto-run is on)
python3 LOCAL_SCRIPTS/send.py --patient <MRN> | --study <UID> | --all  --modality OP \
        [--ae LATERALITY|VESSELS|MONAI-DEPLOY] [--skip-done "aireadi-laterality 0.2.0"]
```

`send.py`: one series at a time, waits for each run. Default AE `LATERALITY`; `MONAI-DEPLOY` = all models.
~15–20 s per colour series, ~6 s per skipped one.

## Everyday commands

```bash
curl -s "http://127.0.0.1:5001/workflowinstances?pageSize=100000" | python3 -c "
import sys,json; d=json.load(sys.stdin)['data']
for w in sorted(d, key=lambda w: w['start_time'])[-10:]: print(w['start_time'][:19], w['status'], [(t['task_id'], t['status']) for t in w['tasks']])"
docker logs mdl-ig --since 10m                                       # receive, ignore, export
docker container prune -f --filter label=org.monai.deploy.app-sdk    # exited MAP containers
```

## Gotchas

- **Overlapping runs delay exports ~30 min each** (upstream IG bug, PR #544). Run shows `export-… Dispatched`.
  Hence auto-run off and serial `send.py`. [workflow-manager.md](LOCAL_DOCS/workflow-manager.md)
- **Task Manager** runs jobs in parallel and leaves exited containers. [task-manager.md](LOCAL_DOCS/task-manager.md)
- **IG rejects all DICOM above the disk watermark** (`/scratch90` ~97% full; set to 99).
- **MinIO is scratch**, kept per workflow `dataRetentionDays` (1 day). [storage.md](LOCAL_DOCS/storage.md)
- MAP images aren't in git (build per `../maps/`); workflows/AEs live in MongoDB (`register.py` restores them).

## Upstream

`git subtree` of `Project-MONAI/monai-deploy` at `0c1f13a`; taking updates: [upstream.md](LOCAL_DOCS/upstream.md).

## Local changes

| File | Change | Why |
|---|---|---|
| `.env` | ports on `127.0.0.1`; Orthanc 14242/18042; MinIO image; `ORTHANC_AUTOROUTE=off` | security; dashboard owns 4242/8042; image gone; export bug |
| `docker-compose.override.yml` | MinIO as root; Lua mount; `AUTOROUTE` switch | Bitnami image; auto-run |
| `configs/orthanc.json` | Lua, `StableAge` 15, modalities `LATERALITY`/`VESSELS`, `OverwriteInstances` false | auto-run; targeted runs; never replace results |
| `configs/orthanc-autoroute.lua` | **new** | stand-in for a PACS routing rule |
| `configs/informatics-gateway.json` | `watermarkPercent` 99 | full disk |
| `configs/config-ig.sh` | grouping per series, `ignoredSopClasses` | correct routing; loop guard |
| `workflows/*.json` | **new**: laterality (SR), fundus-vessels (SEG) | the pipelines |
| `LOCAL_SETUP.md`, `LOCAL_DOCS/`, `LOCAL_SCRIPTS/` | **new** | this |
