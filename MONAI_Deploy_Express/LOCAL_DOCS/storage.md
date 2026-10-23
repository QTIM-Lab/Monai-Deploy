# MinIO and MongoDB

## MinIO — scratch space, not an archive

Container `mdl-minio`, bucket `monaideploy`, console http://127.0.0.1:9001 (minioadmin / minioadmin),
data `.md/minio/data/`. Image swapped to `bitnamilegacy/minio:2024.6.11` (see `.env`).

The services never read or write Orthanc directly; they hand files to each other through MinIO. Per
payload it holds the received images, each task's output (the SRs) and sidecars. The WM deletes them
after `dataRetentionDays` (10, `configs/workflow-manager.json`). Orthanc stays the only long-term copy.

Consequence: for up to 10 days there is a second copy of every image sent to MONAI. A result whose
export was delayed (see [workflow-manager.md](workflow-manager.md)) waits here:
`.md/minio/data/monaideploy/<payload_id>/workflows/<run>/<task_execution>/`.

## MongoDB

Container `mdl-mongodb`, data `.md/mongodb/`. The services' shared record; no built-in UI. Use MongoDB
Compass: `ssh -L 27017:127.0.0.1:27017 ophlapps04`, connect to `mongodb://root:rootpassword@127.0.0.1:27017`.

| Database | Useful collections |
|---|---|
| `InformaticsGateway` | `MonaiApplicationEntity` (AEs, ignoredSopClasses), `SourceApplicationEntity`, `DestinationApplicationEntity`, `DicomAssociationInfo` (every inbound association) |
| `WorkloadManager` (WM + TM) | `Workflows` (incl. old versions), `WorkflowInstances` (runs + task status), `Payloads`, `TaskDispatchEvents`, `ExecutionStats` |

```bash
docker exec -it mdl-mongodb mongosh -u root -p rootpassword    # shell
```

Wiping `.md/mongodb/` loses registered workflows (re-POST `workflows/*.json`) and IG config (recreated by
`configure-ig` at start).
