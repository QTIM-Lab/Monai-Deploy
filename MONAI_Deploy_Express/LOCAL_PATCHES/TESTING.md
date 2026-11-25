# Testing the fixes: stock → PR #544 → PR #544 + WM fix

How the evidence in `README.md` (and the upstream PRs) was produced. Commands from `MONAI_Deploy_Express/`.

## Demo: one participant, laterality only

Same block per mode; only the `switch.sh` line changes (`stock`, `pr`, `fixed`). Needs `AIREADI_ROOT` in `.env`.

```bash
LOCAL_SCRIPTS/reset.sh monai                                   # empty queues, MinIO, run history (Orthanc images kept)
LOCAL_SCRIPTS/reset.sh orthanc                                 # wipe Orthanc state
LOCAL_PATCHES/switch.sh stock                                  # stock | pr | fixed
LOCAL_SCRIPTS/load_aireadi.sh 7034 --replace                   # fresh participant in Orthanc, no results yet
( for s in $(curl -s -X POST http://127.0.0.1:18042/tools/find -d '{"Level":"Series","Query":{"PatientID":"7034","Modality":"OP"}}' | tr -d '[]",'); do curl -s -X POST http://127.0.0.1:18042/modalities/LATERALITY/store -d "{\"Resources\":[\"$s\"],\"Synchronous\":true}" >/dev/null & done; wait )   # laterality (SR)
python3 LOCAL_SCRIPTS/runs.py --minutes 15                     # ~2 min later
python3 LOCAL_SCRIPTS/results.py 7034
```

Same test for the other two models: replace the send line with one of these (one model per run, so the counts stay clean):

```bash
( for s in $(curl -s -X POST http://127.0.0.1:18042/tools/find -d '{"Level":"Series","Query":{"PatientID":"7034","Modality":"OP"}}' | tr -d '[]",'); do curl -s -X POST http://127.0.0.1:18042/modalities/VESSELS/store -d "{\"Resources\":[\"$s\"],\"Synchronous\":true}" >/dev/null & done; wait )      # vessel segmentation (SEG)
( for s in $(curl -s -X POST http://127.0.0.1:18042/tools/find -d '{"Level":"Series","Query":{"PatientID":"7034","Modality":"OP"}}' | tr -d '[]",'); do curl -s -X POST http://127.0.0.1:18042/modalities/DISC-FOVEA/store -d "{\"Resources\":[\"$s\"],\"Synchronous\":true}" >/dev/null & done; wait )   # disc/fovea detection (SR)
```

Vessels and disc/fovea skip the 2 mosaics too (max 23 results), and their MAPs are slower (vessels ~4 GB GPU each,
so 28 at once may queue on the GPUs). Their exports spread out more, so the gateway bug shows less often than with
laterality; the router race shows the same way.

| Mode | export stuck / "already queued" | router only / `FailedToParseCondition` | laterality SRs (max 25) |
|---|---|---|---|
| stock | 6–7 (unacked; 30-min redelivery, then dead-lettered) | 7–11 | 8–12 |
| pr | 0 | 5–11 | 14–21 |
| fixed | 0 | 0 | 25 |

Counts vary run to run (timing races); stuck/skipped are never 0 on stock, always 0 when fixed.
7034 has 30 colour photos in 28 series; 5 are Optomed Aurora, which the App SDK's loader drops (separate issue).

## Where to look

| What | Where | Bug signature |
|---|---|---|
| Export queue | RabbitMQ http://127.0.0.1:15672 (admin/admin) → Queues → `md.export.request.monaiscu` | Unacked > 0 with rates 0 (gateway bug); later `…-dead-letter` Ready > 0 |
| Who holds them | RabbitMQ → Channels → *Unacked* column (gateway = `172.29.0.50:*`) | |
| Gateway log | `docker logs mdl-ig --since 15m 2>&1 \| grep "already queued"` | gateway bug |
| RabbitMQ timeout | `docker logs mdl-rabbitmq --since 1h 2>&1 \| grep "timed out waiting for delivery acknowledgement"` | 30-min redelivery |
| WM log | `docker logs mdl-wm --since 15m 2>&1 \| grep FailedToParseCondition` | router race (invisible in RabbitMQ) |
| Results | Orthanc http://127.0.0.1:18042 → participant: studies with `OP,SR` vs `OP` only | missing SRs |

Force the 30-minute redelivery now: `docker compose restart informatics-gateway` (or Force Close the gateway's
connection in RabbitMQ → Connections).

## Unit tests (C#, no Express needed)

Run in the fix's clone under `../upstream/`; the container provides .NET 8, `chown` returns the build output to you.

```bash
cd ../upstream/monai-deploy-informatics-gateway && git checkout pr-544
docker run --rm -v "$PWD":/src -v ~/.cache/nuget-docker:/root/.nuget/packages -w /src mcr.microsoft.com/dotnet/sdk:8.0 bash -c "
  dotnet test src/InformaticsGateway/Test/Monai.Deploy.InformaticsGateway.Test.csproj;
  chown -R $(id -u):$(id -g) /src /root/.nuget/packages"          # 378 passed, 0 failed, 1 skipped (pre-existing)
git checkout -- . && git checkout main

cd ../monai-deploy-workflow-manager && git checkout fix-parser-race
docker run --rm -v "$PWD":/src -v ~/.cache/nuget-docker:/root/.nuget/packages -w /src mcr.microsoft.com/dotnet/sdk:8.0 bash -c "
  dotnet test tests/UnitTests/WorkflowExecuter.Tests/Monai.Deploy.WorkflowManager.WorkflowExecuter.Tests.csproj --filter 'FullyQualifiedName~ConditionalParameterParserTests';
  chown -R $(id -u):$(id -g) /src /root/.nuget/packages"          # 27 passed; with develop's parser the new test fails
```

Unit tests use mocks (no imaging data): they prove the changed logic. The demo above proves the effect on real
DICOM through the real services.
