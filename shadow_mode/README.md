# Shadow Mode

> **Parked (2026-10-21).** Orchestration now uses MONAI Deploy Express (`MONAI_Deploy_Express/LOCAL_SETUP.md`).
> Kept as-is for later; `python -m shadow_mode.run_map` is still handy for running a MAP on local files.

AI inference running *alongside* the imaging workflow rather than inside it.

> Moving this to another machine? Read **[HANDOFF.md](HANDOFF.md)** first. The
> model, the test DICOMs and the MAP image are all absent from git and get
> **rebuilt from source** on the target host — `prepare_data.py`, `train.py`,
> `monai-deploy package`. Do not transfer the image.

Target demo: a DICOM lands in the dashboard Orthanc → Shadow Mode finds it
eligible → runs an existing MAP → writes a standards-based result back to
Orthanc. This directory is the service; the tutorial app folders at the repo
root are reference material and stay untouched. A MAP is consumed as a built
container image, not as source.

## Status

| Phase | | |
|---|---|---|
| 1 | inspect the repo | done |
| 2 | **local MAP runner** | **done — this code** |
| 3 | Orthanc input | not started (blocked on credentials for `:8043`) |
| 4 | upload SR back to Orthanc | SR linking solved in the MAP (`maps/aireadi_laterality`); upload not built |
| 5 | Orthanc trigger | not started |

## Phase 2 usage

```bash
# from the repo root, in the monai_deploy pyenv
python -m shadow_mode.run_map AIREADI_Laterality_App/demo_input/<file>.dcm
```

```
[docker] laterality ok in 9.0s -> 2 artifact(s) in shadow_mode/runs/laterality-20260921T222646Z
  .../1.2.826.0.1.3680043.8.498.6776....dcm
  .../output.json

prediction: R
```

Options:

```bash
--map laterality           # registered MAP name; --list shows them
--backend docker           # or monai-deploy
-o DIR                     # default shadow_mode/runs/<map>-<utc-timestamp>
--gpus none                # CPU only
--bin /path/to/executable  # e.g. monai-deploy outside the pyenv
-v                         # echo MAP stdout/stderr
```

Both backends are verified working against the laterality MAP (~8–9 s, correct
prediction on the demo image).

## Layout

```
config.py                 MAP registry: name -> image, gpu, timeout, accepted suffixes
execution/executor.py     MapExecutor interface + staging, pre-flight, timeout, collection
execution/docker_backend.py         `docker run`
execution/monai_deploy_backend.py   `monai-deploy run`
run_map.py                CLI
```

`MapExecutor.run(map_name, input_path, output_path) -> RunResult` is the only
contract the layers above may depend on. Backends implement `_build_command`
and nothing else, so a Workflow Manager adapter could be added later without
touching discovery, eligibility, routing or provenance.

The executor knows nothing about ophthalmology. Which MAP runs against which
instance is the eligibility layer's decision, and that layer does not exist yet.

## Which backend

**`docker` is the default.** It needs no Holoscan CLI in the service
environment, and because it passes no `--user`, the container runs as the
image's own intended user — which is by construction the identity the MAP was
packaged with. The UID/GID mismatch failure mode becomes structurally
impossible rather than configured-around.

`monai-deploy` is kept because it is the vendor-supported path the app READMEs
document, which makes it the right thing to reach for when deciding whether a
failure is ours or the SDK's. It requires `holoscan-cli` on the path and passes
`--uid/--gid` from `MapSpec.package_uid/gid` (1000:1000), never `os.getuid()`.
Getting that wrong drops the packaging user's `--user` site-packages off
`sys.path` and every import inside the MAP fails with a `ModuleNotFoundError`
that looks like a missing dependency but is really a mismatched identity.

## Two constraints worth knowing

**One input per run.** The laterality MAP's loader does
`next(input_folder.glob("*.dcm"))`, so staging two files would make the result
depend on glob order. The executor stages exactly one file into a private temp
directory and refuses an ambiguous input directory rather than picking for you.

**Permissions.** `mkdtemp` yields `0700` owned by the calling user; bind-mounted
into a container running as uid 1000 that is unreadable. Staging widens the
directory and file, and a pre-flight check rejects an unwritable output
directory with a clear message instead of letting it fail deep inside Holoscan.

## Known gap for Phase 4

The SR this MAP produces is valid Basic Text SR but **orphaned** — empty
`PatientID`/`PatientName`, a freshly generated `StudyInstanceUID`, and no
`ReferencedSeriesSequence` or `CurrentRequestedProcedureEvidenceSequence`.

Setting `copy_tags=True` in the app does not fix this. In
`monai/deploy/operators/dicom_utils.py::write_common_modules`, `copy_tags` only
copies patient demographics and reuses the source `StudyInstanceUID`; at no
setting does the SDK reference the source `SOPInstanceUID`. Uploading today's
output to Orthanc would produce a free-floating SR under a phantom patient.

Phase 4 therefore needs a reference-injection step in Shadow Mode, applied to
the MAP's output before upload — not a fork of the SDK.

Separately, `ModelInfo(..., "resnet18")` lands in `DeviceUID` (0018,1002), whose
VR is `UI`. pydicom warns; it is not a valid UID and Orthanc may object.
