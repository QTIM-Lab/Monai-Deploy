# Shadow Mode — handoff to a dev-server session

You are picking up a MONAI Deploy prototype that was built and verified on a
different machine. This document is the complete context: the goal, what exists,
what is verified, what is known-broken, and what you must re-derive locally
because it does not travel with the repository.

Read the whole thing before running anything. **§5 (Bootstrap)** is the one that
will waste your afternoon if you skip it: the model, the test fixtures and the
MAP image are all absent from git, and the user wants them **rebuilt on your
dev server from source** rather than copied over. §11 is the checklist version
of the same thing.

---

## 0. Current state — verified on `ophlapps04`, 2026-09-24

Checked directly, not assumed. If you are on `ophlapps04`, most of §5 is already
done and you can skip to the one missing step.

| Thing | State |
|---|---|
| pyenv `/persist/.../monai_deploy` | **present** |
| AI-READI dataset + `manifest.tsv` | **present** |
| Docker daemon | **reachable** |
| GPUs | **4x** RTX A4000, driver 580.173.02 (§4 says 3x — it is 4) |
| `cache/*.npy` (train/val/test) | **present**, built 2026-08-26 |
| `models/model/model.ts` | **present**, 44 MB, 2026-08-28 |
| `demo_input/*.dcm` | **1 file**, not 3 — see note below |
| MAP image `airead_laterality_app-...:1.0` | **ABSENT** — must be packaged |
| `.env` | **absent**; `.env.example` present |
| Dashboard Orthanc | **running here** — `orthancteam/orthanc:26.8.2` |

**So the only build step outstanding is §5 step 4 (`monai-deploy package`).**
`prepare_data.py` and `train.py` do not need re-running — their outputs are on
disk. That turns the bootstrap from an afternoon into a few minutes.

`demo_input/` holds one fixture (`1002_..._cfp_r_...dcm`) though `N_DEMO = 3`
in `prepare_data.py:32`. One fixture is enough for a Phase 2 smoke test; re-run
`prepare_data.py` if you want all three. Splits are deterministic (`SEED = 0`,
by patient), so the same three would be selected.

### Git state — the thing most likely to bite you

**Nothing in this subsystem is committed or pushed.** `HEAD == origin/main`,
and `shadow_mode/` is untracked. Not on `origin`
(`github.com/QTIM-Lab/Monai-Deploy`): `shadow_mode/` in its entirety,
`PRESENTATION_OUTLINE.md`, `.env.example`, the `.gitignore` additions, and edits
to `laterality_classifier_operator.py` and `train.py`.

A fresh `git clone` on another server therefore gets **none** of this work.
Until it is committed and pushed, this working tree on `ophlapps04` is the only
copy, and §5's "clone the repo" instruction cannot be followed.

---

## 1. The goal

Add a lightweight inference layer to an existing ophthalmology research app (the
**CU AAI Dashboard**: Vue 3 / Vuetify, Flask, PostgreSQL, Orthanc, DICOMweb,
Docker Compose, Nginx). The dashboard already has a working Orthanc full of
ophthalmic DICOM.

The demo that defines success:

> An ophthalmic DICOM entered our existing Orthanc. Our Shadow Mode service
> recognized it as eligible, executed a MONAI Application Package, generated a
> standards-based AI result, and put that result back into Orthanc.

```
Orthanc ──> discovery/eligibility ──> executor ──> MAP ──> DICOM SR ──> Orthanc
```

"Shadow Mode" = AI runs *alongside* the normal research/clinical imaging
workflow without controlling it.

### Phase plan — strictly incremental

| Phase | | Status |
|---|---|---|
| 1 | Inspect the repo | **done** |
| 2 | Local MAP runner (`local DICOM -> MAP -> output`) | **done, verified** |
| 3 | Orthanc input (download instance, run MAP) | not started |
| 4 | Round trip (upload SR back, referencing source) | **SR linking solved in the MAP** (2026-10-16); round trip proven via MONAI Deploy Express, not yet via Shadow Mode's own Orthanc client |
| 5 | Orthanc trigger (OnStoredInstance / change polling) | not started |

Do not skip ahead. The user explicitly wants the smallest working vertical slice
at each step.

---

## 2. Architectural rules (non-negotiable)

**The MAP is an execution unit, not an orchestrator.** A MAP is:
`DICOM in -> preprocess -> inference -> DICOM result out`. Nothing else.

Everything else belongs to the surrounding service: discovering images in
Orthanc, deciding eligibility, choosing which MAP runs, staging input, tracking
run state, avoiding duplicate work, recording model/version provenance, writing
results back, retry/error handling, batch/backfill.

**Never bury ophthalmology business rules in the executor.** Keep selection and
routing separate from execution.

**Orthanc stays the DICOM authority.** Do not build a second DICOM index.
Persisted relationships use DICOM UIDs (`StudyInstanceUID`, `SeriesInstanceUID`,
`SOPInstanceUID`), not Orthanc's internal IDs. Orthanc IDs are fine as transient
values while talking to its REST API.

**Provenance stays generic.** When a schema eventually arrives it looks like
`models(id, name, version, map_identifier, created_at)` and
`inference_runs(id, model_id, source_sop_instance_uid, output_sop_instance_uid,
status, started_at, completed_at, error)`. Never per-model columns like
`laterality_prediction`.

**Do not fork MONAI Deploy.** If the SDK looks broken, diagnose and report it.
Source-level changes to the framework must be an explicit, visible engineering
decision with upstream ownership — not hidden private maintenance.

### Explicitly do NOT build

Kubernetes (never, at any stage, for any reason), MONAI Deploy Workflow Manager,
MONAI Informatics Gateway, Celery, Redis, message queues, a second DICOM
archive, an Orthanc replacement, microservices, a Dashboard redesign, or a
redesign of the existing DICOM ingestion pipeline.

The user's words: *"I specifically want to avoid turning this prototype into an
infrastructure project."* The first milestone must stay runnable and
understandable by one engineer.

### Two hazards to design against

1. **Recursive inference.** When Shadow Mode writes an SR back into Orthanc,
   Orthanc fires the same new-instance trigger. Without eligibility filtering:
   `image -> inference -> SR -> trigger -> inference -> ...` Eligibility must
   distinguish source images from AI-generated derived objects.
2. **Duplicate work at scale.** A backfill of roughly **66,000** existing images
   is planned. It must be idempotent and restartable — answer *"has model X
   version Y already processed SOPInstanceUID Z?"* and skip if yes. The
   implementation can start simpler, but design the boundaries so this drops in
   cleanly.

---

## 3. Repository layout

This is a **generic MONAI Deploy learning repo** — a ladder of tutorial apps —
that has grown a real subsystem. Respect the separation:

```
Simple_Image_Processing/              Tutorial 1: Sobel -> Median -> Gaussian. Working.
MedNIST_Classifier_App/               Tutorial 2: train-it-yourself.
MedNIST_Classifier_App_Pre-Built_Model/  Tutorial 2 variant.
AIREADI_Laterality_App/               The real model. Working, packaged, verified.
shadow_mode/                          The service. <-- our code lives here
PRESENTATION_OUTLINE.md
```

**User preference, stated directly:** new subsystems go in their own top-level
folder. Do not thread service code through the tutorial app directories — they
are reference material the user returns to, and mixing destroys their value as
clean worked examples. Treat the tutorial apps as read-only. **The MAP image is
the interface, not the app source.**

---

## 4. Environment

### Three distinct Pythons — never conflate them

| Environment | Python | Purpose |
|---|---|---|
| the `monai_deploy` pyenv | 3.10.0 | host dev + MAP packaging |
| inside a built MAP image | 3.12 | MAP runtime |
| bare `python` shim | 3.9.13 | **wrong env**, unrelated |

The pyenv is at:

```
/persist/python_virtual_environments/bearceb/.pyenv/versions/3.10.0/envs/monai_deploy
```

**`/persist` is a shared mount across all of these servers, so this is the same
path on your dev server** — you do not need to rebuild the environment. Confirm
it resolves before assuming it, but expect it to.

This matters more than it looks: `/persist` also carries the AI-READI dataset,
which is what makes the from-source bootstrap in §5 possible.

### Pinned versions — do not change casually

```
monai-deploy-app-sdk  4.0.0
holoscan-cli          4.2.0
holoscan-cu13         4.2.0
```

These were arrived at by debugging. Installing `monai-deploy-app-sdk` 4.0.0
pulls newer Holoscan by default; Holoscan CLI 4.2 then refuses to package:

```
Invalid SDK version specified (4.5.0): valid values are: 4.2.0
```

`holoscan-cu12` 4.5.0 was removed and must not be reintroduced. SDK 4.0.0
expects the **cu13** variant. If a version change looks genuinely necessary,
explain why and get agreement before touching anything.

### CUDA: host was 12, MAP is 13

Original host exposed `libcudart.so.12`, driver 580.173.02 (supports up to CUDA
13), 3x RTX A4000. Holoscan cu13 therefore could not run from the host venv —
`python app.py` directly on the host is **not** a supported path.

This is not a blocker: the MAP ships its own CUDA 13 userspace and reaches the
GPU through the NVIDIA container stack. **Always prefer containerized
execution.** Do not upgrade host CUDA on a shared server.

The conflict is Holoscan-specific. Plain PyTorch training does not import
Holoscan and cu12 torch wheels bundle their own userspace, so host-side GPU
*training* is viable even where host-side *Holoscan* is not.

**Re-check your dev server's CUDA version and GPU availability.** Do not assume
it matches.

### The UID/GID invariant — read before debugging any MAP import error

**Package-time UID/GID must equal runtime UID/GID.**

MAP dependencies install into the packaging user's user-site directory,
`/home/holoscan/.local/lib/python3.12/site-packages`. The MAP image's internal
user is `uid=1000(holoscan) gid=1000(holoscan)` (verified). `monai-deploy run`
defaults to substituting the *host* UID/GID (on the original machine,
`214196116:214196116`). Python then drops that directory from `sys.path` and
every import inside the MAP fails:

```
ModuleNotFoundError: No module named 'monai'
```

The module is present. The identity is wrong. Bind mounts are irrelevant; only
identity matters. Verify with:

```bash
# works — image's own user
docker run --rm --entrypoint python3 <image> -c "import monai; print(monai.__path__)"

# fails — host UID substituted
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python3 <image> \
  -c "import sys; print('\n'.join(sys.path)); import monai"
```

For a future shared deployment, do **not** bake in a personal UID — define a
fixed service identity (e.g. a `monai-deploy` service user) used consistently at
package and run time.

---

## 5. Bootstrap: rebuild everything from source — READ THIS

Cloning the repo gives you **source only**. Verified by `git ls-files`:

| Artifact | In git? | Size | Notes |
|---|---|---|---|
| App source, `app.yaml`, operators, `train.py` | **yes** | small | all four apps |
| `shadow_mode/` (this service) | **NO — untracked, see §0** | small | committed nowhere yet |
| `AIREADI_Laterality_App/models/model/model.ts` | **NO** | 44 MB | `.gitignore` has `*.ts` |
| `AIREADI_Laterality_App/demo_input/*.dcm` | **NO** | 174 KB | untracked |
| MAP image `airead_laterality_app-...:1.0` | **NO** | **8.78 GB** | local Docker only |
| AI-READI dataset | n/a | large | `/persist/public_datasets/ai_readi` — **shared mount, available to you** |

So you start with no model, no test DICOM and no MAP image.

**The user's explicit decision: re-derive and rebuild these on the dev server.
Do not `docker save` / transfer / `docker load` the image.** Rebuilding proves
the packaging path works on the target host, keeps the dev server independent of
a one-off 8.78 GB tarball, and means a rebuilt MAP is reproducible rather than a
mystery artifact someone once made. Treat rebuilding as the supported path and
image transfer as something to raise only if rebuilding is genuinely blocked.

This is practical because `/persist` is shared: both the pyenv (§4) and the
AI-READI dataset are already on your dev server. Nothing needs copying from the
original machine.

### The full from-source bootstrap

```bash
# 0. Activate the shared pyenv (see §4). Verify, do not assume:
python -c "import monai.deploy, holoscan; print(monai.deploy.__version__)"   # expect 4.0.0
monai-deploy --version
nvidia-smi

# 1. Point at the dataset. This is prepare_data.py's default, so the export is
#    only needed if your dev server mounts it elsewhere.
export AIREADI_ROOT=/persist/public_datasets/ai_readi
ls "$AIREADI_ROOT/retinal_photography/manifest.tsv"    # fail fast if absent

# 2. Build the training cache AND the test fixtures.
#    NOTE: paths inside prepare_data.py are relative to cwd, so the cd is required.
cd AIREADI_Laterality_App && python prepare_data.py && cd -

# 3. Train. ~40 s on one A4000. Writes models/model/model.ts
cd AIREADI_Laterality_App && python train.py && cd -
ls -la AIREADI_Laterality_App/models/model/model.ts

# 4. Package the MAP.
monai-deploy package AIREADI_Laterality_App \
  -c AIREADI_Laterality_App/app.yaml \
  -m AIREADI_Laterality_App/models \
  -t airead_laterality_app:1.0 \
  --platform x86_64 -l DEBUG

docker images | grep airead_laterality_app
```

Step 4 produces `airead_laterality_app-x64-workstation-dgpu-linux-amd64:1.0` —
the exact tag `shadow_mode/config.py` already expects. If you tag it differently,
update the registry in `config.py` rather than renaming by hand at run time.

### Step 2 solves the test-fixture problem too

`prepare_data.py` does double duty. Alongside `cache/*.npy` it copies `N_DEMO = 3`
held-out DICOMs into `AIREADI_Laterality_App/demo_input/` — untouched by
training, which is exactly what you want as a Phase 2 fixture. **You do not need
anyone to send you a test DICOM.**

Ground truth is in both the filename (`..._cfp_l_...` / `..._cfp_r_...`) and the
`ImageLaterality` tag, so you can check a prediction without any label file.

### What the filter selects

Maestro2 + Macula + Color Photography, laterality in `{L, R}`: about 2119 images
across 1036 patients, uniform 1958x2576x3. One device with one framing means
laterality is always read from the same cue. `SEED = 0` and splits are **by
patient**, so the split and the chosen demo files are deterministic — a rebuild
on the dev server should select the same three fixtures.

### Honest caveats about rebuilding

- The retrained model will be **equivalent, not bit-identical**. cuDNN
  nondeterminism means weights differ slightly run to run. Fine for a prototype;
  worth recording once provenance exists (§2), because "which model.ts produced
  this SR" becomes a real question at that point.
- Training is the one step that runs **host-side** rather than in a container.
  That is fine and intended: `train.py` never imports Holoscan, and cu12 torch
  wheels bundle their own CUDA userspace. The host-CUDA-12-vs-MAP-CUDA-13
  conflict in §4 is Holoscan-specific and does not apply here.
- Decoding is ~400 ms/image, so step 2 takes a while on ~2100 images
  (`WORKERS = 12`). It is a one-time cost; the `.npy` cache is what makes
  training 40 s.
- If step 2 finds 0 matching rows, the manifest schema or the dataset path is
  wrong — check `manifest.tsv` columns before touching the filter constants.
- `model.ts` stays gitignored (`*.ts`). Do not commit it.

### Also rebuild the tutorial MAPs, if you want them

`shadow_mode/config.py` registers `simple_imaging` and `mednist_prebuilt` purely
as cheap smoke tests for the runner. They are optional. `simple_imaging` is the
useful one — it needs no model and no GPU, so it isolates "is the executor
working" from "is the model working":

```bash
monai-deploy package Simple_Image_Processing \
  -c Simple_Image_Processing/app.yaml \
  -t simple_imaging_app:1.0 --platform x86_64 -l DEBUG
```

If you skip them, `--list` will still show them but running them will fail with
an image-not-found error from Docker. That is expected, not a bug.

---

## 6. The laterality MAP — contracts and quirks

Image: `airead_laterality_app-x64-workstation-dgpu-linux-amd64:1.0` (8.78 GB).
A ResNet18 TorchScript classifier returning `L` or `R` for a fundus image.

### Standard MAP filesystem contract

```
/var/holoscan/input     input directory (bind mount)
/var/holoscan/output    output directory (bind mount)
/opt/holoscan/models    models baked in at package time
/opt/holoscan/app       app source
/etc/holoscan/app.json  manifest
```

### Input contract — one file per run

`load_dicom_operator.py:55` is the real contract:

```python
input_path = next(input_path.glob("*.dcm"))  # take the first DICOM
```

**One instance per run, and "first" is glob order.** Staging two files makes the
result depend on directory iteration order. The Phase 2 executor already
enforces this; preserve that property in anything you add.

The loader also converts `YBR_FULL_422 -> RGB`. These Topcon fundus images are
stored as YBR and would be silently wrong if treated as RGB. Leave that inside
the MAP; do not reimplement it upstream.

### Test fixture profile

The demo DICOM (`1002_maestro2_3d_macula_cfp_r_...dcm`):

```
Modality              OP
SOPClassUID           1.2.840.10008.5.1.4.1.1.77.1.5.1   (Ophthalmic Photography 8-Bit Image Storage)
Manufacturer          Topcon / 3DOCT-1Maestro2
PatientID             1002
ImageLaterality       R          <- ground truth
PhotometricInterp.    YBR_FULL_422
Rows x Columns        1958 x 2576
```

That SOP Class is your starting eligibility predicate for Phase 5. Ground truth
is also encoded in the filename (`..._cfp_r_...` / `..._cfp_l_...`).

### Output

Two files in `/var/holoscan/output`:

- `output.json` — a bare JSON string, `"R"`. No confidence, no UID. Confidence
  is computed at `laterality_classifier_operator.py:123` and only `print`ed —
  if you need it downstream, that is where to surface it.
- `<generated-uid>.dcm` — Basic Text SR Storage.

### Preprocessing drift hazard

Preprocessing is duplicated in `prepare_data.py`, `train.py`, and the operator's
`preprocess()`. They must agree (224x224 bilinear, ImageNet mean/std). Drift
degrades the model **silently** rather than raising. Also: no horizontal-flip
augmentation, ever — mirroring a fundus image inverts the laterality label and
caps accuracy at chance. Splits are by patient, not image.

`LATERALITY_CLASSES = ["L", "R"]` index order must match `CLASSES` in
`prepare_data.py`. Nothing in `model.ts` records this; disagreement yields a
confident wrong answer, not an error.

---

## 7. RESOLVED 2026-10-16 — the SR was orphaned; the MAP now owns the link

**Current state:** `maps/aireadi_laterality/` is the canonical laterality MAP (SDK loader +
series selector, batched, any DICOM tree in). Its SR writer (`maps/aireadi_laterality/map_ops/`, highdicom
TID 1500) puts each SR in its source's study and references the source image. Verified on a
24-image batch (24/24 correct, all linked) and end to end through MONAI Deploy Express.
Registered in `shadow_mode/config.py` as `aireadi_laterality` (`accepts_tree=True`). Read the
MAP's README; it supersedes the SR-fix plan below. Shadow Mode therefore needs **no**
reference-injection step: upload the MAP's output folder as-is, and use
`map_ops.verify_outputs` as a pre-upload check if wanted.

Two Holoscan packager bugs found on the way (both silent; see `maps/aireadi_laterality/map_ops/README.md` and
`MONAI_Deploy_Express/LOCAL_DOCS/`): `--add` of a flat directory is dropped, and the image
launcher swallows the app's exit code. `shadow_mode`'s executor now detects the latter from the
launcher's `Application exited with N` line.

Original analysis, kept for the reasoning:

Verified against monai-deploy-app-sdk 4.0.0 by reading
`monai/deploy/operators/dicom_utils.py::write_common_modules`.

The MAP's SR is valid Basic Text SR but **has no link to its source image**:

```
PatientID                                    (empty)
PatientName                                  (empty)
StudyInstanceUID      1.2.826.0.1.3680043... <- freshly generated, unrelated
ReferencedSeriesSequence                     ABSENT
CurrentRequestedProcedureEvidenceSequence    ABSENT
```

`app.py:32` sets `copy_tags=False`. **Setting it to `True` does not fix this.**
With `copy_tags=True` the SDK copies patient demographics and reuses the source
`StudyInstanceUID` — and that is all. It always mints a fresh SeriesInstanceUID
and SOPInstanceUID, and at **no** setting does it write
`ReferencedSeriesSequence`, `CurrentRequestedProcedureEvidenceSequence`, or an
SR content item of value type IMAGE. Nothing names the source `SOPInstanceUID`.

Uploading today's output to Orthanc produces a free-floating SR under a phantom
patient.

**Phase 4 therefore needs a reference-injection step inside Shadow Mode**,
applied to the MAP's output before upload. Do not fork the SDK (§2). Note also
that `copy_tags=True` would additionally require wiring a
`study_selected_series_list` input the app does not currently produce.

Second, smaller defect: `ModelInfo("QTIM", "...", "0.1", "resnet18")` in
`app.py:27` puts the string `"resnet18"` into `DeviceUID` (0018,1002), whose VR
is `UI`. pydicom warns; it is not a valid UID and Orthanc may object on upload.

---

## 8. Phase 2 — what is already built and verified

```
shadow_mode/
    config.py                           MAP registry: name -> image, gpu, timeout, suffixes
    execution/executor.py               MapExecutor interface + staging, pre-flight, timeout, collection
    execution/docker_backend.py         `docker run`
    execution/monai_deploy_backend.py   `monai-deploy run`
    run_map.py                          CLI
    README.md                           usage
    HANDOFF.md                          this file
```

### The interface

```python
MapExecutor.run(map_name, input_path, output_path) -> RunResult
```

That is the only contract upper layers may depend on. Backends implement
`_build_command` and nothing else; staging, permission pre-flight, timeouts and
artifact collection live in the base class. A Workflow Manager adapter could be
added later without touching discovery, eligibility, routing or provenance.
`RunResult` carries `artifacts`, `returncode`, `stdout`, `stderr`, `duration_s`,
`timed_out`, `ok`. Failure raises `MapExecutionError`, which carries the
`RunResult`.

The executor knows nothing about ophthalmology. Keep it that way.

### Usage

```bash
python -m shadow_mode.run_map AIREADI_Laterality_App/demo_input/<file>.dcm

--map laterality           # registered name; --list shows them
--backend docker           # or monai-deploy
-o DIR                     # default shadow_mode/runs/<map>-<utc-timestamp>
--gpus none                # CPU only
--bin /path/to/executable  # e.g. monai-deploy outside the pyenv
-v                         # echo MAP stdout/stderr
```

### Verified on the original machine

| Backend | Time | Result |
|---|---|---|
| `docker` | 9.0 s | `R` + SR — correct, matches `ImageLaterality` |
| `monai-deploy` | 7.9 s | `R` + SR — same |

Also exercised: `simple_imaging` MAP on CPU (2.3 s) to prove the runner is not
laterality-shaped; exit codes 0/1; `MapExecutionError` surfacing docker's real
stderr; ambiguous-directory, wrong-suffix, unknown-map and missing-file errors;
and artifact diffing across a reused output directory (3 files on disk, second
run correctly reported only its own 2).

**None of this is verified on your dev server. Re-run it as your first step.**

### Why two backends

`docker` is the default. Passing no `--user` means the container runs as the
image's intended user, which *is* the packaging identity — making the UID/GID
failure mode structurally impossible rather than configured-around. It also
needs no Holoscan CLI in the service environment.

`monai-deploy` is kept because it is the vendor-supported path the app READMEs
document, which makes it the right tool for deciding whether a future failure is
ours or the SDK's. It passes `--uid/--gid` from `MapSpec.package_uid/gid`
(1000:1000), **never** `os.getuid()`.

### Three non-obvious things the executor handles

1. **One input per run** (see §6) — stages exactly one file into a private temp
   dir, refuses an ambiguous input directory rather than choosing for you.
2. **Permissions.** `mkdtemp` yields `0700` owned by the calling user;
   bind-mounted into a container running as uid 1000 that is unreadable. Staging
   widens dir (`0755`) and file (`0644`), and an output-dir pre-flight rejects an
   unwritable target with a clear message instead of failing deep in Holoscan.
3. **Dirty output dirs.** Artifacts are a before/after snapshot diff
   (size + mtime_ns), so reusing a directory reports only the current run's files.

---

## 9. Phase 3 — credentials, `.env`, and what to do next

### The user is supplying a `.env` with Orthanc credentials

This unblocks Phase 3. Expect a `.env` at the **repo root**, not committed.
`.env.example` is committed as the contract:

```bash
ORTHANC_URL=http://localhost:8043
ORTHANC_USERNAME=
ORTHANC_PASSWORD=
ORTHANC_TIMEOUT_S=30
ORTHANC_VERIFY_SSL=true
```

Rules for consuming it:

- **Never commit the real `.env`, and never echo the password** into logs,
  command lines (visible in `ps`), error messages or `RunResult.stdout`. If a
  request fails, report the URL and status code, not the credentials.
- `.gitignore` already excludes `.env`. Keep it that way. If you add new secret
  keys, add them to `.env.example` with empty values so the contract stays
  discoverable.
- Load it in a small `shadow_mode/config.py` addition — environment variables
  win over file values, so a deployment can override without editing files.
  `python-dotenv` is the obvious dependency; plain parsing is also fine and
  avoids adding one.
- Orthanc uses **HTTP Basic** auth. In `requests`, that is
  `auth=(user, password)` on a `Session`, set once.

If the supplied values do not work, say so and stop rather than probing
credentials — see the cautions below.

### What was observed on the original machine

The dashboard Orthanc was container `cu_aai_dashboard-orthanc-1`, image
`orthancteam/orthanc:26.8.2`, REST on **:8043**, DICOM on **:4243**, returning:

```
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Basic realm="Orthanc Secure Area"
```

**Four other Orthanc instances were running on that host** (:8042, :8055 and
others), so confirming *which* instance is the authority matters as much as the
password. On your dev server, re-derive this — `docker ps`, then probe `/system`
— and do not assume port 8043 or that the topology matches.

Once authenticated, these are the useful endpoints:

```
GET  /system                          sanity check + version
GET  /statistics                      how much is actually in there
GET  /instances/<id>/file             download the raw DICOM       <- Phase 3
GET  /instances/<id>/simplified-tags  metadata without downloading pixels
POST /instances                       upload a DICOM               <- Phase 4
```

`/instances/<id>/simplified-tags` is the one to build eligibility on later: it
answers "is this an OP image or an AI-generated SR" without transferring pixel
data, which matters a lot across a 66k-image backfill.

### Cautions specific to a dev server

- Confirm with the user whether the dev Orthanc holds **real patient data**
  before writing anything into it or copying instances out. Phase 4 writes new
  objects into Orthanc; do that against a disposable instance first if there is
  any doubt.
- Read-only exploration (`/system`, `/statistics`, `/instances/<id>/...`) is
  safe. `POST /instances` and anything under `/modalities` is not — it mutates
  shared state. Ask before the first write.

There is **zero Orthanc code in the repo** — `grep -ri orthanc` returns nothing.
Phase 3 is greenfield.

### Intended Phase 3 shape

```
shadow_mode/orthanc/client.py     query, download instance, upload, fetch metadata
```

Responsibilities: talk to the Orthanc REST API and nothing else. It must **not**
decide which models run. Then a thin `run_from_orthanc.py <instance-id>` that
downloads to a temp file and hands it to the existing `MapExecutor`. Manual
invocation is fine at this stage.

Later, alongside it (but not yet):
```
shadow_mode/orthanc/eligibility.py    inspect metadata, accept/reject, choose workflow
shadow_mode/workflows/laterality.py   download -> execute -> upload
shadow_mode/provenance/store.py       idempotency + model/run records
```

---

## 10. Working with this user

Experienced engineer — Python, Docker, Linux, DICOM, MONAI Bundles. Skip
fundamentals. They typically arrive with the issue already diagnosed and expect
that diagnosis respected rather than re-litigated. They value working vertical
slices over infrastructure, and clear interfaces over framework complexity.

Be concrete about what is verified versus assumed. When something is blocked,
finish everything that is not blocked and say plainly what was left out.

---

## 11. Dev-server bootstrap checklist

Work top to bottom. Everything is rebuilt locally — nothing is transferred from
the original machine.

```
ENVIRONMENT
[ ] Clone the repo. Confirm shadow_mode/ is present.
[ ] ls /persist/python_virtual_environments/bearceb/.pyenv/versions/3.10.0/envs/monai_deploy
        Shared mount — expect it to exist. Activate it.
[ ] python -c "import monai.deploy; print(monai.deploy.__version__)"   -> 4.0.0
[ ] monai-deploy --version                     -> holoscan-cli 4.2.0
[ ] nvidia-smi                                 -> GPU present? driver version?
[ ] docker info >/dev/null                     -> daemon reachable, you can use it

BUILD (§5) — re-derive, do not docker load
[ ] export AIREADI_ROOT=/persist/public_datasets/ai_readi
[ ] ls "$AIREADI_ROOT/retinal_photography/manifest.tsv"
[ ] cd AIREADI_Laterality_App && python prepare_data.py && cd -
        -> cache/*.npy AND demo_input/*.dcm (3 held-out test fixtures)
[ ] cd AIREADI_Laterality_App && python train.py && cd -
        -> models/model/model.ts, ~40 s on one A4000
[ ] monai-deploy package AIREADI_Laterality_App -c .../app.yaml -m .../models \
        -t airead_laterality_app:1.0 --platform x86_64 -l DEBUG
[ ] docker images | grep airead_laterality     -> tag matches shadow_mode/config.py
[ ] docker run --rm --entrypoint id <image>    -> expect uid=1000(holoscan)
[ ] (optional) package Simple_Image_Processing as a no-GPU smoke test

VERIFY PHASE 2
[ ] python -m shadow_mode.run_map --list
[ ] python -m shadow_mode.run_map AIREADI_Laterality_App/demo_input/<file>.dcm --backend docker
[ ] python -m shadow_mode.run_map AIREADI_Laterality_App/demo_input/<file>.dcm \
        --backend monai-deploy --bin <path-to-monai-deploy>
        Expect both: ~8-10 s, 2 artifacts, prediction == the fixture's ImageLaterality
        (also encoded in the filename as _cfp_l_ / _cfp_r_).

PHASE 3 PREREQS
[ ] cp .env.example .env, and get the real values from the user (§9).
[ ] docker ps                                  -> find the dashboard Orthanc HERE
[ ] curl -u "$ORTHANC_USERNAME:$ORTHANC_PASSWORD" <url>/system
[ ] Confirm with the user whether this Orthanc holds real patient data.
[ ] Only then start Phase 3. Read-only first; ask before any write.
```

### Triage

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'monai'` inside the MAP | UID/GID invariant (§4), **not** a missing dependency |
| `Invalid SDK version specified (4.5.0)` when packaging | Holoscan version drift — see the pins in §4 |
| Permission error on the staged input or output dir | §8, third item |
| MAP exits 0 but writes nothing | loader found no input it recognised — check the suffix and that exactly one file was staged |
| `prepare_data.py` reports 0 matching rows | wrong dataset path, or the manifest schema changed — inspect `manifest.tsv` columns before editing the filter constants |
| Docker "image not found" for `simple_imaging` / `mednist_prebuilt` | those tutorial MAPs were never packaged here; optional (§5) |
