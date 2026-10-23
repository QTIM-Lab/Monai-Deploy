---
name: monai-deploy
description: Use when working on MONAI Deploy in this repo — building or changing MAPs in maps/, packaging with `monai-deploy package`, running MAPs, operating the MONAI Deploy Express stack (Orthanc, Informatics Gateway, Workflow Manager, workflows, routing), debugging MAP import errors or Holoscan/CUDA conflicts, or handing a model over to ML engineers. Triggers on "MONAI Deploy", "MAP", "holoscan", "monai-deploy package/run", "app.yaml", "model_spec", "Express", "workflow", "Informatics Gateway", "laterality".
---

# MONAI Deploy in this repo

Orthanc → MONAI Deploy Express → our MAPs → DICOM SR/SEG linked to the source image, back in Orthanc.
**Stay canonical**: do things the way MONAI Deploy intends; minimal, clearly-labelled custom code. **No Kubernetes.**

## Where things are (read these, don't duplicate them)

| Path | What |
|---|---|
| `maps/README.md` | MAP contract, anatomy, adding a MAP |
| `maps/aireadi_laterality/` | reference MAP; self-contained (`map_ops/` inside); `model_spec.py` is the only model-specific file |
| `MONAI_Deploy_Express/LOCAL_SETUP.md` | one-page control sheet; detail in `LOCAL_DOCS/<service>.md` |
| `AIREADI_Laterality_App/` | training code + `models/model/model.ts` (gitignored); tutorial MAP |
| `shadow_mode/` | parked custom orchestrator; `python -m shadow_mode.run_map` still runs a MAP on local files |
| Simple_Image_Processing, MedNIST_* | App SDK tutorials |

## Environment

- Python: `/persist/python_virtual_environments/bearceb/.pyenv/versions/3.10.0/envs/monai_deploy` (`/persist` is
  shared across servers). Bare `python` is an unrelated 3.9. MAP runtime inside images is Python 3.12.
- Pinned: `monai-deploy-app-sdk 4.0.0`, `holoscan-cli 4.2.0`, `holoscan-cu13 4.2.0`. Newer Holoscan breaks
  packaging (`Invalid SDK version specified`). Don't change pins without agreement.
- Host CUDA 12, MAPs CUDA 13: run MAPs in containers, never `python app.py` on the host. Don't touch host CUDA
  (shared server). Host-side torch training is fine.
- Rebuild MAP images on each host from source; don't `docker save`/transfer them.

## Traps that look like something else

- `ModuleNotFoundError: monai` inside a MAP = **UID/GID mismatch**, not a missing package. Run as the image's own
  user (`docker run` without `--user`, or `monai-deploy run --uid 1000 --gid 1000`).
- A crashed MAP **exits 0** under the image launcher (`/var/holoscan/tools` drops the code). Check the log for
  `Application exited with N`; workflows run `python3 -u /opt/holoscan/app` directly.
- `monai-deploy package --add <flat dir>` is silently dropped (add the parent). Our MAPs don't need `--add`.
- Holoscan `Operator` has a `spec` property: don't assign `self.spec`.
- The SDK text SR writer never references the source image (`copy_tags` only copies patient/study). Use the MAP's
  `map_ops` SR writer (highdicom `evidence=`); SDK SEG writer is already linked.
- pydicom 3 converts YBR→RGB in `pixel_array`; the current laterality model was trained with a second conversion,
  reproduced on purpose in `model_spec.py` until a retrain.
- Express: overlapping runs of one workflow delay an export ~30 min (upstream IG PR #544 unmerged); the Task
  Manager doesn't serialize jobs and leaves exited containers; the IG exports as `MONAISCU`, listens as
  `MONAI-DEPLOY`. MongoDB stores PascalCase + enum ints; the APIs return snake_case.

## Working style

The user is an experienced engineer, new to parts of MONAI Deploy and still building a mental model: explain
architecture choices briefly, keep docs and answers concise, respect diagnoses already made, and don't edit
upstream files (local additions are named `LOCAL_*`).
