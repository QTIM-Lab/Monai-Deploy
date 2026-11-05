# Fundus Vessels MAP (VascX)

A MONAI Application Package that segments the retinal blood vessels in colour fundus
photographs, and writes one DICOM SEG per image that **references the image it
segmented** and lives in that image's study.

Research use only. The model is third-party: VascX (Erasmus MC), AGPL-3.0. Choosing and checking the model happens in [`VascX_App/`](../../VascX_App/) (including the comparison with the paper's model); this MAP runs what the App decided, and its only own concern is delivering linked, idempotent results.

## Contract

| | |
|---|---|
| **Input** | Any DICOM tree at `/var/holoscan/input`: one image or a batch, nested directories, any mix of modalities. Non-DICOM files are ignored. |
| **Selected** | Series with `Modality == OP` and `ImageType` containing `COLOR` (`model_spec.SELECTION_RULES`), any camera. Within them, single-frame colour instances no wider than 1.6:1 (mosaics are skipped: the fundus crop would cover only one field). Everything else is skipped, not an error; nothing selected finishes with no output. |
| **Output** | DICOM only, at `/var/holoscan/output`: one binary Segmentation (SEG) per image, on the source's own pixel grid. Safe to C-STORE the whole folder. |
| **Each SEG** | Same Patient and Study as its source; one AI series per study per model version; references the source SOP; UIDs derived from the source and version, so re-runs reproduce the same files (see `map_ops/seg_builder.py`); one segment "Retinal blood vessels" (SCT 59820001, region Retina SCT 5665001); algorithm name and version in the segment and in `SeriesDescription` (`AI vascx-vessels 0.2.0 - research use only`). |
| **Failure** | A bad instance is logged and skipped. The run fails only if instances were selected and none could be segmented. |

## Layout

```
maps/fundus_vessels/
    model_spec.py        THE model-specific file: version, segment codes, inference, selection
    app.py               pipeline wiring; no model knowledge
    app.yaml             MAP manifest and resources
    requirements.txt     baked into the image; pydicom/highdicom/fundusprep pinned on purpose
    tools/
        check_idempotency.py   re-runs reproduce the same SEGs; runs inside the image, on data in Orthanc
    map_ops/             model-agnostic: per-image segmenter, SEG writer, output checker
```

Same pipeline shape as [`aireadi_laterality`](../aireadi_laterality/), with a segmenter and a SEG
writer in place of the classifier and SR writer.

**Why our own SEG writer**, not the SDK's `DICOMSegmentationWriterOperator`: in SDK 4.0.0 it
writes one SEG per run (first series of the first study), and highdicom 0.28, which it uses,
fails on Ophthalmic Photography sources. Details and the workaround: `map_ops/seg_builder.py`.

## What the model does (model_spec.py)

1. Decode to RGB (pydicom 3; no extra colour conversion).
2. Find the fundus circle and warp it to 1024×1024: upstream's own `retinalysis-fundusprep`.
3. ImageNet-normalise; run the model on the image and its 3 flips; average logits; softmax.
4. Warp the vessel probability back to the original image; threshold at 0.5.

Copied from the App's prototype `VascX_App/vessels.py` (same functions), where
`compare_paper_model.py` shows it matches the authors' own code and the paper's model.

## Build

From the repo root, in the `monai_deploy` pyenv:

```bash
bash VascX_App/fetch_model.sh          # weights, sha256-checked; not in git
monai-deploy package maps/fundus_vessels \
  -c maps/fundus_vessels/app.yaml \
  -m VascX_App/models \
  -t fundus_vessels:0.2.0 \
  --platform x86_64 -l INFO
# -> fundus_vessels-x64-workstation-dgpu-linux-amd64:0.2.0
```

Keep the tag in step with `ALGORITHM_VERSION`.

## Test

Tests run on data already in Orthanc; nothing assumes a dataset path on this machine.

```bash
# 1. Load data (once): any folder of DICOM, e.g. a few AI-READI patients
MONAI_Deploy_Express/LOCAL_SCRIPTS/upload.sh /path/to/dicom

# 2. Run the MAP on a patient through MONAI Deploy Express, as in production
python3 MONAI_Deploy_Express/LOCAL_SCRIPTS/send.py --patient 1001 --ae VESSELS --skip-done "vascx-vessels 0.2.0"
#    In Orthanc: one SEG per colour photo, in its study; infrared, OCT and mosaics get none.
#    Send it again without --skip-done: the SEG count must not change (same UIDs;
#    Orthanc keeps what it has). A new ALGORITHM_VERSION adds a new series next to the old one.

# 3. After changing model_spec.py or map_ops/seg_builder.py: the idempotency check, inside the image
docker run --rm --gpus device=0 --network host --entrypoint python3 \
  fundus_vessels-x64-workstation-dgpu-linux-amd64:0.2.0 \
  /opt/holoscan/app/tools/check_idempotency.py --patient 1001
```

Any change to the model, inference or output format needs an `ALGORITHM_VERSION` bump; otherwise
the same UIDs get different content. The image's launcher exits 0 even when the app crashes: read
the log for `Application exited with N`. Run the container as its own user (no `--user`).

## Known limitation: Optomed Aurora photos are dropped by the SDK loader

AI-READI's Aurora (handheld) photos have an empty `SeriesNumber`. The SDK's
`DICOMDataLoaderOperator` (4.0.0) calls `int(None)` on it and discards the instance with only a
warning (`Error parsing SOP Instance: int() argument ... 'NoneType'`), so they never reach the
model. This affects every MAP using the SDK loader, `aireadi_laterality` included. Not fixed here.

## Swapping in another vessel model (e.g. our SegFormer)

Everything is in `model_spec.py`: bump `ALGORITHM_VERSION`, rewrite `segment()` (it must return
a `(Rows, Columns)` uint8 mask on the source grid), point `MODEL_NAME` at the new models folder,
after checking the new model in its own App folder against the one it replaces.
`app.py` and `map_ops/` should not need to change.
