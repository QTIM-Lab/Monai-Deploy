# Disc + Fovea Detection MAP (RetinaNet)

A MONAI Application Package that finds the optic disc and the fovea in colour fundus photographs and
writes one DICOM SR per image. Each box is a **rectangle drawn on the source image** (coordinates in its
own pixels), so a viewer can overlay it, and a program can read it without touching pixels.

Research use only. The model, its labels and its evaluation live in [`Disc_Detection_App/`](../../Disc_Detection_App/):
torchvision RetinaNet trained on AI-READI with **VascX-made labels, not clinician labels**; the fovea box
is a 1 disc-diameter square around VascX's fovea point, so judge the fovea by the box centre. This MAP
runs what the App decided; its own concern is linked, idempotent results.

## Contract

| | |
|---|---|
| **Input** | Any DICOM tree at `/var/holoscan/input`: one image or a batch, nested directories, any mix of modalities. Non-DICOM files are ignored. |
| **Selected** | Series with `Modality == OP` and `ImageType` containing `COLOR` (`model_spec.SELECTION_RULES`), any camera. Within them, single-frame colour instances no wider than 1.6:1 (mosaics skipped). Everything else is skipped, not an error. |
| **Output** | DICOM only, at `/var/holoscan/output`: one TID 1500 Comprehensive SR per image. Safe to C-STORE the whole folder. |
| **Each SR** | Same Patient and Study as its source; one AI series per study per model version; references the source SOP; UIDs derived from the source and version (re-runs reproduce the same files); `SeriesDescription` `AI disc-fovea-detection 0.1.0 - research use only`. |
| **Per structure** | Found: a TID 1410 Planar ROI group, Finding = *Optic disc structure* (SCT 81016008) or *Structure of fovea centralis* (SCT 67046006), Image Region = closed `POLYLINE` rectangle (x0,y0 → x1,y1, DICOM pixel convention) on the source image, *Certainty of Finding* (DCM 111012) = the detector's score, 0–1, not a calibrated probability. Not found: a group with that Finding and *Presence of property = Absent* (e.g. the disc in a temporal peripheral field). |
| **Failure** | A bad instance is logged and skipped. The run fails only if instances were selected and none could be processed. |

The top-scoring box per structure is reported whatever its score (torchvision keeps ≥ 0.05). Consumers
choose their own cut-off; low scores mark hazy or unusual photos (see the App's hand check).

## What the model does (model_spec.py)

1. Decode to RGB (pydicom 3; no extra colour conversion), as training's `prepare_cache.py`.
2. Find the fundus circle and warp it to 512×512: upstream's `retinalysis-fundusprep`, as in training.
3. RetinaNet on the [0, 1] crop (it normalises inside); top-scoring box per class.
4. Map the box back through the crop's inverse transform (scale + shift, so it stays a rectangle);
   round to 0.1 px and the score to 4 places, so re-runs are identical.

Copied from the App prototype `Disc_Detection_App/detector.py`. The model file is the TorchScript export
`Disc_Detection_App/export_model.py` writes; `MODEL_SHA256` pins it.

## Layout

```
maps/disc_detection/
    model_spec.py        THE model-specific file: version, codes, inference, selection
    app.py               pipeline wiring; no model knowledge
    app.yaml             MAP manifest and resources
    requirements.txt     baked into the image; torch/torchvision/pydicom/highdicom/fundusprep pinned
    tools/
        check_idempotency.py   re-runs reproduce the same SRs; runs inside the image, on data in Orthanc
    map_ops/             model-agnostic: per-image detector, SR writer (now with boxes), output checker
```

`map_ops/` is `aireadi_laterality`'s, plus `BoxFinding` / `not_found` in `image_finding.py` and the
matching groups in `sr_builder.py`; the laterality-style output is unchanged.

## Build

From the repo root, in the `monai_deploy` pyenv:

```bash
python Disc_Detection_App/export_model.py   # in the vascx env: best.pt -> Disc_Detection_App/export/retinanet/*.pt; prints the SHA-256
monai-deploy package maps/disc_detection \
  -c maps/disc_detection/app.yaml \
  -m Disc_Detection_App/export \
  -t disc_detection:0.1.0 \
  --platform x86_64 -l INFO
# -> disc_detection-x64-workstation-dgpu-linux-amd64:0.1.0
```

`-m` must hold only model folders with one file each (the App SDK's rule). Keep the tag in step with
`ALGORITHM_VERSION`.

## Test

Tests run on data already in Orthanc.

```bash
# 1. Load data (once), e.g. an AI-READI participant
MONAI_Deploy_Express/LOCAL_SCRIPTS/upload.sh /path/to/dicom

# 2. Run through MONAI Deploy Express, as in production
python3 MONAI_Deploy_Express/LOCAL_SCRIPTS/send.py --patient 1019 --ae DISC-FOVEA --skip-done "disc-fovea-detection 0.1.0"
#    In Orthanc: one SR per colour photo, in its study; mosaics, OCT and infrared get none.

# 3. Look at it: the App's hand-check sheet draws the SR's boxes (magenta) over its own (green/cyan)
python Disc_Detection_App/handcheck.py --patient 1019     # vascx env; reports the SR-vs-App difference in px

# 4. After changing model_spec.py or map_ops/sr_builder.py: the idempotency check, inside the image
docker run --rm --gpus device=0 --network host --entrypoint python3 \
  disc_detection-x64-workstation-dgpu-linux-amd64:0.1.0 \
  /opt/holoscan/app/tools/check_idempotency.py --patient 1019
```

Any change to the model, inference or output format needs an `ALGORITHM_VERSION` bump. The image's
launcher exits 0 even when the app crashes: read the log for `Application exited with N`.

Same SDK loader limitation as the other MAPs: Optomed Aurora photos with an empty `SeriesNumber` are
dropped by `DICOMDataLoaderOperator` (see `../fundus_vessels/README.md`).
