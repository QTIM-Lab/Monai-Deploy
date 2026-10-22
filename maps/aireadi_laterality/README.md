# AI-READI Laterality MAP

A MONAI Application Package that classifies colour fundus photographs as left or
right eye, and writes one DICOM SR per image that **references the image it scored**
and lives in that image's study.

Research use only. The model is ours (a ResNet18 trained in
`AIREADI_Laterality_App/`), not a MONAI model.

## Contract

| | |
|---|---|
| **Input** | Any DICOM tree at `/var/holoscan/input`: one image or a batch, nested directories, any mix of modalities. Non-DICOM files are ignored. |
| **Selected** | Series with `Modality == OP` (`model_spec.SELECTION_RULES`). Within them, single-frame colour instances. Everything else is skipped, not an error. |
| **Output** | DICOM only, at `/var/holoscan/output`: one TID 1500 Comprehensive SR per scored image. Safe to C-STORE the whole folder. |
| **Each SR** | Same Patient and Study as its source; new AI series (one per study per run); evidence and Image Library reference the source SOP; `Laterality = Left/Right` (SCT 272741003 / 7771000 / 24028007); `Probability` (DCM 122157); algorithm name and version (TID 4019). |
| **Failure** | A bad instance is logged and skipped. The run fails only if instances were selected and none could be scored. |

## Layout

```
maps/aireadi_laterality/
    model_spec.py        THE model-specific file: version, classes, codes, preprocessing, selection
    app.py               pipeline wiring; no model knowledge
    app.yaml             MAP manifest and resources
    requirements.txt     baked into the image; pydicom/highdicom pinned on purpose
    tools/
        check_preprocessing.py   model_spec vs the training cache, byte for byte
        make_test_batch.py       held-out patients + distractors, in Task Manager layout
        check_batch.py           links valid + accuracy vs ImageLaterality
    map_ops/             model-agnostic operators: SR writer, batched classifier, output checker
```

The pipeline is upstream's canonical one (`DICOMDataLoaderOperator` →
`DICOMSeriesSelectorOperator` → model → SR writer, as in the SDK's
`breast_density_classifier_app`). Two pieces come from `map_ops/` instead of the SDK,
both forced: the SDK's volume operator is monochrome-only, and its text SR writer does
not reference the source image. See `map_ops/README.md`.

## Handing over a new model: what to change

Everything is in `model_spec.py`. In one commit:

1. Put the new `model.ts` under a models dir as `models/model/model.ts`.
2. Bump `ALGORITHM_VERSION`. It is stamped into every SR and is the only way to tell
   results from different models apart later.
3. Check `CLASSES` is in the model's output index order. Nothing in `model.ts` records
   it; a mismatch produces confident wrong answers, not an error.
4. Make `to_rgb` / `resize` / `preprocess` reproduce your training pipeline exactly,
   then prove it: `tools/check_preprocessing.py` (adapt it to your cache).
5. Narrow `SELECTION_RULES` if the model is only valid on some devices or views.

You should not need to touch `app.py` or `map_ops/`. If you do, that is worth a
conversation: it usually means the contract above is missing something.

### Known defect in the current model: double colour conversion

pydicom 3 already returns RGB from `pixel_array` for YBR images. `prepare_data.py`
converts YBR→RGB a second time (its comment describes pydicom 2), so the current
model was trained on miscoloured images. Verified bit-for-bit against the training
cache. Inference reproduces it on purpose
(`REPRODUCE_TRAINING_DOUBLE_YBR_CONVERSION = True`), because fixing only one side
would silently degrade the model. **Fix it at the next retrain:** correct
`prepare_data.py`, set the flag `False`, bump `ALGORITHM_VERSION`, rerun
`check_preprocessing.py`. Until then the MAP is pinned to pydicom 3.x.

## Build

From the repo root, in the `monai_deploy` pyenv. The model is not in git
(`*.ts` is ignored); `AIREADI_Laterality_App/train.py` produces it.

```bash
monai-deploy package maps/aireadi_laterality \
  -c maps/aireadi_laterality/app.yaml \
  -m AIREADI_Laterality_App/models \
  -t aireadi_laterality:0.1.0 \
  --platform x86_64 -l INFO
# -> aireadi_laterality-x64-workstation-dgpu-linux-amd64:0.1.0
```

Keep the tag in step with `ALGORITHM_VERSION`.

## Test

```bash
# 1. Preprocessing still matches training (host, no GPU, ~10 s)
python maps/aireadi_laterality/tools/check_preprocessing.py

# 2. A batch: test-split patients + an OCT that must be skipped + non-DICOM noise
python maps/aireadi_laterality/tools/make_test_batch.py /tmp/lat_batch --patients 10

# 3. Run the MAP on it
mkdir -p /tmp/lat_out && chmod 777 /tmp/lat_out
docker run --rm --gpus all --ulimit stack=33554432 \
  -v /tmp/lat_batch:/var/holoscan/input:ro -v /tmp/lat_out:/var/holoscan/output \
  aireadi_laterality-x64-workstation-dgpu-linux-amd64:0.1.0
#   or: python -m shadow_mode.run_map /tmp/lat_batch --map aireadi_laterality

# 4. Every SR linked, every eligible image scored once, nothing else scored
python maps/aireadi_laterality/tools/check_batch.py /tmp/lat_batch /tmp/lat_out
```

`PYTHONPATH=maps/aireadi_laterality python -m map_ops.verify_outputs IN OUT` checks only the linking rule and works for
any MAP, SEG included.

Run the container as its own user (no `--user`), or `monai-deploy run --uid 1000
--gid 1000`. Any other identity drops the packaged site-packages off `sys.path` and
every import fails with `ModuleNotFoundError` (see `shadow_mode/HANDOFF.md` section 4).

## In MONAI Deploy Express

The workflow needs no flattening step and no SR fix-up task: a `router` on
`Modality == OP`, this MAP with `HOLOSCAN_INPUT_PATH` pointed at the staged input,
then `export` of its output folder. See `MONAI_Deploy_Express/workflows/`.
