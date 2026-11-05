# maps — our MONAI Application Packages

Deployable models. Each subfolder is one self-contained MAP built with the MONAI Deploy App SDK. Training code and model weights live elsewhere (e.g. `AIREADI_Laterality_App/`);
a MAP only *uses* a trained model.

| MAP | Input | Output | Image |
|---|---|---|---|
| [aireadi_laterality](aireadi_laterality/) | colour fundus photos (`Modality OP`, `ImageType` contains `COLOR`) | TID 1500 SR per image: Left / Right + probability | `aireadi_laterality-x64-workstation-dgpu-linux-amd64:0.2.0` |
| [fundus_vessels](fundus_vessels/) | colour fundus photos (`Modality OP`, `ImageType` contains `COLOR`) | binary SEG per image: retinal vessels (VascX) | `fundus_vessels-x64-workstation-dgpu-linux-amd64:0.2.0` |

## The contract every MAP here follows

1. **Input:** any DICOM tree at `/var/holoscan/input` (one image, a study, a batch, nested folders,
   non-DICOM files). The SDK `DICOMDataLoaderOperator` loads it; `DICOMSeriesSelectorOperator` keeps what
   the model accepts. Anything else is skipped, not an error.
2. **Output:** DICOM only at `/var/holoscan/output`, so a workflow can export the folder unchanged.
3. **Linked:** every output (SR or SEG) sits in its source's study and references the source images.
   The MAP owns this; nothing downstream repairs results.
4. **Versioned:** algorithm name and version are stamped into every result.
5. **Deterministic:** result UIDs derive from (source image, algorithm, version), with no clock or random
   values, so re-runs never duplicate and a new version attaches alongside.

Check any MAP's output against 2 and 3: `PYTHONPATH=maps/aireadi_laterality python -m map_ops.verify_outputs IN OUT` (works for SR and SEG).

## Anatomy of a MAP

```
maps/<name>/
    app.py            wiring of SDK + map_ops operators; no model knowledge
    model_spec.py     everything model-specific: version, classes and codes, preprocessing, selection
    app.yaml          manifest and resources
    requirements.txt  baked into the image
    README.md         contract, build, test, how to hand over a new model
    map_ops/          model-agnostic operators (SR writer, batched classifier, output checker)
    tools/            optional checks (preprocessing parity, batch scoring)
```

Which writer: `map_ops` `ImageFindingSRWriterOperator` for classifications and scalars. For
segmentations the SDK `DICOMSegmentationWriterOperator` links to the source but writes only one SEG per
run (first series, then stops) and fails on OP 8-bit photos in highdicom 0.28; `maps/fundus_vessels/`
carries its own per-image SEG writer for that. Details: [aireadi_laterality/map_ops/README.md](aireadi_laterality/map_ops/README.md).

## Adding a MAP

1. Copy `aireadi_laterality/` (with its `map_ops/`) to `maps/<name>/`; rewrite `model_spec.py` (and `app.py` only if the
   pipeline shape differs, e.g. a volume model using the SDK's `DICOMSeriesToVolumeOperator`).
2. Package, from the repo root, in the `monai_deploy` pyenv:
   ```bash
   monai-deploy package maps/<name> -c maps/<name>/app.yaml -m <models dir> -t <name>:<version> --platform x86_64
   ```
   Everything in the folder (including `map_ops/`) goes into the image.
3. Test on files: `docker run --rm --gpus all -v <in>:/var/holoscan/input:ro -v <out>:/var/holoscan/output <image>`,
   then `verify_outputs`. The image's launcher hides crashes (exits 0); read the log for
   `Application exited with N`.
4. Wire it into MONAI Deploy Express with a workflow: see `../MONAI_Deploy_Express/LOCAL_SETUP.md`.
