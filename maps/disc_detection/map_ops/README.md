# map_ops — model-agnostic operators (disc_detection)

Model-agnostic pieces that make a MAP follow one rule:

> **Input is any DICOM tree. Output is DICOM only, and every output object sits in its
> source's study and references the source instances it was derived from.**

Copied from `maps/aireadi_laterality/map_ops/` (the classifier swapped for a detector) and extended so
an SR can carry **boxes**: `BoxFinding` and `not_found` in `image_finding.py`, written by `sr_builder.py`
as TID 1410 Planar ROIs (closed `POLYLINE` rectangles on the source image) or "Presence of property = Absent".
Laterality-style findings (coded value + measurement) are still written exactly as before.

| Module | What | Holoscan? |
|---|---|---|
| `image_finding.py` | `ImageFinding` (+ `BoxFinding`, `not_found`): the contract between the model operator and the writer | no |
| `sr_builder.py` | ImageFinding → TID 1500 Comprehensive SR via highdicom; deterministic UIDs | no |
| `finding_sr_writer_operator.py` | operator wrapping `sr_builder`: one SR per image into the output folder | yes |
| `instance_detector_operator.py` | per-instance detector driven by `model_spec` (`detect()`); refuses weights whose SHA-256 isn't `MODEL_SHA256` | yes |
| `verify_outputs.py` | CLI: every output is DICOM and references an input image (prints boxes for SRs) | no |

```bash
PYTHONPATH=maps/disc_detection python -m map_ops.verify_outputs INPUT_DIR OUTPUT_DIR
```

`map_ops/` sits inside the MAP folder, so `monai-deploy package` copies it with the app. Why our own SR
writer rather than the SDK's `DICOMTextSRWriterOperator` (it never references the source image):
`../../aireadi_laterality/map_ops/README.md`.

Codes: real ones from `pydicom.sr.codedict.codes` (SCT 81016008 optic disc, SCT 67046006 fovea, DCM 111012
Certainty of Finding); private scheme `99QTIM` only where no standard code exists.
