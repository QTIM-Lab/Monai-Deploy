# map_ops — model-agnostic operators

Model-agnostic pieces that make a MAP follow one rule:

> **Input is any DICOM tree. Output is DICOM only, and every output object sits in its
> source's study and references the source instances it was derived from.**

Then the MAP owns the link between a result and its image. Nothing downstream (a
workflow task, an SR "linker", Shadow Mode) has to repair results after the fact,
and any workflow can export a MAP's output folder unchanged.

| Module | What | Holoscan? |
|---|---|---|
| `image_finding.py` | `ImageFinding`: the contract between a model operator and the writer | no |
| `sr_builder.py` | ImageFinding → TID 1500 Comprehensive SR via highdicom | no |
| `finding_sr_writer_operator.py` | operator wrapping `sr_builder` | yes |
| `instance_classifier_operator.py` | batched per-instance 2D classifier, driven by a `model_spec` | yes |
| `verify_outputs.py` | CLI: check any MAP's output against the rule (SR and SEG) | no |

## Which writer for which result

| Result | Writer | Linking |
|---|---|---|
| Pixel segmentation | SDK `DICOMSegmentationWriterOperator` | already right: `highdicom.seg.Segmentation(source_images=...)` |
| Classification / scalar | `ImageFindingSRWriterOperator` (here) | evidence + Image Library + Source Image per measurement group |
| Free text | SDK `DICOMTextSRWriterOperator` | **not linked**: `copy_tags=True` copies patient/study only, never the image |

Why not the SDK text SR writer: in monai-deploy-app-sdk 4.0.0,
`dicom_utils.write_common_modules` copies Patient and Study when `copy_tags=True`, and
mints new Series/SOP UIDs, but never writes `CurrentRequestedProcedureEvidenceSequence`
or an IMAGE content item. Nothing records which image was scored. highdicom (already
an SDK dependency, and what the SDK's SEG writer uses) does this with `evidence=`.
Adding evidence references to the SDK writer is a reasonable upstream contribution;
until then this lives here, visibly, rather than in a patched SDK.

## Using it in a MAP

`map_ops/` sits inside the MAP folder, so the packager copies it with the app; no extra flags. In
`app.py`:

```python
from map_ops.instance_classifier_operator import InstanceClassifierOperator
from map_ops.finding_sr_writer_operator import ImageFindingSRWriterOperator
```

For local use outside a MAP: `PYTHONPATH=maps/aireadi_laterality`.

Sharing it between MAPs later via `monai-deploy package --add`: add the *parent* of `map_ops/`.
Holoscan CLI 4.2 silently drops a flat folder of .py files (packaging still says Succeeded).

A new classification MAP is a copy of `maps/aireadi_laterality/` with a new `model_spec.py`. The spec protocol is in
`instance_classifier_operator.py`'s docstring.

## Codes

Use real codes from pydicom's dictionary (`pydicom.sr.codedict.codes`). Where none
exists, use a private scheme designator starting with `99` (we use `99QTIM`). Do not
invent SNOMED CT values.
