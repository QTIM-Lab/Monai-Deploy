# map_ops — model-agnostic operators (segmentation)

The segmentation counterpart of [`aireadi_laterality/map_ops`](../../aireadi_laterality/map_ops/README.md),
following the same rule: **input is any DICOM tree; output is DICOM only, and every output
object sits in its source's study and references the source instance it was derived from.**

| Module | What | Holoscan? |
|---|---|---|
| `instance_segmenter_operator.py` | per-instance 2D segmenter, driven by a `model_spec` | yes |
| `seg_builder.py` | (source, mask) → binary DICOM SEG via highdicom, one per image | no |
| `seg_writer_operator.py` | operator wrapping `seg_builder` | yes |

## Why not the SDK's DICOMSegmentationWriterOperator

In monai-deploy-app-sdk 4.0.0:

1. **One SEG per run.** `process_images` takes the first selected series of the first study
   and `break`s. A batch of photos would get one SEG, and the rest would be silently dropped.
2. **Fails on fundus photos.** highdicom 0.28 (latest) treats Ophthalmic Photography as an enhanced
   multi-frame image (its IOD allows `NumberOfFrames`) and reads pixel spacing from
   `SharedFunctionalGroupsSequence`, which OP images don't have → `AttributeError`.
   `seg_builder._as_highdicom_source` hands highdicom a header copy with that group filled in;
   the SEG still references the real source UIDs.
3. **Random UIDs.** Every re-run would add a duplicate SEG. Ours derives the SOP UID from (source SOP UID,
   algorithm, version) and the series UID from (study UID, algorithm, version), with no timestamps, so a
   re-run reproduces byte-identical files, which Orthanc (OverwriteInstances: false) ignores as already stored
   (`seg_builder.py` docstring).

All three are reasonable upstream issues (SDK and highdicom). Until then this lives here, visibly.
For a volume (CT/MR) segmentation MAP, the SDK writer is still the right choice.
