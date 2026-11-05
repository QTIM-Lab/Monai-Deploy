"""Model-agnostic operators for 2D segmentation MAPs.

Lives inside the MAP's folder, so `monai-deploy package` copies it into the image with
the app. To reuse it in another MAP, copy the folder.

Same rule as maps/aireadi_laterality/map_ops: input is any DICOM tree; output is DICOM
only, and every output object sits in its source's study and references the source
instance it was derived from. See README.md here for why the SEG writer is ours.

`seg_builder` imports no Holoscan, so it can be used and tested outside a MAP. To check a
run's output folder, use maps/aireadi_laterality's map_ops.verify_outputs (it handles SEG too).
The operator modules import the SDK; import them explicitly.
"""
