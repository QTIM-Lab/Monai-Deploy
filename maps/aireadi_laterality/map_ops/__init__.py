"""Model-agnostic operators shared by our MAPs.

Lives inside the MAP's folder, so `monai-deploy package` copies it into the image with
the app. To reuse it in another MAP, copy the folder.

The one rule every MAP follows: input is any DICOM tree; output is DICOM only, and
every output object sits in its source's study and references the source instances
it was derived from. SEG MAPs get that from the SDK's DICOMSegmentationWriterOperator
(highdicom, source_images=...). Classification MAPs get it from
ImageFindingSRWriterOperator here, because the SDK's DICOMTextSRWriterOperator does
not record which image a result came from.

`image_finding` and `sr_builder` import no Holoscan, so they can be used and tested
outside a MAP. The operator module imports the SDK; import it explicitly.
"""

from map_ops.image_finding import CodedEvaluation, ImageFinding, NumericMeasurement

__all__ = ["CodedEvaluation", "ImageFinding", "NumericMeasurement"]
