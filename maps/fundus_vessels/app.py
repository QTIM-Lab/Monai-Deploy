"""Fundus vessel segmentation MAP: DICOM in, one linked SEG per segmented image out.

Canonical MONAI Deploy pipeline, the same shape as maps/aireadi_laterality:

    DICOMDataLoaderOperator          SDK: any DICOM tree, nested dirs, non-DICOM skipped
      -> DICOMSeriesSelectorOperator SDK: keep series matching model_spec.SELECTION_RULES
      -> InstanceSegmenterOperator   map_ops: one mask per instance (VascX, via model_spec)
      -> InstanceSEGWriterOperator   map_ops: one DICOM SEG per image, referencing it

Why the SEG writer is ours rather than the SDK's: map_ops/seg_builder.py.

Nothing model-specific lives here; see model_spec.py.
"""

import logging
from pathlib import Path

import model_spec
from monai.deploy.conditions import CountCondition
from monai.deploy.core import Application
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator

from map_ops.instance_segmenter_operator import InstanceSegmenterOperator
from map_ops.seg_writer_operator import InstanceSEGWriterOperator


class FundusVesselsApp(Application):
    def compose(self):
        app_context = Application.init_app_context(self.argv)
        logging.info(f"input={app_context.input_path} output={app_context.output_path} models={app_context.model_path}")

        loader = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=Path(app_context.input_path), name="dicom_loader"
        )
        selector = DICOMSeriesSelectorOperator(
            self, rules=model_spec.SELECTION_RULES, all_matched=True, name="series_selector"
        )
        segmenter = InstanceSegmenterOperator(self, app_context=app_context, model_spec=model_spec, name="segmenter")
        seg_writer = InstanceSEGWriterOperator(
            self,
            output_folder=Path(app_context.output_path),
            segment=model_spec.SEGMENT,
            algorithm_name=model_spec.ALGORITHM_NAME,
            algorithm_version=model_spec.ALGORITHM_VERSION,
            name="seg_writer",
        )

        self.add_flow(loader, selector, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(selector, segmenter, {("study_selected_series_list", "study_selected_series_list")})
        self.add_flow(segmenter, seg_writer, {("segmentations", "segmentations")})


if __name__ == "__main__":
    FundusVesselsApp().run()
