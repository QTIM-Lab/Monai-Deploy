"""Optic disc + fovea detection MAP: DICOM in, one linked SR with box coordinates per photo out.

Canonical MONAI Deploy pipeline, the same shape as maps/aireadi_laterality:

    DICOMDataLoaderOperator          SDK: any DICOM tree, nested dirs, non-DICOM skipped
      -> DICOMSeriesSelectorOperator SDK: keep series matching model_spec.SELECTION_RULES
      -> InstanceDetectorOperator    map_ops: the disc and fovea boxes per instance (via model_spec)
      -> ImageFindingSRWriterOperator map_ops: TID 1500 SR per image; each box a Planar ROI
                                     (closed rectangle) drawn on the source image

Nothing model-specific lives here; see model_spec.py.
"""

import logging
from pathlib import Path

import model_spec
from monai.deploy.conditions import CountCondition
from monai.deploy.core import Application
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator

from map_ops.finding_sr_writer_operator import ImageFindingSRWriterOperator
from map_ops.instance_detector_operator import InstanceDetectorOperator


class DiscDetectionApp(Application):
    def compose(self):
        app_context = Application.init_app_context(self.argv)
        logging.info(f"input={app_context.input_path} output={app_context.output_path} models={app_context.model_path}")

        loader = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=Path(app_context.input_path), name="dicom_loader"
        )
        selector = DICOMSeriesSelectorOperator(
            self, rules=model_spec.SELECTION_RULES, all_matched=True, name="series_selector"
        )
        detector = InstanceDetectorOperator(self, app_context=app_context, model_spec=model_spec, name="detector")
        sr_writer = ImageFindingSRWriterOperator(
            self,
            output_folder=Path(app_context.output_path),
            algorithm_name=model_spec.ALGORITHM_NAME,
            algorithm_version=model_spec.ALGORITHM_VERSION,
            procedure_reported=model_spec.PROCEDURE,
            name="sr_writer",
        )

        self.add_flow(loader, selector, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(selector, detector, {("study_selected_series_list", "study_selected_series_list")})
        self.add_flow(detector, sr_writer, {("findings", "findings")})


if __name__ == "__main__":
    DiscDetectionApp().run()
