"""AI-READI laterality MAP: DICOM in, one linked SR per scored image out.

Canonical MONAI Deploy pipeline, as upstream's breast_density_classifier_app:

    DICOMDataLoaderOperator          SDK: any DICOM tree, nested dirs, non-DICOM skipped
      -> DICOMSeriesSelectorOperator SDK: keep series matching model_spec.SELECTION_RULES
      -> InstanceClassifierOperator  map_ops: batched inference, one finding per instance
      -> ImageFindingSRWriterOperator map_ops: TID 1500 SR referencing each source image

Two departures from upstream, both forced and both in map_ops/: the SDK's
DICOMSeriesToVolumeOperator is monochrome-only (fundus photos are colour), and the
SDK's DICOMTextSRWriterOperator never references the source image.

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
from map_ops.instance_classifier_operator import InstanceClassifierOperator


class AireadiLateralityApp(Application):
    def compose(self):
        app_context = Application.init_app_context(self.argv)
        logging.info(f"input={app_context.input_path} output={app_context.output_path} models={app_context.model_path}")

        loader = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=Path(app_context.input_path), name="dicom_loader"
        )
        selector = DICOMSeriesSelectorOperator(
            self, rules=model_spec.SELECTION_RULES, all_matched=True, name="series_selector"
        )
        classifier = InstanceClassifierOperator(self, app_context=app_context, model_spec=model_spec, name="classifier")
        sr_writer = ImageFindingSRWriterOperator(
            self,
            output_folder=Path(app_context.output_path),
            algorithm_name=model_spec.ALGORITHM_NAME,
            algorithm_version=model_spec.ALGORITHM_VERSION,
            procedure_reported=model_spec.PROCEDURE,
            name="sr_writer",
        )

        self.add_flow(loader, selector, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(selector, classifier, {("study_selected_series_list", "study_selected_series_list")})
        self.add_flow(classifier, sr_writer, {("findings", "findings")})


if __name__ == "__main__":
    AireadiLateralityApp().run()
