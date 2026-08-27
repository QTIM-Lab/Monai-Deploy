from pathlib import Path

from load_dicom_operator import LoadDICOMOperator
from laterality_classifier_operator import LateralityClassifierOperator

from monai.deploy.conditions import CountCondition
from monai.deploy.core import Application
from monai.deploy.operators.dicom_text_sr_writer_operator import DICOMTextSRWriterOperator, EquipmentInfo, ModelInfo


class App(Application):
    """Application class for the AI-READI laterality classifier."""

    def compose(self):
        app_context = Application.init_app_context(self.argv)
        app_input_path = Path(app_context.input_path)
        app_output_path = Path(app_context.output_path)
        model_path = Path(app_context.model_path)

        load_dicom_op = LoadDICOMOperator(
            self, CountCondition(self, 1), input_folder=app_input_path, name="dicom_loader_op"
        )
        classifier_op = LateralityClassifierOperator(
            self, app_context=app_context, output_folder=app_output_path, model_path=model_path, name="classifier_op"
        )

        my_model_info = ModelInfo("QTIM", "AI-READI Laterality Classifier", "0.1", "resnet18")
        my_equipment = EquipmentInfo(manufacturer="MONAI Deploy App SDK", manufacturer_model="DICOM SR Writer")
        my_special_tags = {"SeriesDescription": "Not for clinical use. The result is for research use only."}
        dicom_sr_operator = DICOMTextSRWriterOperator(
            self,
            copy_tags=False,
            model_info=my_model_info,
            equipment_info=my_equipment,
            custom_tags=my_special_tags,
            output_folder=app_output_path,
        )

        self.add_flow(load_dicom_op, classifier_op, {("image", "image")})
        self.add_flow(classifier_op, dicom_sr_operator, {("result_text", "text")})


if __name__ == "__main__":
    App().run()
