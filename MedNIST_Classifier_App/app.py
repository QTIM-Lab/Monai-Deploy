# Copyright 2021-2023 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from load_pil_operator import LoadPILOperator
from mednist_classifier_operator import MedNISTClassifierOperator

from monai.deploy.conditions import CountCondition
from monai.deploy.core import AppContext, Application
from monai.deploy.operators.dicom_text_sr_writer_operator import DICOMTextSRWriterOperator, EquipmentInfo, ModelInfo


# @md.resource(cpu=1, gpu=1, memory="1Gi")
class App(Application):
    """Application class for the MedNIST classifier."""

    def compose(self):
        app_context = AppContext({})  # Let it figure out all the attributes without overriding
        app_input_path = Path(app_context.input_path)
        app_output_path = Path(app_context.output_path)
        model_path = Path(app_context.model_path)
        load_pil_op = LoadPILOperator(self, CountCondition(self, 1), input_folder=app_input_path, name="pil_loader_op")
        classifier_op = MedNISTClassifierOperator(
            self, app_context=app_context, output_folder=app_output_path, model_path=model_path, name="classifier_op"
        )

        my_model_info = ModelInfo("MONAI WG Trainer", "MEDNIST Classifier", "0.1", "xyz")
        my_equipment = EquipmentInfo(manufacturer="MOANI Deploy App SDK", manufacturer_model="DICOM SR Writer")
        my_special_tags = {"SeriesDescription": "Not for clinical use. The result is for research use only."}
        dicom_sr_operator = DICOMTextSRWriterOperator(
            self,
            copy_tags=False,
            model_info=my_model_info,
            equipment_info=my_equipment,
            custom_tags=my_special_tags,
            output_folder=app_output_path,
        )

        self.add_flow(load_pil_op, classifier_op, {("image", "image")})
        self.add_flow(classifier_op, dicom_sr_operator, {("result_text", "text")})


if __name__ == "__main__":
    App().run()
