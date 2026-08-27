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

import os
from pathlib import Path
from typing import Optional

import torch

from monai.deploy.core import AppContext, ConditionType, Fragment, Operator, OperatorSpec
from monai.transforms import EnsureChannelFirst, Compose, EnsureType, ScaleIntensity

# The class order is an assumption this app makes about the pre-built model. Nothing in
# classifier.zip states it, and a mismatch produces a confident wrong label, not an error.
MEDNIST_CLASSES = ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"]


class MedNISTClassifierOperator(Operator):
    """Classifies the given image and returns the class name.

    Named inputs:
        image: Image object for which to generate the classification.
        output_folder: Optional, the path to save the results JSON file, overridingthe the one set on __init__

    Named output:
        result_text: The classification results in text.
    """

    DEFAULT_OUTPUT_FOLDER = Path.cwd() / "classification_results"
    # For testing the app directly, the model should be at the following path.
    MODEL_LOCAL_PATH = Path(os.environ.get("HOLOSCAN_MODEL_PATH", Path.cwd() / "model/model.ts"))

    def __init__(
        self,
        fragment: Fragment,
        *args,
        app_context: AppContext,
        model_name: Optional[str] = "",
        model_path: Path = MODEL_LOCAL_PATH,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        **kwargs,
    ):
        """Creates an instance with the reference back to the containing application/fragment.

        fragment (Fragment): An instance of the Application class which is derived from Fragment.
        model_name (str, optional): Name of the model. Default to "" for single model app.
        model_path (Path): Path to the model file. Defaults to model/models.ts of current working dir.
        output_folder (Path, optional): output folder for saving the classification results JSON file.
        """

        # the names used for the model inference input and output
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"

        # The names used for the operator input and output
        self.input_name_image = "image"
        self.output_name_result = "result_text"

        # The name of the optional input port for passing data to override the output folder path.
        self.input_name_output_folder = "output_folder"

        # The output folder set on the object can be overridden at each compute by data in the optional named input
        self.output_folder = output_folder

        # Need the name when there are multiple models loaded
        self._model_name = model_name.strip() if isinstance(model_name, str) else ""
        # Need the path to load the models when they are not loaded in the execution context
        self.model_path = model_path
        self.app_context = app_context
        self.model = self._get_model(self.app_context, self.model_path, self._model_name)

        # This needs to be at the end of the constructor.
        super().__init__(fragment, *args, **kwargs)

    def _get_model(self, app_context: AppContext, model_path: Path, model_name: str):
        """Load the model with the given name from context or model path

        Args:
            app_context (AppContext): The application context object holding the model(s)
            model_path (Path): The path to the model file, as a backup to load model directly
            model_name (str): The name of the model, when multiples are loaded in the context
        """

        if app_context.models:
            # `app_context.models.get(model_name)` returns a model instance if exists.
            # If model_name is not specified and only one model exists, it returns that model.
            model = app_context.models.get(model_name)
        else:
            model = torch.jit.load(
                MedNISTClassifierOperator.MODEL_LOCAL_PATH,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

        return model

    def setup(self, spec: OperatorSpec):
        """Set up the operator named input and named output, both are in-memory objects."""

        spec.input(self.input_name_image)
        spec.input(self.input_name_output_folder).condition(ConditionType.NONE)  # Optional for overriding.
        spec.output(self.output_name_result).condition(ConditionType.NONE)  # Not forcing a downstream receiver.

    @property
    def transform(self):
        return Compose([EnsureChannelFirst(channel_dim="no_channel"), ScaleIntensity(), EnsureType()])

    def compute(self, op_input, op_output, context):
        import json

        import torch

        img = op_input.receive(self.input_name_image).asnumpy()  # (64, 64), uint8. Input validation can be added.
        image_tensor = self.transform(img)  # (1, 64, 64), torch.float64
        image_tensor = image_tensor[None].float()  # (1, 1, 64, 64), torch.float32

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        _, output_classes = outputs.max(dim=1)

        result = MEDNIST_CLASSES[output_classes[0]]  # get the class name
        print(result)
        op_output.emit(result, self.output_name_result)

        # Get output folder, with value in optional input port overriding the obj attribute
        output_folder_on_compute = op_input.receive(self.input_name_output_folder) or self.output_folder
        Path.mkdir(output_folder_on_compute, parents=True, exist_ok=True)  # Let exception bubble up if raised.
        output_path = output_folder_on_compute / "output.json"
        with open(output_path, "w") as fp:
            json.dump(result, fp)
