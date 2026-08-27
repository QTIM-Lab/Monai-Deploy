import os
from pathlib import Path
from typing import Optional

import torch

from monai.deploy.core import AppContext, ConditionType, Fragment, Operator, OperatorSpec

# Index order must match CLASSES in prepare_data.py. Nothing in model.ts records this,
# so if the two ever disagree the app returns a confident wrong answer, not an error.
LATERALITY_CLASSES = ["L", "R"]

# Must also match prepare_data.py / train.py. Same reasoning.
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class LateralityClassifierOperator(Operator):
    """Classifies a fundus image as left or right eye and returns the label.

    Named inputs:
        image: Image object for which to generate the classification.
        output_folder: Optional, the path to save the results JSON file, overriding the one set on __init__

    Named output:
        result_text: The classification result in text.
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
        model_path (Path): Path to the model file. Defaults to model/model.ts of current working dir.
        output_folder (Path, optional): output folder for saving the classification results JSON file.
        """

        self.input_name_image = "image"
        self.output_name_result = "result_text"
        self.input_name_output_folder = "output_folder"

        self.output_folder = output_folder

        self._model_name = model_name.strip() if isinstance(model_name, str) else ""
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
            model = app_context.models.get(model_name)
        else:
            model = torch.jit.load(
                LateralityClassifierOperator.MODEL_LOCAL_PATH,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

        return model

    def setup(self, spec: OperatorSpec):
        """Set up the operator named input and named output, both are in-memory objects."""

        spec.input(self.input_name_image)
        spec.input(self.input_name_output_folder).condition(ConditionType.NONE)  # Optional for overriding.
        spec.output(self.output_name_result).condition(ConditionType.NONE)  # Not forcing a downstream receiver.

    def preprocess(self, img):
        """Reproduce prepare_data.py's resize and train.py's normalisation, exactly.

        The MAP is the only place these steps live at inference time, so any drift between
        here and training silently degrades the model rather than raising.
        """
        import numpy as np
        from PIL import Image as PILImage

        pil = PILImage.fromarray(img).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR)
        x = torch.from_numpy(np.asarray(pil, dtype=np.uint8)).permute(2, 0, 1).float().div_(255.0)
        mean = torch.tensor(MEAN).view(3, 1, 1)
        std = torch.tensor(STD).view(3, 1, 1)
        return ((x - mean) / std)[None]  # (1, 3, 224, 224)

    def compute(self, op_input, op_output, context):
        import json

        img = op_input.receive(self.input_name_image).asnumpy()  # (H, W, 3) uint8
        image_tensor = self.preprocess(img)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)[0]
        _, output_classes = outputs.max(dim=1)

        result = LATERALITY_CLASSES[output_classes[0]]  # get the class name
        confidence = float(probabilities[output_classes[0]])
        print(f"{result} (confidence {confidence:.4f})")
        op_output.emit(result, self.output_name_result)

        # Get output folder, with value in optional input port overriding the obj attribute
        output_folder_on_compute = op_input.receive(self.input_name_output_folder) or self.output_folder
        Path.mkdir(output_folder_on_compute, parents=True, exist_ok=True)  # Let exception bubble up if raised.
        output_path = output_folder_on_compute / "output.json"
        with open(output_path, "w") as fp:
            json.dump(result, fp)
