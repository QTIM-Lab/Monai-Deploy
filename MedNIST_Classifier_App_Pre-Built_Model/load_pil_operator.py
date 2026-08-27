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

import logging
from pathlib import Path

from monai.deploy.core import Fragment, Image, Operator, OperatorSpec


class LoadPILOperator(Operator):
    """Load image from the given input (DataPath) and set numpy array to the output (Image)."""

    DEFAULT_INPUT_FOLDER = Path.cwd() / "input"
    DEFAULT_OUTPUT_NAME = "image"

    def __init__(
        self,
        fragment: Fragment,
        *args,
        input_folder: Path = DEFAULT_INPUT_FOLDER,
        output_name: str = DEFAULT_OUTPUT_NAME,
        **kwargs,
    ):
        """Creates an loader object with the input folder and the output port name overrides as needed.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            input_folder (Path): Folder from which to load input file(s).
                                 Defaults to `input` in the current working directory.
            output_name (str): Name of the output port, which is an image object. Defaults to `image`.
        """

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.input_path = input_folder
        self.index = 0
        self.output_name_image = (
            output_name.strip() if output_name and len(output_name.strip()) > 0 else LoadPILOperator.DEFAULT_OUTPUT_NAME
        )

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Set up the named input and output port(s)"""
        spec.output(self.output_name_image)

    def compute(self, op_input, op_output, context):
        import numpy as np
        from PIL import Image as PILImage

        input_path = self.input_path
        if input_path.is_dir():
            input_path = next(self.input_path.glob("*.*"))  # take the first file

        image = PILImage.open(input_path)
        image = image.convert("L")  # convert to greyscale image
        image_arr = np.asarray(image)

        output_image = Image(image_arr)  # create Image domain object with a numpy array
        op_output.emit(output_image, self.output_name_image)  # cannot omit the name even if single output.
