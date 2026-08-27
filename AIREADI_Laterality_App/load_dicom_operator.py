import logging
from pathlib import Path

from monai.deploy.core import Fragment, Image, Operator, OperatorSpec


class LoadDICOMOperator(Operator):
    """Load a single-frame fundus DICOM from the input folder and emit it as an Image.

    This is the AI-READI equivalent of the MedNIST app's LoadPILOperator. Retinal
    photography DICOMs are ordinary single-frame colour images, so pydicom reads them
    directly. MONAI's DICOM series operators are for volumetric CT/MR and are the wrong
    tool here.
    """

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
        """Creates a loader object with the input folder and output port name overrides as needed.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            input_folder (Path): Folder from which to load input file(s).
                                 Defaults to `input` in the current working directory.
            output_name (str): Name of the output port, which is an image object. Defaults to `image`.
        """

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.input_path = input_folder
        self.output_name_image = (
            output_name.strip() if output_name and len(output_name.strip()) > 0 else LoadDICOMOperator.DEFAULT_OUTPUT_NAME
        )

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Set up the named input and output port(s)"""
        spec.output(self.output_name_image)

    def compute(self, op_input, op_output, context):
        import numpy as np
        import pydicom
        from pydicom.pixel_data_handlers.util import convert_color_space

        input_path = self.input_path
        if input_path.is_dir():
            input_path = next(input_path.glob("*.dcm"))  # take the first DICOM

        self._logger.info(f"Reading DICOM: {input_path}")
        ds = pydicom.dcmread(input_path)
        arr = ds.pixel_array

        # pydicom returns the stored colour space without converting. These fundus images
        # are frequently YBR_FULL_422, which would look wrong if treated as RGB.
        if ds.PhotometricInterpretation.startswith("YBR"):
            arr = convert_color_space(arr, ds.PhotometricInterpretation, "RGB")

        # Log the ground truth when the tag is present, so the demo can be checked live.
        truth = getattr(ds, "ImageLaterality", None) or getattr(ds, "Laterality", None)
        if truth:
            self._logger.info(f"DICOM ImageLaterality tag says: {truth}")

        op_output.emit(Image(np.asarray(arr, dtype=np.uint8)), self.output_name_image)
