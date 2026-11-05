"""Holoscan operator: list of (source, mask) in, one linked DICOM SEG file per image out.

Replaces the SDK's DICOMSegmentationWriterOperator for 2D image MAPs; seg_builder.py
says why.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from monai.deploy.core import Fragment, Operator, OperatorSpec
from pydicom import Dataset

from map_ops.seg_builder import build_segs


class InstanceSEGWriterOperator(Operator):
    """Writes one binary DICOM SEG per (source, mask) into `output_folder`.

    Named input:
        segmentations: List[(source Dataset, mask)] from a model operator. Empty writes nothing.

    Writes DICOM only, so the whole output folder can go straight to a DICOM export.
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        output_folder: Path,
        segment: dict,
        algorithm_name: str,
        algorithm_version: str,
        manufacturer: str = "QTIM",
        series_description: str = "",
        series_number: int = 9100,
        **kwargs,
    ):
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.input_name = "segmentations"
        self.output_folder = Path(output_folder)
        self.seg_kwargs = dict(
            segment=segment,
            algorithm_name=algorithm_name,
            algorithm_version=algorithm_version,
            manufacturer=manufacturer,
            series_description=series_description or f"AI {algorithm_name} {algorithm_version} - research use only",
            series_number=series_number,
        )
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name)

    def compute(self, op_input, op_output, context):
        results: List[Tuple[Dataset, np.ndarray]] = op_input.receive(self.input_name) or []
        if not results:
            self._logger.warning("nothing segmented; no SEG written")
            return

        self.output_folder.mkdir(parents=True, exist_ok=True)
        for seg in build_segs(results, **self.seg_kwargs):
            path = self.output_folder / f"{seg.SOPInstanceUID}.dcm"
            seg.save_as(path, enforce_file_format=True)
            self._logger.debug(f"wrote {path.name}")
        self._logger.info(f"wrote {len(results)} SEG(s) to {self.output_folder}")
