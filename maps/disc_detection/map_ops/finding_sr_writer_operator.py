"""Holoscan operator: list[ImageFinding] in, linked DICOM SR files out.

Drop-in replacement for the SDK's DICOMTextSRWriterOperator for classification MAPs
that must record which image each result came from. The SDK writer copies patient and
study identity at best (copy_tags=True) but never references the source instance; see
map_ops/README.md. Candidate for an upstream contribution rather than a fork.
"""

import logging
from pathlib import Path
from typing import List

from monai.deploy.core import Fragment, Operator, OperatorSpec
from pydicom.sr.coding import Code

from map_ops.image_finding import ImageFinding
from map_ops.sr_builder import DEFAULT_PROCEDURE, build_srs


class ImageFindingSRWriterOperator(Operator):
    """Writes one TID 1500 Comprehensive SR per ImageFinding into `output_folder`.

    Named input:
        findings: List[ImageFinding] from a model operator. An empty list writes nothing.

    Writes DICOM only, so the whole output folder can go straight to a DICOM export.
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        output_folder: Path,
        algorithm_name: str,
        algorithm_version: str,
        manufacturer: str = "QTIM",
        series_description: str = "",
        procedure_reported: Code = DEFAULT_PROCEDURE,
        series_number: int = 9000,
        **kwargs,
    ):
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.input_name_findings = "findings"
        self.output_folder = Path(output_folder)
        self.sr_kwargs = dict(
            algorithm_name=algorithm_name,
            algorithm_version=algorithm_version,
            manufacturer=manufacturer,
            series_description=series_description or f"AI {algorithm_name} {algorithm_version} - research use only",
            procedure_reported=procedure_reported,
            series_number=series_number,
        )
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_findings)

    def compute(self, op_input, op_output, context):
        findings: List[ImageFinding] = op_input.receive(self.input_name_findings) or []
        if not findings:
            self._logger.warning("no findings received; no SR written")
            return

        self.output_folder.mkdir(parents=True, exist_ok=True)
        for sr in build_srs(findings, **self.sr_kwargs):
            path = self.output_folder / f"{sr.SOPInstanceUID}.dcm"
            sr.save_as(path, enforce_file_format=True)
            self._logger.debug(f"wrote {path.name}")
        self._logger.info(f"wrote {len(findings)} SR(s) to {self.output_folder}")
