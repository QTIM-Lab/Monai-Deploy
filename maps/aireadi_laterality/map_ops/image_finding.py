"""The contract between a model operator and the result writer.

A model operator emits a list of ImageFinding, one per image it scored. The writer
turns each into a DICOM SR that references `source`. Model code never builds DICOM,
and the writer never knows what the model computed beyond these coded values.

Use real codes (SNOMED CT "SCT", DICOM "DCM", LOINC "LN"); pydicom ships a dictionary:
`from pydicom.sr.codedict import codes; codes.SCT.Laterality`. Where no standard code
exists, use a private scheme designator beginning with "99" rather than inventing a
SNOMED value.
"""

from dataclasses import dataclass, field
from typing import List

from pydicom.dataset import Dataset
from pydicom.sr.coding import Code


@dataclass(frozen=True)
class CodedEvaluation:
    """A categorical result, e.g. Laterality = Right."""

    name: Code
    value: Code


@dataclass(frozen=True)
class NumericMeasurement:
    """A numeric result with UCUM units, e.g. Probability = 0.97 {unity}."""

    name: Code
    value: float
    unit: Code


@dataclass
class ImageFinding:
    """Everything the model concluded about one source image.

    `source` is the instance's dataset as loaded by DICOMDataLoaderOperator. Only its
    header is used (patient, study, series and SOP identity), never its pixels.
    """

    source: Dataset
    evaluations: List[CodedEvaluation] = field(default_factory=list)
    measurements: List[NumericMeasurement] = field(default_factory=list)

    def __post_init__(self):
        for keyword in ("SOPClassUID", "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID"):
            if keyword not in self.source:
                raise ValueError(f"source dataset lacks {keyword}; cannot reference it from a result")
        if not self.evaluations and not self.measurements:
            raise ValueError("a finding needs at least one evaluation or measurement")
