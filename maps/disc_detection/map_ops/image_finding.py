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
from typing import List, Tuple

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


@dataclass(frozen=True)
class BoxFinding:
    """A located structure, e.g. Optic disc at this rectangle, with its own measurements (a score).

    `box` is (x0, y0, x1, y1) in the source image's pixel coordinates, DICOM SCOORD convention:
    (0, 0) is the top-left corner of the top-left pixel. Written as a closed POLYLINE rectangle
    on the source image (TID 1410 Planar ROI), so a viewer can draw it over the image.
    """

    finding_type: Code
    box: Tuple[float, float, float, float]
    measurements: List[NumericMeasurement] = field(default_factory=list)


@dataclass
class ImageFinding:
    """Everything the model concluded about one source image.

    `source` is the instance's dataset as loaded by DICOMDataLoaderOperator. Only its
    header is used (patient, study, series and SOP identity), never its pixels.
    """

    source: Dataset
    evaluations: List[CodedEvaluation] = field(default_factory=list)
    measurements: List[NumericMeasurement] = field(default_factory=list)
    boxes: List[BoxFinding] = field(default_factory=list)
    # Structures the model looked for and did not find: written as "<structure>: absent", so an image
    # with nothing found still gets its one result (and is not re-run as "missing").
    not_found: List[Code] = field(default_factory=list)

    def __post_init__(self):
        for keyword in ("SOPClassUID", "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID"):
            if keyword not in self.source:
                raise ValueError(f"source dataset lacks {keyword}; cannot reference it from a result")
        if not (self.evaluations or self.measurements or self.boxes or self.not_found):
            raise ValueError("a finding needs at least one evaluation, measurement, box or not-found entry")
