"""Build TID 1500 Imaging Measurement Report SRs from ImageFindings, with highdicom.

Pure functions, no Holoscan, so they can be tested on any host.

Each SR:
  - copies Patient and Study identity from its source image (highdicom `evidence=`),
    so it lands in the source study rather than as an orphan;
  - lists the source in CurrentRequestedProcedureEvidenceSequence and in the Image
    Library, and names it as the Source Image of the measurement group;
  - identifies the algorithm (name, version) per TID 4019 and the observer device;
  - carries the coded evaluations and numeric measurements.

All SRs produced for one source study in one run share a Series, so a batch adds one
AI series per study instead of one per image.
"""

from collections import OrderedDict
from typing import Dict, Iterable, List, Optional

import highdicom as hd
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code
from pydicom.uid import generate_uid

from map_ops.image_finding import ImageFinding

# No verified standard code for "AI analysis of an ophthalmic photograph" is available in
# pydicom's dictionary, so the procedure defaults to a private-scheme code. Override per MAP.
DEFAULT_PROCEDURE = Code("AI-IMAGE-ANALYSIS", "99QTIM", "AI image analysis")


def algorithm_device_uid(algorithm_name: str, algorithm_version: str) -> str:
    """A stable UID for the algorithm as an observer device.

    Deterministic, so every SR from the same model version names the same device. That
    makes "which results did version X produce" a plain tag query.
    """
    return generate_uid(entropy_srcs=["map_ops-algorithm", algorithm_name, algorithm_version])


def build_sr(
    finding: ImageFinding,
    *,
    algorithm_name: str,
    algorithm_version: str,
    series_instance_uid: str,
    series_number: int,
    instance_number: int,
    manufacturer: str,
    series_description: str,
    procedure_reported: Code = DEFAULT_PROCEDURE,
    sop_instance_uid: Optional[str] = None,
) -> hd.sr.ComprehensiveSR:
    """One SR for one source image."""
    source = finding.source

    algorithm = hd.sr.AlgorithmIdentification(name=algorithm_name, version=algorithm_version)
    observer = hd.sr.ObservationContext(
        observer_device_context=hd.sr.ObserverContext(
            observer_type=codes.DCM.Device,
            observer_identifying_attributes=hd.sr.DeviceObserverIdentifyingAttributes(
                uid=algorithm_device_uid(algorithm_name, algorithm_version),
                name=algorithm_name,
                manufacturer_name=manufacturer,
                model_name=f"{algorithm_name} {algorithm_version}",
            ),
        )
    )

    group = hd.sr.MeasurementsAndQualitativeEvaluations(
        tracking_identifier=hd.sr.TrackingIdentifier(uid=generate_uid(), identifier=algorithm_name),
        algorithm_id=algorithm,
        qualitative_evaluations=[hd.sr.QualitativeEvaluation(e.name, e.value) for e in finding.evaluations] or None,
        measurements=[
            hd.sr.Measurement(name=m.name, value=m.value, unit=m.unit, algorithm_id=algorithm)
            for m in finding.measurements
        ]
        or None,
        source_images=[
            hd.sr.SourceImageForMeasurementGroup(
                referenced_sop_class_uid=source.SOPClassUID,
                referenced_sop_instance_uid=source.SOPInstanceUID,
            )
        ],
    )

    report = hd.sr.MeasurementReport(
        observation_context=observer,
        procedure_reported=procedure_reported,
        imaging_measurements=[group],
        referenced_images=[source],
    )

    return hd.sr.ComprehensiveSR(
        evidence=[source],
        content=report,
        series_instance_uid=series_instance_uid,
        series_number=series_number,
        sop_instance_uid=sop_instance_uid or generate_uid(),
        instance_number=instance_number,
        manufacturer=manufacturer,
        manufacturer_model_name=algorithm_name,
        software_versions=algorithm_version,
        series_description=series_description,
        is_complete=True,
        is_final=False,
        is_verified=False,
    )


def build_srs(findings: Iterable[ImageFinding], **kwargs) -> List[hd.sr.ComprehensiveSR]:
    """SRs for a batch: one per finding, one shared Series per source study.

    kwargs are build_sr's, minus the per-instance series/instance numbering.
    """
    series_number = kwargs.pop("series_number", 9000)
    by_study: Dict[str, List[ImageFinding]] = OrderedDict()
    for finding in findings:
        by_study.setdefault(finding.source.StudyInstanceUID, []).append(finding)

    srs = []
    for study_findings in by_study.values():
        series_uid = generate_uid()
        for number, finding in enumerate(study_findings, start=1):
            srs.append(
                build_sr(
                    finding,
                    series_instance_uid=series_uid,
                    series_number=series_number,
                    instance_number=number,
                    **kwargs,
                )
            )
    return srs

