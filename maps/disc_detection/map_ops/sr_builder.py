"""Build TID 1500 Imaging Measurement Report SRs from ImageFindings, with highdicom.

Pure functions, no Holoscan, so they can be tested on any host.

Each SR:
  - copies Patient and Study identity from its source image (highdicom `evidence=`),
    so it lands in the source study rather than as an orphan;
  - lists the source in CurrentRequestedProcedureEvidenceSequence and in the Image
    Library, and names it as the Source Image of the measurement group;
  - identifies the algorithm (name, version) per TID 4019 and the observer device;
  - carries the coded evaluations and numeric measurements (one TID 1501 group);
  - one TID 1410 Planar ROI group per located structure (BoxFinding): finding type, a closed
    POLYLINE rectangle drawn on the source image, and its measurements (e.g. the score);
  - one TID 1501 group per structure looked for but not found: Presence of property = Absent.

Deterministic: the same source image and algorithm version always give the same SR, with the
same UIDs, whether it was scored alone, in a study or in a batch. SOP Instance UID derives from
(source SOP Instance UID, algorithm, version); Series Instance UID from (study, algorithm,
version), so each study gets one AI series per model version. No clock or random values are
written. With Orthanc's OverwriteInstances off, a re-send is ignored as already stored and a
new version is stored alongside.
"""

from typing import Iterable, List

import highdicom as hd
import numpy as np
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code
from pydicom.uid import generate_uid

from map_ops.image_finding import ImageFinding

# No verified standard code for "AI analysis of an ophthalmic photograph" is available in
# pydicom's dictionary, so the procedure defaults to a private-scheme code. Override per MAP.
DEFAULT_PROCEDURE = Code("AI-IMAGE-ANALYSIS", "99QTIM", "AI image analysis")

# Root for the UIDs we mint: pydicom's public root, as earlier SRs used. Never change it
# casually: every UID would change.
UID_ROOT = "1.2.826.0.1.3680043.8.498."


def stable_uid(*parts: str) -> str:
    """Same parts -> same UID, always (SHA-512 of the parts under UID_ROOT)."""
    return generate_uid(prefix=UID_ROOT, entropy_srcs=["map_ops-sr", *map(str, parts)])


def algorithm_device_uid(algorithm_name: str, algorithm_version: str) -> str:
    """A stable UID for the algorithm as an observer device.

    Deterministic, so every SR from the same model version names the same device. That
    makes "which results did version X produce" a plain tag query.
    """
    return generate_uid(entropy_srcs=["map_ops-algorithm", algorithm_name, algorithm_version])


def _remove_volatile(sr: hd.sr.ComprehensiveSR, source) -> None:
    """Replace what highdicom fills from the clock with values tied to the source image."""
    sr.ContentDate = source.get("ContentDate") or source.get("StudyDate") or "19000101"
    sr.ContentTime = source.get("ContentTime") or source.get("StudyTime") or "000000"
    for keyword in ("InstanceCreationDate", "InstanceCreationTime", "SeriesDate", "SeriesTime"):
        if keyword in sr:
            delattr(sr, keyword)
    for item in sr.get("ContributingEquipmentSequence", []):
        if "ContributionDateTime" in item:
            del item.ContributionDateTime


def build_sr(
    finding: ImageFinding,
    *,
    algorithm_name: str,
    algorithm_version: str,
    manufacturer: str,
    series_description: str,
    procedure_reported: Code = DEFAULT_PROCEDURE,
    series_number: int = 9000,
) -> hd.sr.ComprehensiveSR:
    """One SR for one source image."""
    source = finding.source
    key = (algorithm_name, algorithm_version)

    algorithm = hd.sr.AlgorithmIdentification(name=algorithm_name, version=algorithm_version)
    observer = hd.sr.ObservationContext(
        observer_device_context=hd.sr.ObserverContext(
            observer_type=codes.DCM.Device,
            observer_identifying_attributes=hd.sr.DeviceObserverIdentifyingAttributes(
                uid=algorithm_device_uid(*key),
                name=algorithm_name,
                manufacturer_name=manufacturer,
                model_name=f"{algorithm_name} {algorithm_version}",
            ),
        )
    )

    def measurements(items):
        return [hd.sr.Measurement(name=m.name, value=m.value, unit=m.unit, algorithm_id=algorithm) for m in items] or None

    def tracking(*what):
        uid = stable_uid("tracking", source.SOPInstanceUID, *what, *key)
        return hd.sr.TrackingIdentifier(uid=uid, identifier=" ".join([algorithm_name, *what]))

    source_image = [
        hd.sr.SourceImageForMeasurementGroup(
            referenced_sop_class_uid=source.SOPClassUID,
            referenced_sop_instance_uid=source.SOPInstanceUID,
        )
    ]
    groups = []
    if finding.evaluations or finding.measurements:
        groups.append(
            hd.sr.MeasurementsAndQualitativeEvaluations(
                tracking_identifier=hd.sr.TrackingIdentifier(
                    uid=stable_uid("tracking", source.SOPInstanceUID, *key), identifier=algorithm_name
                ),
                algorithm_id=algorithm,
                qualitative_evaluations=[hd.sr.QualitativeEvaluation(e.name, e.value) for e in finding.evaluations] or None,
                measurements=measurements(finding.measurements),
                source_images=source_image,
            )
        )
    for box in finding.boxes:
        x0, y0, x1, y1 = box.box
        groups.append(
            hd.sr.PlanarROIMeasurementsAndQualitativeEvaluations(
                tracking_identifier=tracking(box.finding_type.meaning),
                referenced_region=hd.sr.ImageRegion(
                    graphic_type=hd.sr.GraphicTypeValues.POLYLINE,
                    graphic_data=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=float),
                    source_image=hd.sr.SourceImageForRegion(
                        referenced_sop_class_uid=source.SOPClassUID,
                        referenced_sop_instance_uid=source.SOPInstanceUID,
                    ),
                ),
                finding_type=box.finding_type,
                algorithm_id=algorithm,
                measurements=measurements(box.measurements),
            )
        )
    for structure in finding.not_found:
        groups.append(
            hd.sr.MeasurementsAndQualitativeEvaluations(
                tracking_identifier=tracking(structure.meaning),
                finding_type=structure,
                algorithm_id=algorithm,
                qualitative_evaluations=[hd.sr.QualitativeEvaluation(codes.DCM.PresenceOfProperty, codes.SCT.Absent)],
                source_images=source_image,
            )
        )

    report = hd.sr.MeasurementReport(
        observation_context=observer,
        procedure_reported=procedure_reported,
        imaging_measurements=groups,
        referenced_images=[source],
    )

    sr = hd.sr.ComprehensiveSR(
        evidence=[source],
        content=report,
        series_instance_uid=stable_uid("series", source.StudyInstanceUID, *key),
        series_number=series_number,
        sop_instance_uid=stable_uid("sr", source.SOPInstanceUID, *key),
        instance_number=int(source.get("InstanceNumber") or 1),
        manufacturer=manufacturer,
        manufacturer_model_name=algorithm_name,
        software_versions=algorithm_version,
        series_description=series_description,
        is_complete=True,
        is_final=False,
        is_verified=False,
    )
    _remove_volatile(sr, source)
    return sr


def build_srs(findings: Iterable[ImageFinding], **kwargs) -> List[hd.sr.ComprehensiveSR]:
    """SRs for a batch: one per finding. kwargs are build_sr's."""
    return [build_sr(finding, **kwargs) for finding in findings]
