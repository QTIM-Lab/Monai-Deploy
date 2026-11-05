"""Build one DICOM Segmentation per source image, with highdicom. Deterministic.

Pure functions, no Holoscan, so they can be tested on any host.

Each SEG:
  - sits in its source image's Patient and Study, on the source's own pixel grid;
  - references the source instance (ReferencedSeriesSequence, Source Image Sequence);
  - identifies the algorithm (name, version) in the segment description and in
    SeriesDescription "AI <name> <version> - research use only".

Idempotent: one result per (source image, algorithm, version), whatever the batch.
  - SOPInstanceUID    = hash(source SOPInstanceUID, algorithm, version)
  - SeriesInstanceUID = hash(source StudyInstanceUID, algorithm, version): one AI series
                        per study per model version, however the runs were split
  - no clock or random values: dates come from the source, InstanceNumber from the source
So a re-run writes byte-identical files with the same UIDs. Orthanc runs with OverwriteInstances
false, so it keeps what it already has: re-running the same version changes nothing. A new
ALGORITHM_VERSION gets new UIDs, stored next to the old results.
If a re-run's file differs from the stored one under the same UID, that is a conflict
(inference or code changed without a version bump); tools/check_idempotency.py tests for it.

Why not the SDK's DICOMSegmentationWriterOperator (monai-deploy-app-sdk 4.0.0):
  1. it writes one SEG per run, for the first selected series of the first study, and
     drops the rest of a batch silently;
  2. it fails on ophthalmic photographs: see _as_highdicom_source below;
  3. its UIDs are random, so every re-run duplicates results.
"""

import copy
from typing import List, Sequence, Tuple

import highdicom as hd
import numpy as np
from pydicom import Dataset
from pydicom.sr.codedict import codes
from pydicom.uid import generate_uid

# Root for the UIDs we mint: pydicom's public root, the one all our AI outputs already use
# (laterality SRs, earlier SEGs). Not a device vendor's root (e.g. Topcon's 2.16.840.1.114517):
# minting under it would claim to be that vendor's object and could collide with its UIDs.
# Determinism doesn't depend on the root. Never change it casually: every UID would change.
UID_ROOT = "1.2.826.0.1.3680043.8.498."


def stable_uid(*parts: str) -> str:
    """Same parts -> same UID, always (SHA-512 of the parts under UID_ROOT)."""
    return generate_uid(prefix=UID_ROOT, entropy_srcs=["map_ops-seg", *map(str, parts)])


def _as_highdicom_source(source: Dataset) -> Dataset:
    """Header copy of the source that highdicom 0.28 can segment.

    highdicom treats every SOP class whose IOD *allows* NumberOfFrames as an enhanced
    multi-frame image, and reads pixel spacing from SharedFunctionalGroupsSequence.
    Ophthalmic Photography allows NumberOfFrames but has no functional groups, so
    hd.seg.Segmentation raises AttributeError. Supplying the group in a copy fixes it;
    the original dataset and every UID the SEG references are unchanged.
    """
    shim = copy.copy(source)  # shallow: only the header is read
    measures = Dataset()
    measures.PixelSpacing = source.get("PixelSpacing", [1, 1])
    group = Dataset()
    group.PixelMeasuresSequence = [measures]
    shim.SharedFunctionalGroupsSequence = [group]
    return shim


def _remove_volatile(seg: Dataset, source: Dataset) -> None:
    """Replace what highdicom fills from the clock or a random generator."""
    # Content date/time: the source image's, so the value is stable and still meaningful
    # ("the content this describes was acquired then"). Creation and contribution times are
    # optional (Type 3) and only ever say "when this run happened": dropped.
    seg.ContentDate = source.get("ContentDate") or source.get("StudyDate") or "19000101"
    seg.ContentTime = source.get("ContentTime") or source.get("StudyTime") or "000000"
    for keyword in ("InstanceCreationDate", "InstanceCreationTime"):
        if keyword in seg:
            delattr(seg, keyword)
    for item in seg.get("ContributingEquipmentSequence", []):
        if "ContributionDateTime" in item:
            del item.ContributionDateTime
    # highdicom mints a random Dimension Organization UID; tie it to this SEG instead.
    dimension_uid = stable_uid("dimension-organization", seg.SOPInstanceUID)
    for item in seg.DimensionOrganizationSequence:
        item.DimensionOrganizationUID = dimension_uid
    for item in seg.DimensionIndexSequence:
        item.DimensionOrganizationUID = dimension_uid


def build_seg(
    source: Dataset,
    mask: np.ndarray,
    *,
    segment: dict,
    algorithm_name: str,
    algorithm_version: str,
    series_number: int,
    manufacturer: str,
    series_description: str,
) -> hd.seg.Segmentation:
    """One binary SEG for one 2D source image. mask: (Rows, Columns), nonzero = in segment."""
    description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label=segment["label"],
        segmented_property_category=segment["category"],
        segmented_property_type=segment["type"],
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=hd.AlgorithmIdentificationSequence(
            name=algorithm_name, version=algorithm_version, family=codes.DCM.ArtificialIntelligence
        ),
        anatomic_regions=[segment["anatomic_region"]] if segment.get("anatomic_region") else None,
    )
    seg = hd.seg.Segmentation(
        source_images=[_as_highdicom_source(source)],
        pixel_array=(mask > 0).astype(np.uint8)[None],  # one frame
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[description],
        series_instance_uid=stable_uid("series", source.StudyInstanceUID, algorithm_name, algorithm_version),
        series_number=series_number,
        sop_instance_uid=stable_uid("seg", source.SOPInstanceUID, algorithm_name, algorithm_version),
        instance_number=int(source.get("InstanceNumber") or 1),
        manufacturer=manufacturer,
        manufacturer_model_name=algorithm_name,
        software_versions=algorithm_version,
        device_serial_number="0",
        series_description=series_description,
        omit_empty_frames=False,  # an image with no vessels found still gets a (blank) SEG
    )
    # The pixel measures came from the shim. highdicom's own reader rejects them without a
    # SliceThickness, which a photograph doesn't have. The SEG keeps the source's pixel grid
    # (Spatial Locations Preserved = YES), so spacing is the source image's.
    del seg.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence
    _remove_volatile(seg, source)
    return seg


def build_segs(results: Sequence[Tuple[Dataset, np.ndarray]], **kwargs) -> List[hd.seg.Segmentation]:
    """One SEG per (source, mask). Each SEG's identity depends only on its own source, so
    the same image gives the same SEG whether it is run alone, with its study, or in a batch."""
    kwargs.setdefault("series_number", 9100)
    return [build_seg(source, mask, **kwargs) for source, mask in results]
