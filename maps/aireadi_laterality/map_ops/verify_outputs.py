"""Check a MAP's output against the linking rule. No Holoscan; runs anywhere.

    PYTHONPATH=maps/aireadi_laterality python -m map_ops.verify_outputs INPUT_DIR OUTPUT_DIR [--json]

For every file in OUTPUT_DIR:
  - it is DICOM (a non-DICOM file would break a C-STORE export of the folder);
  - its StudyInstanceUID and PatientID match an input instance's;
  - every instance it references (SR evidence, SEG source images) is an input instance;
  - it references at least one.

Also reports which input instances received no result, and, for SRs, the coded and
numeric values found, so a batch can be eyeballed or scored.

Exit status: 0 all outputs linked, 1 any violation, 2 no outputs at all.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError


def read_headers(root: Path) -> Iterator[Tuple[Path, Dataset]]:
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        try:
            yield path, pydicom.dcmread(path, stop_before_pixels=True)
        except (InvalidDicomError, OSError):
            yield path, None


def referenced_sops(ds: Dataset) -> Set[str]:
    """SOP Instance UIDs an output object says it was derived from."""
    refs: Set[str] = set()
    for study in ds.get("CurrentRequestedProcedureEvidenceSequence", []):
        for series in study.get("ReferencedSeriesSequence", []):
            refs.update(item.ReferencedSOPInstanceUID for item in series.get("ReferencedSOPSequence", []))
    for series in ds.get("ReferencedSeriesSequence", []):  # SEG
        refs.update(item.ReferencedSOPInstanceUID for item in series.get("ReferencedInstanceSequence", []))
    for item in ds.get("SourceImageSequence", []):
        refs.add(item.ReferencedSOPInstanceUID)
    return refs


def sr_values(ds: Dataset) -> List[str]:
    """'Name = Value' for the CODE and NUM items under TID 1500's Imaging Measurements.

    Skips the report header and Image Library, which describe context, not results.
    """
    values = []

    def walk(items, inside):
        for item in items:
            code = item.ConceptNameCodeSequence[0] if "ConceptNameCodeSequence" in item else None
            here = inside or (code is not None and code.CodeValue == "126010")  # DCM Imaging Measurements
            if here and code is not None and code.CodeValue not in ("111001", "111003", "112039", "112040"):
                if item.ValueType == "CODE":
                    values.append(f"{code.CodeMeaning} = {item.ConceptCodeSequence[0].CodeMeaning}")
                elif item.ValueType == "NUM":
                    values.append(f"{code.CodeMeaning} = {float(item.MeasuredValueSequence[0].NumericValue):.4f}")
            walk(item.get("ContentSequence", []), here)

    walk(ds.get("ContentSequence", []), False)
    return values


def verify(input_dir: Path, output_dir: Path) -> Dict:
    inputs = {ds.SOPInstanceUID: ds for _, ds in read_headers(input_dir) if ds is not None and "SOPInstanceUID" in ds}
    report = {"inputs": len(inputs), "outputs": [], "violations": [], "unreferenced_inputs": []}
    covered: Set[str] = set()

    for path, ds in read_headers(output_dir):
        entry = {"file": path.name}
        report["outputs"].append(entry)
        if ds is None:
            report["violations"].append(f"{path.name}: not DICOM; a DICOM export of this folder would fail")
            continue

        entry.update(modality=ds.get("Modality"), sop_class=ds.SOPClassUID.name)
        refs = referenced_sops(ds)
        entry["references"] = sorted(refs)
        if ds.get("Modality") == "SR":
            entry["values"] = sr_values(ds)

        if not refs:
            report["violations"].append(f"{path.name}: references no source instance")
        unknown = refs - inputs.keys()
        if unknown:
            report["violations"].append(f"{path.name}: references {len(unknown)} instance(s) not in the input")
        for sop in refs & inputs.keys():
            src = inputs[sop]
            covered.add(sop)
            if ds.StudyInstanceUID != src.StudyInstanceUID:
                report["violations"].append(f"{path.name}: StudyInstanceUID differs from its source {sop}")
            if ds.get("PatientID", "") != src.get("PatientID", ""):
                report["violations"].append(f"{path.name}: PatientID differs from its source {sop}")

    report["unreferenced_inputs"] = [
        {"sop": sop, "modality": ds.get("Modality"), "sop_class": getattr(ds.get("SOPClassUID"), "name", "?")}
        for sop, ds in inputs.items()
        if sop not in covered
    ]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="map_ops.verify_outputs", description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)

    report = verify(args.input_dir, args.output_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{report['inputs']} input instance(s), {len(report['outputs'])} output file(s)")
        for entry in report["outputs"]:
            values = "; ".join(entry.get("values", []))
            print(f"  {entry['file']}  {entry.get('modality', '?')}  refs={len(entry.get('references', []))}  {values}")
        for item in report["unreferenced_inputs"]:
            print(f"  no result for {item['sop']} ({item['modality']}, {item['sop_class']})")
        for violation in report["violations"]:
            print(f"  VIOLATION {violation}")
        print("OK: every output is linked to its source" if not report["violations"] else "FAILED")

    if not report["outputs"]:
        return 2
    return 1 if report["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
