"""Score a batch run: links valid (map_ops.verify_outputs) and predictions vs ImageLaterality.

    python maps/aireadi_laterality/tools/check_batch.py INPUT_DIR OUTPUT_DIR

Exit 0 only if every output is linked to its source, every eligible (OP colour) input
got exactly one result, and nothing ineligible got one. Accuracy is reported, not
gated: it is a model property, not a pipeline property.
"""

import sys
from collections import Counter
from pathlib import Path

import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the MAP folder, holding map_ops/
from map_ops.verify_outputs import read_headers, verify  # noqa: E402

LABEL = {"Left": "L", "Right": "R"}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    input_dir, output_dir = Path(sys.argv[1]), Path(sys.argv[2])

    report = verify(input_dir, output_dir)
    failures = list(report["violations"])

    inputs = {ds.SOPInstanceUID: ds for _, ds in read_headers(input_dir) if ds is not None and "SOPInstanceUID" in ds}
    eligible = {sop for sop, ds in inputs.items() if ds.get("Modality") == "OP" and int(ds.get("SamplesPerPixel", 1)) == 3}

    results = Counter()
    predictions = {}
    for entry in report["outputs"]:
        for sop in entry.get("references", []):
            results[sop] += 1
            value = next((v.split(" = ")[1] for v in entry.get("values", []) if v.startswith("Laterality")), None)
            prob = next((float(v.split(" = ")[1]) for v in entry.get("values", []) if v.startswith("Probability")), None)
            predictions[sop] = (LABEL.get(value, value), prob)

    for sop in eligible - results.keys():
        failures.append(f"eligible input got no result: {sop}")
    for sop, n in results.items():
        if n > 1:
            failures.append(f"input got {n} results: {sop}")
        if sop not in eligible:
            failures.append(f"ineligible input got a result: {sop} ({inputs[sop].get('Modality')})")

    correct = 0
    for sop, (pred, prob) in sorted(predictions.items(), key=lambda kv: inputs[kv[0]].get("PatientID", "")):
        truth = inputs[sop].get("ImageLaterality", "?")
        correct += pred == truth
        mark = "ok " if pred == truth else "XX "
        print(f"  {mark} patient {inputs[sop].get('PatientID', '?'):>5}  truth {truth}  pred {pred}  p={prob:.4f}")

    skipped = sorted({inputs[s].get("Modality") for s in inputs.keys() - eligible})
    print(f"\n{len(inputs)} input instances: {len(eligible)} eligible, {len(inputs) - len(eligible)} ineligible {skipped}")
    print(f"{len(report['outputs'])} SRs, all linked: {not report['violations']}")
    if predictions:
        print(f"accuracy {correct}/{len(predictions)} = {correct / len(predictions):.1%}")
    for failure in failures:
        print(f"  FAIL {failure}")
    print("PASS" if not failures else "FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
