"""Phase 2 CLI: run one MAP against one local file.

    python -m shadow_mode.run_map AIREADI_Laterality_App/demo_input/<file>.dcm

No Orthanc, no eligibility, no provenance — this exists to prove
`local DICOM -> MAP -> output` and to give the later layers something stable to
call.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from shadow_mode.config import MAPS
from shadow_mode.execution import BACKENDS, DEFAULT_BACKEND, MapExecutionError, get_executor

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _default_output_dir(map_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RUNS_DIR / f"{map_name}-{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shadow_mode.run_map",
        description="Execute a MONAI Application Package against a single local file.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Input file, or a directory holding exactly one file the MAP accepts.",
    )
    parser.add_argument("--map", default="laterality", help="Registered MAP name (default: laterality).")
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=sorted(BACKENDS),
        help=f"Execution backend (default: {DEFAULT_BACKEND}).",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output directory (default: shadow_mode/runs/<map>-<ts>).")
    parser.add_argument("--gpus", default="all", help="Value for --gpus; pass 'none' to run CPU-only.")
    parser.add_argument("--bin", help="Override the backend executable path.")
    parser.add_argument("--list", action="store_true", help="List registered MAPs and exit.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Echo MAP stdout/stderr.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.list:
        for name, spec in sorted(MAPS.items()):
            accepts = "/".join(spec.input_suffixes)
            print(f"{name:20} {spec.image}  (accepts {accepts}, gpu={spec.gpu})")
        return 0

    if args.input is None:
        build_parser().error("an input path is required (or use --list)")

    output_dir = args.output or _default_output_dir(args.map)

    kwargs = {"gpus": None if args.gpus == "none" else args.gpus}
    if args.bin:
        kwargs["docker_bin" if args.backend == "docker" else "monai_deploy_bin"] = args.bin

    try:
        executor = get_executor(args.backend, **kwargs)
        result = executor.run(args.map, args.input, output_dir)
    except MapExecutionError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        if args.verbose and exc.result.stdout:
            print(f"\n--- stdout ---\n{exc.result.stdout}", file=sys.stderr)
        return 1
    except (FileNotFoundError, KeyError, PermissionError, ValueError) as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        if result.stdout:
            print(f"--- stdout ---\n{result.stdout}")
        if result.stderr:
            print(f"--- stderr ---\n{result.stderr}")

    print(f"\n{result.summary()}")
    for artifact in result.artifacts:
        print(f"  {artifact}")

    # output.json is the classifier's scalar result; showing it turns the run
    # into an answer rather than a list of files.
    predictions = output_dir / "output.json"
    if predictions.exists():
        try:
            print(f"\nprediction: {json.loads(predictions.read_text())}")
        except json.JSONDecodeError:
            print(f"\nprediction: <unparseable> {predictions.read_text()[:200]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
