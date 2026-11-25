"""Holoscan operator: locate structures in every selected instance, one image at a time.

The unit of work is the instance. The selector hands over one entry per study, each with the series
that passed SELECTION_RULES; this operator flattens them into one list, across every study and series.

Model-agnostic. Everything model-specific comes from a `model_spec` module (see
maps/disc_detection/model_spec.py for the reference one), which must provide:

    MODEL_NAME     str, model folder under /opt/holoscan/models, holding exactly one .pt file
    MODEL_SHA256   str, that file's SHA-256; any other weights stop the MAP (see checked_model_file)
    detect(model, ds, device) -> ImageFinding for one instance (boxes and/or not_found)
    UnsupportedImage  exception detect raises for instances to skip

Same skip/error rules as InstanceClassifierOperator in maps/aireadi_laterality.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Tuple

import torch
from monai.deploy.core import AppContext, Fragment, Operator, OperatorSpec

from map_ops.image_finding import ImageFinding


def checked_model_file(folder: Path, expected_sha256: str) -> Path:
    """The one .pt file in folder, refusing it unless its SHA-256 is the expected one.

    SR UIDs depend on ALGORITHM_VERSION, not on the weights, so running other weights under the
    same version would give different boxes under the same UIDs. This makes that impossible:
    new weights need a new MODEL_SHA256 in model_spec.py, edited together with the version.
    """
    files = sorted(Path(folder).glob("*.pt"))
    if len(files) != 1:
        raise RuntimeError(f"expected exactly one .pt file in {folder}, found {[f.name for f in files]}")
    digest = hashlib.sha256()
    with open(files[0], "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError(
            f"{files[0].name} has SHA-256 {digest.hexdigest()}, model_spec expects {expected_sha256}. "
            f"New weights: update MODEL_SHA256 and bump ALGORITHM_VERSION in model_spec.py."
        )
    return files[0]


class InstanceDetectorOperator(Operator):
    """Named input:  study_selected_series_list (from DICOMSeriesSelectorOperator)
    Named output: findings, List[ImageFinding], one per instance processed, all studies

    Instances the model isn't for (spec.UnsupportedImage) are skipped quietly; a series of only
    those finishes with no findings, which is a normal outcome. Undecodable instances are errors:
    logged and skipped, and the run fails only if errors left nothing processed.
    """

    def __init__(self, fragment: Fragment, *args, app_context: AppContext, model_spec, **kwargs):
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.input_name = "study_selected_series_list"
        self.output_name = "findings"
        # Not `self.spec`: that name is Holoscan's Operator.spec property.
        self.model_spec = model_spec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not app_context.models:
            raise RuntimeError(
                f"No model under {app_context.model_path}. Package the MAP with -m <models dir>; "
                f"see the app README."
            )
        checked_model_file(Path(app_context.model_path) / model_spec.MODEL_NAME, model_spec.MODEL_SHA256)
        self.model = app_context.models.get(model_spec.MODEL_NAME)
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name)
        spec.output(self.output_name)

    def compute(self, op_input, op_output, context):
        selected = op_input.receive(self.input_name) or []
        instances = [
            sop.get_native_sop_instance()
            for study_selection in selected
            for selection in study_selection.selected_series
            for sop in selection.series.get_sop_instances()
        ]
        self._logger.info(f"{len(instances)} instance(s) selected across {len(selected)} study(ies), on {self.device}")

        findings: List[ImageFinding] = []
        skipped: List[Tuple[str, str]] = []  # not for this model: fine
        errors: List[Tuple[str, str]] = []  # couldn't read: a problem
        for ds in instances:
            try:
                finding = self.model_spec.detect(self.model, ds, self.device)
            except self.model_spec.UnsupportedImage as exc:
                skipped.append((ds.SOPInstanceUID, str(exc)))
                continue
            except Exception as exc:  # undecodable pixel data, no fundus circle found and the like
                errors.append((ds.SOPInstanceUID, f"{type(exc).__name__}: {exc}"))
                continue
            findings.append(finding)
            found = [f"{b.finding_type.meaning} {b.measurements[0].value if b.measurements else ''}" for b in finding.boxes]
            missing = [f"{c.meaning} absent" for c in finding.not_found]
            self._logger.info(f"{ds.SOPInstanceUID}: {', '.join(found + missing)}")

        for sop, reason in skipped:
            self._logger.info(f"skipped {sop}: {reason}")
        for sop, reason in errors:
            self._logger.warning(f"error on {sop}: {reason}")
        self._logger.info(f"processed {len(findings)}, skipped {len(skipped)}, errors {len(errors)}")

        if errors and not findings:
            raise RuntimeError(f"none of {len(instances)} selected instance(s) could be processed; see warnings above")
        op_output.emit(findings, self.output_name)
