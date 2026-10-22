"""Holoscan operator: classify every instance of the selected series, in batches.

Model-agnostic. Everything model-specific comes from a `model_spec` module (see
maps/aireadi_laterality/model_spec.py for the reference one), which must provide:

    MODEL_NAME     str, model name under /opt/holoscan/models ("" = the only one)
    FINDING        Code, the concept being classified (e.g. SCT Laterality)
    CLASSES        list[Code], in the model's output index order
    CONFIDENCE     Code for the probability measurement, UNITY its unit
    preprocess(ds) -> torch.Tensor (C, H, W) for one instance
    UnsupportedImage  exception preprocess raises for instances to skip

Suits 2D single-frame classifiers, colour or not. Volumetric models should use the
SDK's DICOMSeriesToVolumeOperator instead, which is monochrome-only.
"""

import logging
from typing import List, Tuple

import torch
from monai.deploy.core import AppContext, Fragment, Operator, OperatorSpec

from map_ops.image_finding import CodedEvaluation, ImageFinding, NumericMeasurement


class InstanceClassifierOperator(Operator):
    """Named input:  study_selected_series_list (from DICOMSeriesSelectorOperator)
    Named output: findings, List[ImageFinding], one per instance scored

    One bad instance does not end the run: unsupported or undecodable instances are
    logged and skipped. The run fails only if instances were selected and none could
    be scored, which points at a systematic problem rather than a bad file.
    """

    def __init__(self, fragment: Fragment, *args, app_context: AppContext, model_spec, batch_size: int = 16, **kwargs):
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.input_name = "study_selected_series_list"
        self.output_name = "findings"
        # Not `self.spec`: that name is Holoscan's Operator.spec property.
        self.model_spec = model_spec
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not app_context.models:
            raise RuntimeError(
                f"No model under {app_context.model_path}. Package the MAP with -m <models dir>; "
                f"see the app README."
            )
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
        skipped: List[Tuple[str, str]] = []
        for start in range(0, len(instances), self.batch_size):
            tensors, kept = [], []
            for ds in instances[start : start + self.batch_size]:
                try:
                    tensors.append(self.model_spec.preprocess(ds))
                    kept.append(ds)
                except self.model_spec.UnsupportedImage as exc:
                    skipped.append((ds.SOPInstanceUID, str(exc)))
                except Exception as exc:  # undecodable pixel data and the like
                    skipped.append((ds.SOPInstanceUID, f"{type(exc).__name__}: {exc}"))
            if not tensors:
                continue

            with torch.no_grad():
                probabilities = torch.softmax(self.model(torch.stack(tensors).to(self.device)), dim=1).cpu()

            for ds, probs in zip(kept, probabilities):
                index = int(probs.argmax())
                findings.append(
                    ImageFinding(
                        source=ds,
                        evaluations=[CodedEvaluation(self.model_spec.FINDING, self.model_spec.CLASSES[index])],
                        measurements=[
                            NumericMeasurement(self.model_spec.CONFIDENCE, round(float(probs[index]), 6), self.model_spec.UNITY)
                        ],
                    )
                )
                self._logger.info(
                    f"{ds.SOPInstanceUID}: {self.model_spec.CLASSES[index].meaning} ({float(probs[index]):.4f})"
                )

        for sop, reason in skipped:
            self._logger.warning(f"skipped {sop}: {reason}")
        self._logger.info(f"scored {len(findings)}, skipped {len(skipped)}")

        if instances and not findings:
            raise RuntimeError(f"none of {len(instances)} selected instance(s) could be scored; see warnings above")
        op_output.emit(findings, self.output_name)
