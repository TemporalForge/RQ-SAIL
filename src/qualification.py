from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualificationEvidence:
    holdout_mae: float
    minimum_action_fraction: float
    minimum_input_restoration_fraction: float
    minimum_tps_restoration_fraction: float
    proposed_action_rate: float


@dataclass(frozen=True)
class QualificationDecision:
    enabled: bool
    reason: str


def qualify_restoration(
    evidence: QualificationEvidence,
    *,
    maximum_holdout_mae: float = 0.04,
    minimum_stress_score: float = 0.50,
    maximum_proposed_action_rate: float = 0.25,
) -> QualificationDecision:
    checks = (
        (evidence.holdout_mae <= maximum_holdout_mae,
         "holdout_mae_exceeds_limit"),
        (evidence.minimum_action_fraction >= minimum_stress_score,
         "stress_action_fraction_below_limit"),
        (evidence.minimum_input_restoration_fraction >= minimum_stress_score,
         "stress_input_restoration_below_limit"),
        (evidence.minimum_tps_restoration_fraction >= minimum_stress_score,
         "stress_prediction_restoration_below_limit"),
        (evidence.proposed_action_rate <= maximum_proposed_action_rate,
         "proposed_action_rate_exceeds_limit"),
    )
    for passed, reason in checks:
        if not passed:
            return QualificationDecision(False, reason)
    return QualificationDecision(True, "qualified")
