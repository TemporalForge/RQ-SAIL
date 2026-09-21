#!/usr/bin/env python3
"""Evaluate raw and qualification-selected RQ-SAIL deployment."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset import BatteryDataModule
from src.metrics import evaluate_nasa_metrics, mean_absolute_error, predict_sequences
from src.moderntcn import ModernTCN, compatible_state_dict
from src.qualification import QualificationEvidence, qualify_restoration
from src.restoration import RobustLocalRepair
from src.rq_sail import RepresentationRisk, extract_embeddings
from src.runtime import embedded_model_config, get_device, load_public_config, set_seed


def _load_state(path: Path, device: torch.device) -> tuple[dict, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    return compatible_state_dict(payload), payload


def _array(frame: pd.DataFrame) -> np.ndarray:
    return np.stack(frame["sequence"].to_list()).astype(np.float32)


def _labels(frame: pd.DataFrame) -> np.ndarray:
    return frame["soh"].to_numpy(dtype=np.float64)


def _triggered_mask(clean_frame: pd.DataFrame,
                    poisoned_frame: pd.DataFrame) -> np.ndarray:
    if len(clean_frame) != len(poisoned_frame):
        raise ValueError("Clean and poisoned test frames must have equal length")
    if "poison_mask" not in poisoned_frame.columns:
        raise ValueError("Poisoned test frame does not contain poison_mask")
    identity_columns = [column for column in ("battery_id", "cycle_index")
                        if column in clean_frame.columns and column in poisoned_frame.columns]
    for column in identity_columns:
        clean_identity = clean_frame[column].reset_index(drop=True).astype(str)
        poisoned_identity = poisoned_frame[column].reset_index(drop=True).astype(str)
        if not clean_identity.equals(poisoned_identity):
            raise ValueError(f"Clean/poisoned test rows are misaligned by {column}")
    mask = poisoned_frame["poison_mask"].to_numpy(dtype=bool)
    if not np.any(mask):
        raise ValueError("Poisoned test frame contains no triggered records")
    return mask


def _restore_batch(localizer: RobustLocalRepair, sequences: np.ndarray,
                   gates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output = np.asarray(sequences, dtype=np.float32).copy()
    acted = np.zeros(len(output), dtype=bool)
    for index, (sequence, gate) in enumerate(zip(output, gates)):
        if not gate:
            continue
        result = localizer.repair(sequence)
        if result.support is not None:
            output[index] = result.repaired
            acted[index] = True
    return output, acted


def _record_gates(reference: ModernTCN, risk: RepresentationRisk,
                  threshold: float, sequences: np.ndarray,
                  localizer: RobustLocalRepair,
                  device: torch.device) -> np.ndarray:
    frame = pd.DataFrame({"sequence": list(sequences)})
    scores = risk.score(extract_embeddings(reference, frame, device))
    supports = np.asarray([localizer.locate(item) is not None for item in sequences])
    return (scores > threshold) & supports


def _triangular_probe(values: np.ndarray, relative_start: float,
                      length_ratio: float, amplitude: float,
                      channel_direction: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    sequence_length = result.shape[1]
    start = int(round(float(relative_start) * sequence_length))
    width = max(3, int(round(float(length_ratio) * sequence_length)))
    end = min(sequence_length - 1, start + width)
    if end <= start:
        raise ValueError("Qualification probe window is empty")
    direction = np.asarray(channel_direction, dtype=np.float32)[:result.shape[2]]
    if len(direction) != result.shape[2]:
        raise ValueError("Qualification probe does not define every input channel")
    center = (end - start - 1) / 2.0
    distance = np.abs(np.arange(end - start, dtype=np.float32) - center)
    envelope = (np.ones(end - start, dtype=np.float32) if center <= 0.0
                else 1.0 - distance / center)
    result[:, start:end, :] = np.clip(
        result[:, start:end, :]
        + float(amplitude) * envelope[None, :, None] * direction[None, None, :],
        0.0,
        1.0,
    )
    return result


def _restoration_fraction(before: float, after: float) -> float:
    return 0.0 if before <= 1.0e-12 else float(1.0 - after / before)


def _stress_evidence(model: ModernTCN, reference: ModernTCN, risk: RepresentationRisk,
                     threshold: float, holdout: pd.DataFrame,
                     localizer: RobustLocalRepair, config: dict,
                     device: torch.device,
                     proposed_action_rate: float) -> QualificationEvidence:
    qualification = config["qualification"]
    holdout_values = _array(holdout)
    clean_prediction = predict_sequences(model, holdout_values, device)
    starts = qualification["probes"]["relative_starts"]
    length_ratio = float(qualification["probes"]["length_ratio"])
    direction = np.asarray(qualification["probes"]["channel_direction"], dtype=np.float32)
    amplitude = float(qualification["probes"]["amplitude"])
    action_fractions, input_fractions, prediction_fractions = [], [], []
    for relative_start in starts:
        stressed = _triangular_probe(
            holdout_values,
            float(relative_start),
            length_ratio,
            amplitude,
            direction,
        )
        gates = _record_gates(reference, risk, threshold, stressed, localizer, device)
        repaired, acted = _restore_batch(localizer, stressed, gates)
        stressed_prediction = predict_sequences(model, stressed, device)
        repaired_prediction = predict_sequences(model, repaired, device)
        input_before = float(np.mean(np.abs(stressed - holdout_values)))
        input_after = float(np.mean(np.abs(repaired - holdout_values)))
        prediction_before = float(np.mean(np.abs(stressed_prediction - clean_prediction)))
        prediction_after = float(np.mean(np.abs(repaired_prediction - clean_prediction)))
        action_fractions.append(float(acted.mean()))
        input_fractions.append(_restoration_fraction(input_before, input_after))
        prediction_fractions.append(_restoration_fraction(prediction_before, prediction_after))

    return QualificationEvidence(
        holdout_mae=mean_absolute_error(_labels(holdout), clean_prediction),
        minimum_action_fraction=min(action_fractions),
        minimum_input_restoration_fraction=min(input_fractions),
        minimum_tps_restoration_fraction=min(prediction_fractions),
        proposed_action_rate=float(proposed_action_rate),
    )


def evaluate(config_or_path, checkpoint: str | None = None,
             output: str | None = None) -> list[dict]:
    config = load_public_config(config_or_path)
    set_seed(int(config["experiment"]["seed"]))
    device = get_device(True)
    module = BatteryDataModule(config)
    module.load_poisoned_data(config["data"]["poisoned_metadata_path"])
    model = ModernTCN(embedded_model_config(config)).to(device)
    reference = ModernTCN(embedded_model_config(config)).to(device)
    checkpoint_path = Path(checkpoint or Path(config["checkpoint"]["save_dir"]) / "best_model.pth")
    updated_state, payload = _load_state(checkpoint_path, device)
    model.load_state_dict(updated_state)
    source_state, _ = _load_state(Path(config["model"]["load_task1_ckpt"]), device)
    reference.load_state_dict(source_state); reference.eval()

    trusted_frame = pd.concat([module.val_data, module.holdout_data], ignore_index=True)
    if "representation_risk" in payload:
        risk = RepresentationRisk.from_state_dict(payload["representation_risk"])
    else:
        rep_cfg = config["rq_sail"]["representation"]
        risk = RepresentationRisk.fit(extract_embeddings(reference, trusted_frame, device),
                                      float(rep_cfg["normalization_percentile"]), float(rep_cfg["epsilon"]))
    threshold_key = "sample_threshold" if "sample_threshold" in payload else "sample_tau"
    threshold = float(payload[threshold_key])
    localizer = RobustLocalRepair(_array(trusted_frame), **config["qualification"]["localizer"])
    deployment_frame = module.test_data_poisoned
    deployment_values = _array(deployment_frame)
    proposed_gate = _record_gates(
        reference, risk, threshold, deployment_values, localizer, device,
    )
    evidence = _stress_evidence(
        model, reference, risk, threshold, module.holdout_data,
        localizer, config, device, float(proposed_gate.mean()),
    )
    decision = qualify_restoration(
        evidence,
        maximum_holdout_mae=float(config["qualification"]["holdout_mae_max"]),
        minimum_stress_score=float(config["qualification"]["stress_score_min"]),
        maximum_proposed_action_rate=float(config["qualification"]["proposed_action_rate_max"]),
    )

    with (Path(config["data"]["processed_data_dir"]) / "metadata_task1.pkl").open("rb") as handle:
        old_frame = pickle.load(handle)["test_old"]
    clean_frame = module.test_data_clean
    triggered_frame = deployment_frame
    triggered_mask = _triggered_mask(clean_frame, triggered_frame)
    clean_values, old_values = map(_array, (clean_frame, old_frame))
    triggered_values = deployment_values
    clean_prediction = predict_sequences(model, clean_values, device)
    old_prediction = predict_sequences(model, old_values, device)
    raw_triggered_prediction = predict_sequences(model, triggered_values, device)
    repaired, acted = _restore_batch(localizer, triggered_values,
                                     proposed_gate if decision.enabled else np.zeros_like(proposed_gate))
    full_triggered_prediction = predict_sequences(model, repaired, device)
    target = config["attack"]["target"]
    shared = dict(clean_truth=_labels(clean_frame), old_truth=_labels(old_frame),
                  triggered_truth=_labels(clean_frame)[triggered_mask],
                  clean_prediction=clean_prediction,
                  paired_clean_prediction=clean_prediction[triggered_mask],
                  old_prediction=old_prediction, target=float(target["y_target"]),
                  tolerance=float(target["epsilon_asr"]))
    rows = []
    for variant, all_predictions in (("RQ-SAIL (raw)", raw_triggered_prediction),
                                     ("RQ-SAIL", full_triggered_prediction)):
        triggered_prediction = all_predictions[triggered_mask]
        row = {"attack": config["attack"]["type"], "method": variant,
               **evaluate_nasa_metrics(triggered_prediction=triggered_prediction, **shared),
               "qualification_enabled": decision.enabled,
               "qualification_reason": decision.reason,
               "holdout_mae": evidence.holdout_mae,
               "minimum_action_fraction": evidence.minimum_action_fraction,
               "minimum_input_restoration_fraction": evidence.minimum_input_restoration_fraction,
               "minimum_tps_restoration_fraction": evidence.minimum_tps_restoration_fraction,
               "minimum_stress_score": min(evidence.minimum_action_fraction,
                                             evidence.minimum_input_restoration_fraction,
                                             evidence.minimum_tps_restoration_fraction),
               "proposed_action_rate": evidence.proposed_action_rate,
               "test_action_rate": float(acted.mean()) if variant == "RQ-SAIL" else 0.0,
               "triggered_records": int(triggered_mask.sum()),
               "total_test_records": int(len(triggered_mask))}
        rows.append(row)
    output_path = Path(output or checkpoint_path.parent.parent / "evaluation.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"qualification={decision.enabled} ({decision.reason})")
    print(f"results={output_path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nasa_bt1.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
