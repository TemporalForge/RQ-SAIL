from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def predict_sequences(
    model: torch.nn.Module,
    sequences: Iterable[np.ndarray] | np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    values = np.asarray(list(sequences) if not isinstance(sequences, np.ndarray) else sequences,
                        dtype=np.float32)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start:start + batch_size], device=device)
            predictions.append(model(batch).squeeze(-1).cpu().numpy())
    return np.concatenate(predictions).astype(np.float64) if predictions else np.empty(0)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("MAE requires equally shaped, non-empty arrays")
    return float(np.mean(np.abs(truth - pred)))


def triggered_prediction_shift(clean_prediction: np.ndarray,
                               triggered_prediction: np.ndarray) -> float:
    clean = np.asarray(clean_prediction, dtype=np.float64)
    triggered = np.asarray(triggered_prediction, dtype=np.float64)
    if clean.shape != triggered.shape or clean.size == 0:
        raise ValueError("TPS requires equally shaped, paired predictions")
    return float(np.mean(np.abs(triggered - clean)))


def attack_success_rate(triggered_prediction: np.ndarray,
                        target: float = 0.90,
                        tolerance: float = 0.01) -> float:
    values = np.asarray(triggered_prediction, dtype=np.float64)
    if values.size == 0:
        raise ValueError("ASR requires at least one triggered prediction")
    return 100.0 * float(np.mean(np.abs(values - float(target)) <= float(tolerance)))


def evaluate_nasa_metrics(
    *,
    clean_truth: np.ndarray,
    old_truth: np.ndarray,
    triggered_truth: np.ndarray,
    clean_prediction: np.ndarray,
    paired_clean_prediction: np.ndarray,
    old_prediction: np.ndarray,
    triggered_prediction: np.ndarray,
    target: float = 0.90,
    tolerance: float = 0.01,
) -> dict[str, float]:
    return {
        "clean_mae": mean_absolute_error(clean_truth, clean_prediction),
        "old_mae": mean_absolute_error(old_truth, old_prediction),
        "triggered_mae": mean_absolute_error(triggered_truth, triggered_prediction),
        "tps": triggered_prediction_shift(paired_clean_prediction, triggered_prediction),
        "asr_percent": attack_success_rate(triggered_prediction, target, tolerance),
    }
