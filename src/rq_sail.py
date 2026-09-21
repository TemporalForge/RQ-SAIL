from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader

from .dataset import BatteryDataModule
from .moderntcn import ModernTCN, compatible_state_dict
from .runtime import embedded_model_config, get_device, load_public_config, set_seed


def _state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        return compatible_state_dict(checkpoint)
    raise TypeError("Unsupported checkpoint payload")


def _sequences(frame: pd.DataFrame) -> np.ndarray:
    if frame is None or len(frame) == 0:
        raise ValueError("A non-empty sequence dataframe is required")
    return np.stack(frame["sequence"].to_list()).astype(np.float32)


def extract_embeddings(model: ModernTCN, frame: pd.DataFrame, device: torch.device,
                       batch_size: int = 64) -> np.ndarray:
    values = _sequences(frame)
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start:start + batch_size], device=device)
            output.append(model.get_feature_embedding(batch).cpu().numpy())
    return np.concatenate(output)


@dataclass
class RepresentationRisk:
    center: np.ndarray
    precision: np.ndarray
    reference_scale: float
    epsilon: float = 1.0e-6

    @classmethod
    def fit(cls, features: np.ndarray, percentile: float = 99.0,
            epsilon: float = 1.0e-6) -> "RepresentationRisk":
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("Representation fitting requires at least two samples")
        covariance = LedoitWolf().fit(values)
        center = np.mean(values, axis=0)
        try:
            precision = np.linalg.inv(covariance.covariance_)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(covariance.covariance_)
        delta = values - center
        distance = np.sqrt(np.maximum(np.sum((delta @ precision) * delta, axis=1), 0.0))
        scale = float(np.percentile(distance, percentile) + epsilon)
        return cls(center, precision, scale, epsilon)

    def score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        delta = values - self.center
        distance = np.sqrt(np.maximum(np.sum((delta @ self.precision) * delta, axis=1), 0.0))
        return (distance / (distance + self.reference_scale)).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "center": self.center,
            "precision": self.precision,
            "reference_scale": self.reference_scale,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "RepresentationRisk":
        return cls(np.asarray(state["center"]), np.asarray(state["precision"]),
                   float(state["reference_scale"]), float(state.get("epsilon", 1.0e-6)))


def _topk_mean(values: np.ndarray, ratio: float, minimum: int) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    count = min(len(flat), max(int(minimum), int(math.ceil(len(flat) * ratio))))
    return float(np.partition(flat, len(flat) - count)[-count:].mean())


def _windows(frame: pd.DataFrame, size: int, stride: int) -> Iterable[pd.DataFrame]:
    ordered = frame.sort_values(["battery_id", "cycle_index"])
    for _, group in ordered.groupby("battery_id"):
        group = group.reset_index(drop=True)
        if len(group) < size:
            if len(group):
                yield group
            continue
        for start in range(0, len(group) - size + 1, stride):
            yield group.iloc[start:start + size]


def _freeze_to_head(model: ModernTCN) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = model.get_last_layer()
    head.weight.requires_grad = True
    head.bias.requires_grad = True
    return head.weight, head.bias


def _validation_mse(model: ModernTCN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["features"].to(device)).squeeze(-1)
            target = batch["label"].to(device).squeeze(-1)
            losses.append(float(F.mse_loss(prediction, target)))
    if not losses:
        raise ValueError("Validation loader is empty")
    return float(np.mean(losses))


def _load_data(config: dict) -> BatteryDataModule:
    module = BatteryDataModule(config)
    path = Path(config["data"]["poisoned_metadata_path"])
    if not path.exists():
        raise FileNotFoundError(f"Missing poisoned metadata: {path}. Run the poison stage first.")
    module.load_poisoned_data(str(path))
    return module


def train_rq_sail(config_or_path) -> dict:
    config = load_public_config(config_or_path)
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    device = get_device(prefer_gpu=True)
    module = _load_data(config)
    train_loader, validation_loader, _ = module.get_dataloaders(
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model_config = embedded_model_config(config)
    model = ModernTCN(model_config).to(device)
    reference = ModernTCN(model_config).to(device)
    checkpoint_path = Path(config["model"]["load_task1_ckpt"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Task-1 checkpoint: {checkpoint_path}")
    source_state = _state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.load_state_dict(source_state)
    reference.load_state_dict(source_state)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    weight, bias = _freeze_to_head(model)

    trusted = pd.concat([module.val_data, module.holdout_data], ignore_index=True)
    rq = config["rq_sail"]
    rep_cfg = rq["representation"]
    risk = RepresentationRisk.fit(
        extract_embeddings(reference, trusted, device),
        percentile=float(rep_cfg["normalization_percentile"]),
        epsilon=float(rep_cfg["epsilon"]),
    )

    sample_cfg = rq["sample_gate"]
    train_scores = risk.score(extract_embeddings(reference, module.train_data, device))
    sample_quantile = float(np.clip(
        1.0 - float(sample_cfg["risk_budget"]) - float(sample_cfg["safety_margin"]),
        float(sample_cfg["min_quantile"]), float(sample_cfg["max_quantile"]),
    ))
    sample_threshold = float(np.quantile(train_scores, sample_quantile))

    batch_cfg = rq["batch_gate"]
    clean_batch_scores: list[float] = []
    for window in _windows(trusted, int(batch_cfg["window_size"]), int(batch_cfg["stride"])):
        scores = risk.score(extract_embeddings(reference, window, device))
        representation = _topk_mean(scores, float(batch_cfg["topk_ratio"]),
                                    int(batch_cfg["minimum_topk"]))
        clean_batch_scores.append(
            float(batch_cfg["representation_weight"]) * representation
            + float(batch_cfg["gradient_weight"]) * 0.5
        )
    if not clean_batch_scores:
        raise ValueError("No clean calibration windows were available")
    batch_threshold = float(np.quantile(clean_batch_scores,
                                        float(batch_cfg["calibration_quantile"])))

    optimizer_cfg = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW([weight, bias], lr=float(optimizer_cfg["lr"]),
                                  weight_decay=float(optimizer_cfg["weight_decay"]))
    epochs = int(config["training"]["epochs"])
    best_mse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    reference_weight = float(rq["reference_weight"])
    eps = float(batch_cfg["gradient_epsilon"])

    if device.type == "cuda":
        torch.cuda.synchronize()
    core_started = time.perf_counter()
    for _epoch in range(epochs):
        model.train()
        for batch in train_loader:
            features = batch["features"].to(device)
            target = batch["label"].to(device).squeeze(-1)
            with torch.no_grad():
                reference_prediction = reference(features).squeeze(-1)
                embedding = reference.get_feature_embedding(features).cpu().numpy()
                score_tensor = torch.as_tensor(risk.score(embedding), device=device)
                clean_weight = torch.sigmoid(
                    (sample_threshold - score_tensor) / float(sample_cfg["temperature"])
                ).clamp_min(float(sample_cfg["minimum_clean_weight"]))
                suspicious_weight = 1.0 - clean_weight
                label_weight = clean_weight + float(sample_cfg["suspicious_label_weight"]) * suspicious_weight

            prediction = model(features).squeeze(-1)
            label_loss = torch.mean(label_weight * (prediction - target).square())
            consistency_loss = torch.mean(suspicious_weight * (prediction - reference_prediction).square())
            grad_label_w = torch.autograd.grad(label_loss, weight, retain_graph=True)[0]
            grad_reference_w = torch.autograd.grad(
                reference_weight * consistency_loss, weight, retain_graph=True,
                allow_unused=True,
            )[0]
            grad_label_flat = grad_label_w.flatten()
            grad_reference_flat = (
                torch.zeros_like(grad_label_flat)
                if grad_reference_w is None
                else grad_reference_w.flatten()
            )
            norm_label = torch.norm(grad_label_flat)
            norm_reference = torch.norm(grad_reference_flat)
            if norm_label.item() < eps or norm_reference.item() < eps:
                gradient_conflict = 0.5
            else:
                cosine = torch.dot(grad_label_flat, grad_reference_flat) / (
                    norm_label * norm_reference + eps
                )
                cosine = torch.clamp(cosine, min=-1.0, max=1.0)
                gradient_conflict = float(((1.0 - cosine) / 2.0).detach())
            representation = _topk_mean(score_tensor.detach().cpu().numpy(),
                                        float(batch_cfg["topk_ratio"]),
                                        int(batch_cfg["minimum_topk"]))
            anomaly = (float(batch_cfg["representation_weight"]) * representation
                       + float(batch_cfg["gradient_weight"]) * gradient_conflict)
            gate_input = (batch_threshold - anomaly) / float(batch_cfg["temperature"])
            gate = float(1.0 / (1.0 + math.exp(-gate_input)))
            grad_weight = torch.autograd.grad(label_loss, weight, retain_graph=True)[0]
            grad_bias = torch.autograd.grad(
                label_loss + reference_weight * consistency_loss, bias
            )[0]
            optimizer.zero_grad(set_to_none=True)
            weight.grad = gate * grad_weight
            bias.grad = gate * grad_bias
            optimizer.step()
        score = _validation_mse(model, validation_loader, device)
        if score < best_mse:
            best_mse = score
            best_state = copy.deepcopy(model.state_dict())

    if device.type == "cuda":
        torch.cuda.synchronize()
    core_update_time = time.perf_counter() - core_started
    model.load_state_dict(best_state)
    save_dir = Path(config["checkpoint"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "best_model.pth"
    total = sum(parameter.numel() for parameter in model.parameters())
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "validation_mse": best_mse,
        "sample_threshold": sample_threshold,
        "batch_threshold": batch_threshold,
        "representation_risk": risk.state_dict(),
        "trainable_parameters": weight.numel() + bias.numel(),
        "total_parameters": total,
        "core_update_time_s": core_update_time,
    }, output_path)
    return {
        "checkpoint": str(output_path),
        "validation_mse": best_mse,
        "sample_threshold": sample_threshold,
        "batch_threshold": batch_threshold,
        "trainable_parameters": weight.numel() + bias.numel(),
        "total_parameters": total,
        "core_update_time_s": core_update_time,
    }
