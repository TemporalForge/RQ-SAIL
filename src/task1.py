from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .dataset import BatteryDataModule
from .moderntcn import ModernTCN
from .runtime import embedded_model_config, get_device, load_public_config, set_seed


class PhysicsTeacherMLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        dimensions = [int(config["input_dim"]),
                      *[int(value) for value in config["hidden_dims"]],
                      int(config["output_dim"])]
        activation = nn.GELU if str(config["activation"]).lower() == "gelu" else nn.ReLU
        layers: list[nn.Module] = []
        for index in range(len(dimensions) - 2):
            layers.extend([nn.Linear(dimensions[index], dimensions[index + 1]), activation()])
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()


def _validation_mse(model: torch.nn.Module, loader: DataLoader,
                    device: torch.device) -> float:
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


def _monotonicity_loss(predictions: torch.Tensor, battery_ids,
                       cycle_indices, margin: float) -> torch.Tensor | None:
    battery_list = list(battery_ids)
    losses: list[torch.Tensor] = []
    for battery in dict.fromkeys(battery_list):
        indices = [index for index, value in enumerate(battery_list) if value == battery]
        if len(indices) < 2:
            continue
        ordered = sorted(indices, key=lambda index: int(cycle_indices[index]))
        values = predictions[ordered]
        losses.append(torch.relu(values[1:] - values[:-1] + margin).mean())
    return torch.stack(losses).mean() if losses else None


def _train_physics_teacher(model: PhysicsTeacherMLP, frame,
                           config: dict, device: torch.device) -> None:
    valid = frame.copy()
    valid = valid[valid["ic_valid"].fillna(False).astype(bool)]
    valid = valid.dropna(subset=["vp1", "hp1"])
    valid = valid[~((valid["vp1"].abs() < 1.0e-10)
                    & (valid["hp1"].abs() < 1.0e-10))]
    if len(valid) == 0:
        raise ValueError("Task-1 physics-teacher training requires valid IC features")
    features = torch.tensor(valid[["vp1", "hp1"]].to_numpy(dtype=np.float32))
    labels = torch.tensor(valid["soh"].to_numpy(dtype=np.float32)).unsqueeze(1)
    loader = DataLoader(TensorDataset(features, labels), batch_size=32, shuffle=True)
    settings = config["physics_teacher"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))
    model.train()
    for _ in range(int(settings["epochs"])):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch_features.to(device)), batch_labels.to(device))
            loss.backward()
            optimizer.step()
    if bool(settings.get("freeze_after_training", True)):
        model.freeze()


def train_task1(config_or_path) -> dict:
    config = load_public_config(config_or_path)
    set_seed(int(config["experiment"]["seed"]))
    device = get_device(True)

    module = BatteryDataModule(config)
    module.load_processed_data("task1")
    train_loader, validation_loader, _ = module.get_dataloaders(
        num_workers=int(config["data"].get("num_workers", 0))
    )

    model = ModernTCN(embedded_model_config(config)).to(device)
    teacher = PhysicsTeacherMLP(config["physics_teacher"]).to(device)
    _train_physics_teacher(teacher, module.train_data, config, device)

    settings = config["task1"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    epochs = int(settings["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(settings["scheduler_eta_min"])
    )
    best_mse = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            features = batch["features"].to(device)
            target = batch["label"].to(device).squeeze(-1)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features).squeeze(-1)
            loss = F.mse_loss(prediction, target)

            valid = batch["ic_valid"].to(device).bool()
            if torch.any(valid):
                with torch.no_grad():
                    physics_prediction = teacher(
                        batch["ic_features"].to(device)[valid]
                    ).squeeze(-1)
                loss = loss + float(settings["lambda_phys"]) * F.mse_loss(
                    prediction[valid], physics_prediction
                )
            monotonicity = _monotonicity_loss(
                prediction, batch["battery_id"], batch["cycle_index"],
                float(settings["mono_margin"]),
            )
            if monotonicity is not None:
                loss = loss + float(settings["lambda_mono"]) * monotonicity

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(settings["gradient_clip_norm"])
            )
            optimizer.step()

        scheduler.step()
        validation_mse = _validation_mse(model, validation_loader, device)
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())

    checkpoint_directory = Path(settings["checkpoint_dir"])
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_directory / "best_model.pth"
    teacher_path = checkpoint_directory / "best_physics_teacher.pth"
    torch.save({
        "model_state_dict": best_state,
        "optimizer_state_dict": best_optimizer_state,
        "epoch": best_epoch,
        "metrics": {"val_loss": best_mse},
    }, model_path)
    torch.save({
        "model_state_dict": teacher.state_dict(),
        "config": config["physics_teacher"],
    }, teacher_path)
    return {
        "checkpoint": str(model_path),
        "physics_teacher_checkpoint": str(teacher_path),
        "validation_mse": best_mse,
    }
