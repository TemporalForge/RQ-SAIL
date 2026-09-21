from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.signal import find_peaks, savgol_filter
from torch.utils.data import DataLoader, Dataset

from .runtime import load_public_config


class NASABatteryLoader:
    def __init__(self, config: dict):
        self.raw_directory = Path(config["data"]["raw_data_dir"])

    @staticmethod
    def _unwrap(value):
        while isinstance(value, np.ndarray) and value.size == 1:
            value = value.reshape(-1)[0]
        return value

    def load(self, battery_id: str) -> list[dict]:
        path = self.raw_directory / f"{battery_id}.mat"
        if not path.exists():
            raise FileNotFoundError(f"Missing NASA battery file: {path}")
        root = loadmat(path)[battery_id][0, 0]["cycle"][0]
        cycles: list[dict] = []
        for index, item in enumerate(root):
            try:
                kind = str(self._unwrap(item["type"])).lower()
                if kind not in {"charge", "discharge", "impedance"}:
                    continue
                data = item["data"][0, 0]
                record = {"cycle_index": index, "type": kind}
                if kind == "discharge":
                    record["capacity"] = float(self._unwrap(data["Capacity"]))
                if kind in {"charge", "discharge"}:
                    record["voltage"] = np.asarray(data["Voltage_measured"][0], dtype=np.float64)
                    record["current"] = np.asarray(data["Current_measured"][0], dtype=np.float64)
                    record["time"] = np.asarray(data["Time"][0], dtype=np.float64)
                    if "Temperature_measured" in data.dtype.names:
                        record["temperature"] = np.asarray(data["Temperature_measured"][0], dtype=np.float64)
                    else:
                        record["temperature"] = np.full_like(record["time"], np.nan)
                    length = len(record["time"])
                    if length < 10 or len(record["voltage"]) != length or len(record["current"]) != length:
                        continue
                    if not np.all(np.diff(record["time"]) > 0):
                        continue
                    if len(record["temperature"]) != length:
                        record["temperature"] = np.full(length, np.nan)
                cycles.append(record)
            except Exception as error:
                warnings.warn(f"Skipping {battery_id} cycle {index}: {error}")
        return cycles


class SequencePreprocessor:
    def __init__(self, config: dict):
        settings = config["preprocessing"]
        normalization = settings["normalization"]
        self.length = int(settings["sequence_length"])
        self.voltage_window = settings["voltage_window"]
        self.ranges = (normalization["voltage_range"], normalization["current_range"],
                       normalization["temperature_range"])

    @staticmethod
    def _resample(values: np.ndarray, length: int) -> np.ndarray:
        if len(values) == length:
            return values.copy()
        source = np.linspace(0.0, 1.0, len(values))
        target = np.linspace(0.0, 1.0, length)
        return interp1d(source, values, bounds_error=False, fill_value=np.nan)(target)

    @staticmethod
    def _normalize(values: np.ndarray, bounds: list[float]) -> np.ndarray:
        low, high = map(float, bounds)
        return np.where(np.isnan(values), 0.5, np.clip((values - low) / (high - low), 0.0, 1.0))

    def transform(self, charge: dict) -> Optional[np.ndarray]:
        voltage = charge["voltage"]
        mask = ((voltage >= float(self.voltage_window["min"]))
                & (voltage <= float(self.voltage_window["max"])))
        if int(mask.sum()) < 10:
            return None
        channels = [charge["voltage"][mask], charge["current"][mask], charge["temperature"][mask]]
        output = [self._normalize(self._resample(channel, self.length), bounds)
                  for channel, bounds in zip(channels, self.ranges)]
        return np.stack(output, axis=1).astype(np.float32)


class ICFeatureExtractor:
    def __init__(self, config: dict):
        settings = config["ic_extraction"]
        self.delta_v_min = float(settings["delta_v_min"])
        self.min_points = int(settings["min_points"])
        self.voltage_upper_cutoff = float(settings["voltage_upper_cutoff"])
        self.peak_voltage_min = float(settings["peak_voltage_min"])
        self.peak_voltage_max = float(settings["peak_voltage_max"])
        self.sg_window = int(settings["sg_window"])
        self.sg_polyorder = int(settings["sg_polyorder"])

    def extract(self, voltage: np.ndarray, current: np.ndarray,
                time: np.ndarray) -> Optional[tuple[float, float]]:
        voltage = np.asarray(voltage, dtype=np.float64).reshape(-1)
        current = np.asarray(current, dtype=np.float64).reshape(-1)
        time = np.asarray(time, dtype=np.float64).reshape(-1)
        if (len(voltage) < self.min_points or len(current) != len(voltage)
                or len(time) != len(voltage) or not np.all(np.diff(time) > 0)
                or np.isnan(voltage).all() or np.isnan(current).all()):
            return None

        delta_time = np.diff(time, prepend=time[0]) / 3600.0
        voltage_points: list[float] = []
        capacity_gradients: list[float] = []
        for start in range(len(voltage) - 1):
            end = start + 1
            while (end < len(voltage)
                   and abs(voltage[end] - voltage[start]) < self.delta_v_min):
                end += 1
            if end >= len(voltage):
                break
            delta_voltage = abs(voltage[end] - voltage[start])
            if delta_voltage < 1.0e-10:
                continue
            segment_current = current[start:end + 1]
            midpoint_current = segment_current[:-1] + 0.5 * np.diff(segment_current)
            delta_capacity = float(np.sum(np.abs(midpoint_current)
                                          * delta_time[start:end + 1][1:]))
            if abs(delta_capacity) > 1.0e-8 and delta_voltage > 1.0e-8:
                voltage_points.append(0.5 * (voltage[start] + voltage[end]))
                capacity_gradients.append(delta_capacity / delta_voltage)

        if len(voltage_points) < 5:
            return None
        voltage_array = np.asarray(voltage_points)
        gradient_array = np.asarray(capacity_gradients)
        valid = voltage_array <= self.voltage_upper_cutoff
        if int(valid.sum()) < 5:
            return None
        voltage_array = voltage_array[valid]
        gradient_array = gradient_array[valid]
        unique_voltage, inverse = np.unique(np.round(voltage_array, 6),
                                            return_inverse=True)
        merged = np.zeros(len(unique_voltage))
        counts = np.zeros(len(unique_voltage))
        for index, group in enumerate(inverse):
            merged[group] += gradient_array[index]
            counts[group] += 1
        merged /= counts

        if len(unique_voltage) < self.min_points // 2:
            return None

        if len(merged) >= 7:
            window = min(self.sg_window, len(merged) - 1)
            if window % 2 == 0:
                window -= 1
            window = max(window, 5)
            order = min(self.sg_polyorder, window - 1)
            try:
                merged = savgol_filter(merged, window_length=window,
                                        polyorder=order)
            except Exception:
                pass

        peaks, properties = find_peaks(merged, height=0)
        if not len(peaks):
            peak = int(np.argmax(merged))
        else:
            peak_voltage = unique_voltage[peaks]
            in_range = ((peak_voltage >= self.peak_voltage_min)
                        & (peak_voltage <= self.peak_voltage_max))
            if np.any(in_range):
                candidates = peaks[in_range]
                heights = properties["peak_heights"][in_range]
                peak = int(candidates[int(np.argmax(heights))])
            else:
                peak = int(peaks[int(np.argmax(properties["peak_heights"]))])
        return float(unique_voltage[peak]), float(merged[peak])


def _battery_frame(battery_id: str, cycles: list[dict], preprocessor: SequencePreprocessor,
                   ic_extractor: ICFeatureExtractor, initial_cycles: int) -> pd.DataFrame:
    capacities = [cycle["capacity"] for cycle in cycles
                  if cycle["type"] == "discharge" and np.isfinite(cycle.get("capacity", np.nan))]
    if len(capacities) < initial_cycles:
        raise ValueError(f"{battery_id} has fewer than {initial_cycles} valid discharge cycles")
    initial_capacity = float(np.mean(capacities[:initial_cycles]))
    rows: list[dict] = []
    for index, charge in enumerate(cycles):
        if charge["type"] != "charge":
            continue
        discharge = None
        for future in cycles[index + 1:]:
            if future["type"] == "charge":
                break
            if future["type"] == "discharge":
                discharge = future
                break
        if discharge is None or "capacity" not in discharge:
            continue
        sequence = preprocessor.transform(charge)
        soh = float(discharge["capacity"] / initial_capacity)
        if sequence is None or not 0.0 < soh <= 1.2:
            continue
        ic_features = ic_extractor.extract(charge["voltage"], charge["current"],
                                           charge["time"])
        row = {
            "battery_id": battery_id,
            "cycle_index": int(charge["cycle_index"]),
            "discharge_cycle_index": int(discharge["cycle_index"]),
            "sequence": sequence,
            "soh": soh,
            "discharge_capacity": float(discharge["capacity"]),
            "initial_capacity": initial_capacity,
            "ic_valid": ic_features is not None,
        }
        if ic_features is not None:
            row["vp1"], row["hp1"] = ic_features
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cycle_index").reset_index(drop=True)


class BatteryDataset(Dataset):
    def __init__(self, data: pd.DataFrame, feature_columns=None, label_column: str = "soh", **_):
        self.data = data.reset_index(drop=True)
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        row = self.data.iloc[index]
        sequence = np.asarray(row["sequence"], dtype=np.float32)
        if sequence.ndim != 2:
            raise ValueError(f"sequence must have shape (T,C), got {sequence.shape}")
        label = torch.tensor([float(row[self.label_column])], dtype=torch.float32)
        sample = {
            "features": torch.from_numpy(sequence.copy()),
            "label": label,
            "soh": label,
            "battery_id": str(row.get("battery_id", "unknown")),
            "cycle_index": int(row.get("cycle_index", 0)),
        }
        if "poison_mask" in self.data.columns:
            sample["poison_mask"] = torch.tensor(int(row.get("poison_mask", 0)))
        if "ic_valid" in self.data.columns:
            valid = bool(row.get("ic_valid", False))
            vp1 = float(row.get("vp1", 0.0)) if valid else 0.0
            hp1 = float(row.get("hp1", 0.0)) if valid else 0.0
            if not np.isfinite(vp1) or not np.isfinite(hp1):
                valid, vp1, hp1 = False, 0.0, 0.0
            sample["ic_features"] = torch.tensor([vp1, hp1], dtype=torch.float32)
            sample["ic_valid"] = torch.tensor(valid, dtype=torch.bool)
        return sample


class BatteryDataModule:
    def __init__(self, config: dict):
        self.config = config
        self.processed_dir = Path(config["data"]["processed_data_dir"])
        self.batch_size = int(config["data"].get("batch_size", 32))
        self.train_data = self.val_data = self.test_data = pd.DataFrame()
        self.test_old_data = self.holdout_data = pd.DataFrame()
        self.test_data_clean = self.test_data_poisoned = pd.DataFrame()
        self.feature_columns = ["sequence"]
        self.attack_info: dict = {}

    def _load(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Missing prepared metadata: {path}")
        with path.open("rb") as handle:
            return pickle.load(handle)

    def load_processed_data(self, task: str | None = None) -> None:
        metadata = self._load(self.processed_dir / (f"metadata_{task}.pkl" if task else "metadata.pkl"))
        self.train_data = metadata.get("train", pd.DataFrame())
        self.val_data = metadata.get("val", pd.DataFrame())
        self.test_old_data = metadata.get("test_old", pd.DataFrame())
        self.test_data = metadata.get("test_clean", metadata.get("test", pd.DataFrame()))

    def load_poisoned_data(self, metadata_path: str) -> None:
        metadata = self._load(Path(metadata_path))
        self.train_data = metadata["train"]
        self.val_data = metadata["val"]
        self.holdout_data = metadata.get("holdout", pd.DataFrame())
        self.test_data_clean = metadata.get("test_clean", metadata.get("test", pd.DataFrame()))
        self.test_data_poisoned = metadata.get("test_poisoned", pd.DataFrame())
        self.test_data = self.test_data_clean
        self.attack_info = metadata.get("attack_info", {})

    def get_dataloaders(self, num_workers: int = 0, pin_memory: bool = False):
        train = DataLoader(BatteryDataset(self.train_data), batch_size=self.batch_size,
                           shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        validation = DataLoader(BatteryDataset(self.val_data), batch_size=self.batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test = DataLoader(BatteryDataset(self.test_data), batch_size=self.batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train, validation, test


def prepare_nasa(config_or_path) -> dict:
    config = load_public_config(config_or_path)
    loader = NASABatteryLoader(config)
    preprocessor = SequencePreprocessor(config)
    ic_extractor = ICFeatureExtractor(config)
    count = int(config["preprocessing"]["initial_capacity"]["num_valid_discharge_cycles"])
    frames = {battery: _battery_frame(battery, loader.load(battery), preprocessor,
                                      ic_extractor, count)
              for battery in config["data"]["battery_ids"]}
    task1 = config["protocol"]["task1_split"]
    source = frames[task1["train_battery"]]
    split = int(len(source) * float(task1["train_ratio"]))
    task1_payload = {
        "train": source.iloc[:split].reset_index(drop=True),
        "val": frames[task1["validation_battery"]],
        "test_old": source.iloc[split:].reset_index(drop=True),
        "test": pd.DataFrame(),
        "feature_columns": ["sequence"],
    }
    task2 = config["protocol"]["task2_split"]
    target = frames[task2["incremental_battery"]]
    train_end = int(len(target) * float(task2["train_ratio"]))
    val_end = train_end + int(len(target) * float(task2["validation_ratio"]))
    task2_payload = {
        "train": target.iloc[:train_end].reset_index(drop=True),
        "val": target.iloc[train_end:val_end].reset_index(drop=True),
        "holdout": target.iloc[val_end:].reset_index(drop=True),
        "feature_columns": ["sequence"],
    }
    final_payload = {
        "test_clean": frames[config["protocol"]["final_test_battery"]],
        "feature_columns": ["sequence"],
    }
    directory = Path(config["data"]["processed_data_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "task1": directory / "metadata_task1.pkl",
        "task2": directory / "metadata_task2_clean.pkl",
        "final_test": directory / "metadata_final_test.pkl",
    }
    for key, payload in (("task1", task1_payload), ("task2", task2_payload),
                         ("final_test", final_payload)):
        with outputs[key].open("wb") as handle:
            pickle.dump(payload, handle)
    return {key: str(path) for key, path in outputs.items()}
