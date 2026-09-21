from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .runtime import load_public_config


FORMAL_ATTACK_KEYS = {
    "BT1": "Attack-BTO-strong",
    "BT2": "Attack-BTO-bilevel",
}


class BackdoorAttack:
    def __init__(self, config: dict):
        self.config = config.get("attack", config)
        self.base_seed = int(self.config.get("seed", 42))
        self.y_target = float(self.config.get("target", {}).get("y_target", 0.90))
        self.epsilon_asr = float(self.config.get("target", {}).get("epsilon_asr", 0.01))
        self._prepared_triggers: Dict[str, Dict[str, object]] = {}

    @staticmethod
    def _split_layout(feature: np.ndarray) -> Tuple[np.ndarray, str]:
        sample = np.asarray(feature, dtype=np.float32)
        if sample.ndim != 2:
            raise ValueError(f"Expected a two-dimensional sample, got {sample.shape}")
        if sample.shape[0] >= sample.shape[1]:
            return sample.copy(), "TC"
        return sample.T.copy(), "CT"

    @staticmethod
    def _restore_layout(feature: np.ndarray, layout: str) -> np.ndarray:
        if layout == "TC":
            return feature
        if layout == "CT":
            return feature.T
        raise ValueError(f"Unknown layout: {layout}")

    def prepare_attack(self, attack_type: str, reference_features: np.ndarray,
                       reference_labels=None) -> Dict[str, object]:
        del reference_labels
        if attack_type not in FORMAL_ATTACK_KEYS:
            raise ValueError(f"Unsupported attack type: {attack_type}")
        if attack_type in self._prepared_triggers:
            return self._prepared_triggers[attack_type]

        values = np.asarray(reference_features, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(f"Expected a three-dimensional reference batch, got {values.shape}")
        if values.shape[1] >= values.shape[2]:
            sequence_length, channels = values.shape[1], values.shape[2]
        else:
            sequence_length, channels = values.shape[2], values.shape[1]

        trigger = self.config.get("trigger")
        if not isinstance(trigger, dict):
            raise ValueError("attack.trigger must be a mapping")
        start = int(trigger["start"])
        length = int(trigger["length"])
        end = start + length
        patch = np.asarray(trigger["patch"], dtype=np.float32)
        if start < 0 or length < 1 or end > sequence_length:
            raise ValueError("Trigger window is outside the input sequence")
        expected_shape = (length, channels)
        if patch.shape != expected_shape:
            raise ValueError(f"Trigger patch must have shape {expected_shape}, got {patch.shape}")

        prepared = {"start": start, "end": end, "patch": patch}
        self._prepared_triggers[attack_type] = prepared
        return prepared

    def _apply(self, feature: np.ndarray, attack_type: str) -> np.ndarray:
        prepared = self._prepared_triggers.get(attack_type)
        if prepared is None:
            raise RuntimeError(f"Attack {attack_type} has not been prepared")
        feature_tc, layout = self._split_layout(feature)
        start = int(prepared["start"])
        end = int(prepared["end"])
        patch = np.asarray(prepared["patch"], dtype=np.float32)
        feature_tc[start:end] = np.clip(feature_tc[start:end] + patch, 0.0, 1.0)
        return self._restore_layout(feature_tc, layout)

    def generate_poisoned_samples(self, features, labels, poison_ratio=0.20,
                                  attack_type="BT1", rng=None):
        if attack_type not in FORMAL_ATTACK_KEYS:
            raise ValueError(f"Unsupported attack type: {attack_type}")
        ratio = float(poison_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("poison_ratio must lie in [0, 1]")
        rng = rng or np.random.default_rng(self.base_seed)
        poisoned_features = np.array(features, copy=True)
        poisoned_labels = np.array(labels, copy=True)
        poison_mask = np.zeros(len(features), dtype=bool)
        if ratio == 0.0 or len(features) == 0:
            return poisoned_features, poisoned_labels, poison_mask
        count = max(1, int(round(len(features) * ratio)))
        indices = rng.choice(len(features), size=count, replace=False)
        for index in indices:
            poisoned_features[index] = self._apply(poisoned_features[index], attack_type)
            poisoned_labels[index] = self.y_target
            poison_mask[index] = True
        return poisoned_features, poisoned_labels, poison_mask


def derive_stable_seed(base_seed: int, *parts: object) -> int:
    payload = "::".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**32 - 1)


def generate_poisoned_dataframe(
    clean_df: pd.DataFrame,
    attack: BackdoorAttack,
    attack_type: str,
    poison_ratio: float,
    rng_seed: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if len(clean_df) == 0:
        return clean_df.copy(), np.zeros(0, dtype=bool)
    features = np.stack(clean_df["sequence"].to_list()).astype(np.float32)
    labels = clean_df["soh"].to_numpy(dtype=np.float32)
    poisoned_features, poisoned_labels, poison_mask = attack.generate_poisoned_samples(
        features,
        labels,
        poison_ratio=poison_ratio,
        attack_type=attack_type,
        rng=np.random.default_rng(rng_seed),
    )
    poisoned_df = clean_df.copy()
    for offset, row_index in enumerate(clean_df.index):
        poisoned_df.at[row_index, "sequence"] = poisoned_features[offset]
        poisoned_df.at[row_index, "soh"] = float(poisoned_labels[offset])
    poisoned_df["poison_mask"] = poison_mask.astype(int)
    return poisoned_df, poison_mask


def build_paper_metadata(
    task2_train_clean: pd.DataFrame,
    task2_val_clean: pd.DataFrame,
    task2_holdout_clean: pd.DataFrame,
    final_test_clean: pd.DataFrame,
    poisoned_train_df: pd.DataFrame,
    poisoned_test_df: pd.DataFrame,
    config: dict,
    attack_info: dict,
) -> Dict:
    val_with_mask = task2_val_clean.copy()
    if "poison_mask" not in val_with_mask.columns:
        val_with_mask["poison_mask"] = 0
    holdout_with_mask = task2_holdout_clean.copy()
    if "poison_mask" not in holdout_with_mask.columns:
        holdout_with_mask["poison_mask"] = 0
    test_clean_with_mask = final_test_clean.copy()
    if "poison_mask" not in test_clean_with_mask.columns:
        test_clean_with_mask["poison_mask"] = 0
    return {
        "train": poisoned_train_df,
        "val": val_with_mask,
        "holdout": holdout_with_mask,
        "test": test_clean_with_mask,
        "test_clean": test_clean_with_mask,
        "test_poisoned": poisoned_test_df,
        "feature_columns": ["sequence"],
        "attack_info": attack_info,
        "config": config,
    }


def build_poison_metadata(config_or_path) -> str:
    config = load_public_config(config_or_path)
    attack_cfg = config["attack"]
    attack_seed = int(attack_cfg.get("seed", 42))
    attack_type = str(attack_cfg["type"])
    seed_attack_type = FORMAL_ATTACK_KEYS[attack_type]
    poison_ratio = float(attack_cfg["poison_ratio"])
    processed_dir = Path(config["data"].get("processed_data_dir", "data/processed"))
    with (processed_dir / "metadata_task2_clean.pkl").open("rb") as handle:
        task2 = pickle.load(handle)
    with (processed_dir / "metadata_final_test.pkl").open("rb") as handle:
        final = pickle.load(handle)

    attack = BackdoorAttack(config)
    clean_train = task2["train"]
    reference_features = np.stack(clean_train["sequence"].to_list()).astype(np.float32)
    attack.prepare_attack(attack_type, reference_features)
    train_seed = derive_stable_seed(attack_seed, seed_attack_type, poison_ratio, "train")
    test_seed = derive_stable_seed(attack_seed, seed_attack_type, poison_ratio, "final_test")
    train_df, train_mask = generate_poisoned_dataframe(
        clean_train, attack, attack_type, poison_ratio, train_seed
    )
    test_df, test_mask = generate_poisoned_dataframe(
        final["test_clean"], attack, attack_type, poison_ratio, test_seed
    )
    attack_info = {
        "attack_type": attack_type,
        "formal_attack_type": seed_attack_type,
        "poison_ratio": poison_ratio,
        "y_target": attack.y_target,
        "epsilon_asr": attack.epsilon_asr,
        "n_poisoned_train": int(train_mask.sum()),
        "n_poisoned_test": int(test_mask.sum()),
        "base_seed": attack_seed,
        "train_seed": train_seed,
        "test_seed": test_seed,
    }
    payload = build_paper_metadata(
        clean_train,
        task2["val"],
        task2.get("holdout", pd.DataFrame()),
        final["test_clean"],
        train_df,
        test_df,
        config,
        attack_info,
    )
    output_dir = processed_dir / "poisoned"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"metadata_task2_poisoned_{attack_type}_ratio_{poison_ratio:.2f}.pkl"
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return str(path)
