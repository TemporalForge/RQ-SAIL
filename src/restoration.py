from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class LocalRepairResult:
    support: Optional[tuple[int, int]]
    repaired: np.ndarray


class RobustLocalRepair:
    def __init__(
        self,
        trusted_sequences: np.ndarray,
        high_multiplier: float = 8.0,
        low_multiplier: float = 2.0,
        min_core_length: int = 3,
        position_scale_floor: float = 1.0e-3,
    ) -> None:
        sequences = np.asarray(trusted_sequences, dtype=np.float32)
        if sequences.ndim != 3 or len(sequences) == 0:
            raise ValueError(
                "trusted_sequences must have non-empty shape (N, T, C)"
            )
        if not 0.0 < low_multiplier < high_multiplier:
            raise ValueError("Require 0 < low_multiplier < high_multiplier")
        if min_core_length < 1:
            raise ValueError("min_core_length must be positive")

        self.high_multiplier = float(high_multiplier)
        self.low_multiplier = float(low_multiplier)
        self.min_core_length = int(min_core_length)
        self.position_median = np.median(sequences, axis=0)
        self.position_mad = (
            1.4826
            * np.median(np.abs(sequences - self.position_median), axis=0)
            + float(position_scale_floor)
        )

    def locate(self, sequence: np.ndarray) -> Optional[tuple[int, int]]:
        values = np.asarray(sequence, dtype=np.float32)
        if values.shape != self.position_median.shape:
            raise ValueError(
                f"Sequence shape {values.shape} does not match calibration "
                f"shape {self.position_median.shape}"
            )
        position_score = np.mean(
            np.abs((values - self.position_median) / self.position_mad),
            axis=1,
        )
        center = float(np.median(position_score))
        scale = float(
            1.4826 * np.median(np.abs(position_score - center)) + 1.0e-6
        )
        high = position_score > center + self.high_multiplier * scale
        low = position_score > center + self.low_multiplier * scale

        runs: list[tuple[int, int, float]] = []
        start: Optional[int] = None
        for index, active in enumerate(np.append(high, False)):
            if active and start is None:
                start = index
            elif not active and start is not None:
                if index - start >= self.min_core_length:
                    runs.append(
                        (start, index, float(position_score[start:index].sum()))
                    )
                start = None
        if not runs:
            return None

        start, end, _ = max(
            runs, key=lambda item: (item[1] - item[0], item[2])
        )
        while start > 0 and low[start - 1]:
            start -= 1
        while end < len(position_score) and low[end]:
            end += 1

        if start <= 0 or end >= len(position_score):
            return None
        return int(start), int(end)

    @staticmethod
    def interpolate(
        sequence: np.ndarray, support: tuple[int, int]
    ) -> np.ndarray:
        start, end = support
        repaired = np.asarray(sequence, dtype=np.float32).copy()
        if not (0 < start < end < len(repaired)):
            raise ValueError(f"Invalid interpolation support: {support}")
        left = repaired[start - 1]
        right = repaired[end]
        width = end - start
        for offset in range(width):
            fraction = float(offset + 1) / float(width + 1)
            repaired[start + offset] = (
                (1.0 - fraction) * left + fraction * right
            )
        return repaired

    def repair(self, sequence: np.ndarray) -> LocalRepairResult:
        support = self.locate(sequence)
        if support is None:
            return LocalRepairResult(
                support=None,
                repaired=np.asarray(sequence, dtype=np.float32).copy(),
            )
        return LocalRepairResult(
            support=support,
            repaired=self.interpolate(sequence, support),
        )
