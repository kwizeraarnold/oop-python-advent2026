"""Reusable OOP logic for Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser.

Models monthly rainfall regimes across Ugandan regions and classifies
month-by-month crop suitability, replacing a flawed prior-cohort method that
used ``math.cos()`` (the cosine of a scalar angle) where cosine *similarity*
of two vectors was required.
"""

from __future__ import annotations

import statistics
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


class Region:
    """A region's monthly rainfall profile (12 values, mm).

    Attributes:
        name: Region name.
        rainfall_mm: Length-12 NumPy array of monthly rainfall, Jan..Dec.
    """

    def __init__(self, name: str, rainfall_mm: Iterable[float]) -> None:
        """Initialise a region, validating that it has exactly 12 non-negative values.

        Args:
            name: Region name.
            rainfall_mm: 12 monthly rainfall values (mm), Jan..Dec.

        Raises:
            ValueError: If there are not exactly 12 values, or any is negative.
        """
        arr = np.asarray(list(rainfall_mm), dtype=float)
        if arr.shape != (12,):
            raise ValueError(f"rainfall_mm must have exactly 12 values, got {arr.shape[0] if arr.ndim else 0}")
        if np.any(arr < 0):
            raise ValueError("rainfall cannot be negative")
        self.name = name
        self.rainfall_mm = arr

    def __repr__(self) -> str:
        return f"Region({self.name!r}, annual_total={self.annual_total():.1f}mm)"

    def __len__(self) -> int:
        return len(self.rainfall_mm)

    def annual_total(self) -> float:
        """Total annual rainfall, mm."""
        return float(self.rainfall_mm.sum())

    def mean(self) -> float:
        """Mean monthly rainfall, mm."""
        return float(self.rainfall_mm.mean())

    def wettest_month(self) -> tuple[str, float]:
        """The wettest month and its rainfall (mm)."""
        idx = int(np.argmax(self.rainfall_mm))
        return MONTH_NAMES[idx], float(self.rainfall_mm[idx])

    def driest_month(self) -> tuple[str, float]:
        """The driest month and its rainfall (mm)."""
        idx = int(np.argmin(self.rainfall_mm))
        return MONTH_NAMES[idx], float(self.rainfall_mm[idx])

    def coefficient_of_variation(self) -> float:
        """CV of monthly rainfall (stdev / mean), a scale-free measure of seasonality."""
        data = self.rainfall_mm.tolist()
        mean = statistics.mean(data)
        if mean == 0:
            return 0.0
        return statistics.stdev(data) / mean


class CropRule:
    """A crop's suitable monthly rainfall band, with drought/waterlogging cutoffs.

    Attributes:
        crop: Crop name.
        drought_max_mm: Rainfall at or below this is classified a drought risk.
        waterlog_min_mm: Rainfall at or above this is classified a waterlogging risk.
        source: Short citation for the thresholds.
    """

    def __init__(self, crop: str, drought_max_mm: float, waterlog_min_mm: float, source: str) -> None:
        """Initialise a crop rule, validating the threshold ordering.

        Args:
            crop: Crop name.
            drought_max_mm: Upper bound (mm) of the drought-risk band.
            waterlog_min_mm: Lower bound (mm) of the waterlogging-risk band.
            source: Short citation justifying the thresholds.

        Raises:
            ValueError: If drought_max_mm >= waterlog_min_mm.
        """
        if drought_max_mm >= waterlog_min_mm:
            raise ValueError("drought_max_mm must be less than waterlog_min_mm")
        self.crop = crop
        self.drought_max_mm = drought_max_mm
        self.waterlog_min_mm = waterlog_min_mm
        self.source = source

    def __repr__(self) -> str:
        return (
            f"CropRule({self.crop!r}, good band=({self.drought_max_mm}, "
            f"{self.waterlog_min_mm}) mm/month)"
        )

    def classify(self, rainfall_mm: float) -> str:
        """Classify a single month's rainfall for this crop.

        Args:
            rainfall_mm: Rainfall for one month, mm.

        Returns:
            "Drought risk for {crop}", "Waterlogging risk for {crop}", or
            "Good for {crop}".

        Raises:
            ValueError: If rainfall_mm is negative.
        """
        if rainfall_mm < 0:
            raise ValueError("rainfall_mm cannot be negative")
        if rainfall_mm <= self.drought_max_mm:
            return f"Drought risk for {self.crop}"
        if rainfall_mm >= self.waterlog_min_mm:
            return f"Waterlogging risk for {self.crop}"
        return f"Good for {self.crop}"


def suitability_table(regions: list[Region], crops: list[CropRule]) -> pd.DataFrame:
    """Classify every (region, month, crop) combination.

    Args:
        regions: List of Region instances.
        crops: List of CropRule instances.

    Returns:
        Long-format DataFrame with columns region, month, month_num, crop, label.
    """
    rows = []
    for region in regions:
        for m in range(12):
            for crop in crops:
                rows.append({
                    "region": region.name,
                    "month": MONTH_NAMES[m],
                    "month_num": m + 1,
                    "crop": crop.crop,
                    "rainfall_mm": float(region.rainfall_mm[m]),
                    "label": crop.classify(region.rainfall_mm[m]),
                })
    return pd.DataFrame(rows)


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    """Cosine similarity of two vectors: (a . b) / (||a|| * ||b||).

    This is the corrected replacement for a prior method that used
    ``math.cos()``, which computes the cosine of a scalar *angle* and has no
    meaning as a measure of similarity between two rainfall vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1] (in practice [0, 1] for non-negative
        rainfall vectors).

    Raises:
        ValueError: If a and b have different shapes, or either is a zero vector.
    """
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(a, b) / (norm_a * norm_b))


def similarity_matrices(regions: dict[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    """Pairwise cosine similarity, Pearson correlation and Euclidean distance.

    Args:
        regions: Mapping of region name -> 12-month rainfall vector.

    Returns:
        Dict with keys "cosine", "pearson", "euclidean", each an NxN
        DataFrame indexed and columned by region name.
    """
    names = list(regions)
    n = len(names)
    cos = np.zeros((n, n))
    pear = np.zeros((n, n))
    euc = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            a, b = regions[ni], regions[nj]
            cos[i, j] = cosine_similarity(a, b)
            pear[i, j] = float(np.corrcoef(a, b)[0, 1])
            euc[i, j] = float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    return {
        "cosine": pd.DataFrame(cos, index=names, columns=names),
        "pearson": pd.DataFrame(pear, index=names, columns=names),
        "euclidean": pd.DataFrame(euc, index=names, columns=names),
    }


def find_rainy_seasons(rainfall_mm: Iterable[float], prominence: float = 15.0) -> dict[str, object]:
    """Detect rainy-season peaks in a 12-month rainfall series, handling wraparound.

    The 12-month series is tiled three times so a peak spanning the Dec/Jan
    boundary is still detected, then only peaks from the middle copy are kept.

    Args:
        rainfall_mm: 12 monthly rainfall values, Jan..Dec.
        prominence: Minimum peak prominence (mm) passed to ``scipy.signal.find_peaks``.

    Returns:
        Dict with ``peak_months`` (1-indexed month numbers) and ``pattern``
        ("Unimodal", "Bimodal", "Multimodal" or "No clear peak").

    Raises:
        ValueError: If rainfall_mm does not have exactly 12 values.
    """
    arr = np.asarray(list(rainfall_mm), dtype=float)
    if arr.shape != (12,):
        raise ValueError("rainfall_mm must have exactly 12 values")
    extended = np.tile(arr, 3)
    peaks, _props = find_peaks(extended, prominence=prominence)
    mid_peaks = sorted({int(p) - 12 for p in peaks if 12 <= p < 24})
    n_peaks = len(mid_peaks)
    if n_peaks == 0:
        pattern = "No clear peak"
    elif n_peaks == 1:
        pattern = "Unimodal"
    elif n_peaks == 2:
        pattern = "Bimodal"
    else:
        pattern = "Multimodal"
    return {"peak_months": [m + 1 for m in mid_peaks], "pattern": pattern}
