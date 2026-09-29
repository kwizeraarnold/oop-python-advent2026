"""Reusable OOP logic for Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model.

Models a Jinja fish-export cooperative's stock and revenue:

- ``FishStock``: discrete logistic growth with harvesting.
- ``PriceModel``: a bounded random walk for weekly export price (UGX/kg).
- ``RiskAssessor``: classifies revenue risk by coefficient of variation (CV)
  and runs Monte Carlo simulations to estimate Value-at-Risk.
"""

from __future__ import annotations

import statistics
from typing import Iterable

import numpy as np


def fibonacci_stock(n: int = 15) -> np.ndarray:
    """Generate the first n Fibonacci numbers as a naive 'stock' baseline.

    This is the flawed baseline model being replaced by ``FishStock``:
    unbounded Fibonacci growth has no notion of carrying capacity.

    Args:
        n: Number of terms to generate.

    Returns:
        Array of the first n Fibonacci numbers (1, 1, 2, 3, 5, ...).

    Raises:
        ValueError: If n < 1.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    seq = [1, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return np.array(seq[:n], dtype=float)


class FishStock:
    """Discrete logistic growth model of a harvested fish population.

    N(t+1) = N(t) + r*N(t)*(1 - N(t)/K) - h*N(t)

    Attributes:
        r: Intrinsic growth rate per week.
        K: Carrying capacity, tonnes.
        N0: Initial stock, tonnes.
    """

    def __init__(self, r: float = 0.4, K: float = 10_000.0, N0: float = 4_000.0) -> None:
        """Initialise the stock model, validating its parameters.

        Args:
            r: Intrinsic growth rate per week (must be >= 0).
            K: Carrying capacity in tonnes (must be > 0).
            N0: Initial stock in tonnes (must be >= 0).

        Raises:
            ValueError: If K <= 0, N0 < 0, or r < 0.
        """
        if K <= 0:
            raise ValueError("K must be positive")
        if N0 < 0:
            raise ValueError("N0 cannot be negative")
        if r < 0:
            raise ValueError("r cannot be negative")
        self.r = r
        self.K = K
        self.N0 = N0

    def __repr__(self) -> str:
        return f"FishStock(r={self.r}, K={self.K}, N0={self.N0})"

    @property
    def msy(self) -> float:
        """Theoretical maximum sustainable yield, tonnes/week (rK/4)."""
        return self.r * self.K / 4

    def simulate(
        self,
        weeks: int = 52,
        h: float = 0.10,
        closed_weeks: set[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate weekly stock and harvest under a constant harvest rate.

        Args:
            weeks: Number of weeks to simulate.
            h: Fractional weekly harvest rate applied to N(t).
            closed_weeks: Week-of-year indices (0-based, taken mod 52) with
                zero harvesting, e.g. a closed breeding season. None for no
                closed season.

        Returns:
            A tuple ``(stock, harvest)``: ``stock`` has length ``weeks + 1``
            (index 0 is N0), ``harvest`` has length ``weeks`` (tonnes
            removed at each step).

        Raises:
            ValueError: If weeks < 1 or h is outside [0, 1].
        """
        if weeks < 1:
            raise ValueError("weeks must be >= 1")
        if not 0.0 <= h <= 1.0:
            raise ValueError("h must be between 0 and 1")
        stock = np.empty(weeks + 1)
        harvest = np.empty(weeks)
        stock[0] = self.N0
        for t in range(weeks):
            week_h = 0.0 if (closed_weeks and (t % 52) in closed_weeks) else h
            harvest[t] = week_h * stock[t]
            growth = self.r * stock[t] * (1 - stock[t] / self.K)
            stock[t + 1] = max(stock[t] + growth - harvest[t], 0.0)
        return stock, harvest


class PriceModel:
    """A bounded random-walk model of weekly fish export price (UGX/kg)."""

    def __init__(
        self,
        start_price: float = 12_000.0,
        low: float = 9_000.0,
        high: float = 16_000.0,
        step_std: float = 350.0,
    ) -> None:
        """Initialise the price model, validating its parameters.

        Args:
            start_price: Price at week 0, UGX/kg.
            low: Lower bound clip, UGX/kg.
            high: Upper bound clip, UGX/kg.
            step_std: Standard deviation of the weekly random step, UGX/kg.

        Raises:
            ValueError: If start_price does not lie strictly between low and high.
        """
        if not low < start_price < high:
            raise ValueError("start_price must lie strictly between low and high")
        self.start_price = start_price
        self.low = low
        self.high = high
        self.step_std = step_std

    def __repr__(self) -> str:
        return (
            f"PriceModel(start_price={self.start_price}, low={self.low}, "
            f"high={self.high}, step_std={self.step_std})"
        )

    def simulate(self, weeks: int = 52, rng: np.random.Generator | None = None) -> np.ndarray:
        """Simulate a bounded random walk of weekly prices.

        Args:
            weeks: Number of weeks to simulate.
            rng: A seeded ``np.random.default_rng`` generator. Pass your own
                for reproducibility; a fresh unseeded one is created otherwise.

        Returns:
            Array of length ``weeks`` with the simulated price path (UGX/kg).

        Raises:
            ValueError: If weeks < 1.
        """
        if weeks < 1:
            raise ValueError("weeks must be >= 1")
        if rng is None:
            rng = np.random.default_rng()
        prices = np.empty(weeks)
        price = self.start_price
        for t in range(weeks):
            price = float(np.clip(price + rng.normal(0, self.step_std), self.low, self.high))
            prices[t] = price
        return prices


class RiskAssessor:
    """Classifies revenue risk from its coefficient of variation (CV) and runs
    Monte Carlo simulations to estimate Value-at-Risk.

    A raw variance threshold (e.g. "flag risk if variance > 50000") is
    meaningless here because variance has squared revenue units (UGX^2) and
    scales with the magnitude of the numbers involved. CV = stdev / mean is
    dimensionless, so it is comparable across harvest rates and price
    scenarios.
    """

    def __init__(self, low_cv: float = 0.15, high_cv: float = 0.35) -> None:
        """Initialise the risk thresholds.

        Args:
            low_cv: CV below this is classified "Low" risk.
            high_cv: CV at or above this is classified "High" risk (between
                the two is "Moderate").

        Raises:
            ValueError: If not 0 < low_cv < high_cv.
        """
        if not 0 < low_cv < high_cv:
            raise ValueError("Require 0 < low_cv < high_cv")
        self.low_cv = low_cv
        self.high_cv = high_cv

    def classify(self, cv: float) -> str:
        """Classify risk from a coefficient of variation.

        Args:
            cv: Coefficient of variation (stdev / mean); must be >= 0.

        Returns:
            One of "Low", "Moderate", "High".

        Raises:
            ValueError: If cv is negative.
        """
        if cv < 0:
            raise ValueError("cv cannot be negative")
        if cv < self.low_cv:
            return "Low"
        if cv < self.high_cv:
            return "Moderate"
        return "High"

    def simulate_annual_revenue(
        self,
        fish_stock: FishStock,
        price_model: PriceModel,
        h: float,
        weeks: int = 52,
        n_sims: int = 1000,
        seed: int = 42,
        closed_weeks: set[int] | None = None,
    ) -> np.ndarray:
        """Monte Carlo simulate annual revenue over many price paths.

        The harvest trajectory is deterministic given ``(fish_stock, h)``:
        only the price path is stochastic, so it is what gets resampled.

        Args:
            fish_stock: Configured FishStock instance.
            price_model: Configured PriceModel instance.
            h: Weekly harvest rate.
            weeks: Weeks per simulated year.
            n_sims: Number of Monte Carlo price paths (>= 1000 recommended).
            seed: Seed for reproducibility.
            closed_weeks: Optional closed-season weeks passed through to
                ``FishStock.simulate``.

        Returns:
            Array of length ``n_sims`` with total annual revenue (UGX) per
            simulation.
        """
        rng = np.random.default_rng(seed)
        _stock, harvest_tonnes = fish_stock.simulate(weeks=weeks, h=h, closed_weeks=closed_weeks)
        harvest_kg = harvest_tonnes * 1000.0
        revenues = np.empty(n_sims)
        for i in range(n_sims):
            prices = price_model.simulate(weeks=weeks, rng=rng)
            revenues[i] = float(np.sum(harvest_kg * prices))
        return revenues

    @staticmethod
    def value_at_risk(revenues: np.ndarray, pct: float = 0.05) -> float:
        """5% Value-at-Risk: the revenue level below which only ``pct`` of
        simulated outcomes fall.

        Args:
            revenues: Array of simulated revenue outcomes.
            pct: Tail probability, e.g. 0.05 for the 5th percentile.

        Returns:
            The VaR threshold, in the same units as ``revenues``.

        Raises:
            ValueError: If revenues is empty.
        """
        revenues = np.asarray(revenues, dtype=float)
        if revenues.size == 0:
            raise ValueError("revenues cannot be empty")
        return float(np.percentile(revenues, pct * 100))


def revenue_stats(revenues: Iterable[float]) -> dict[str, float]:
    """Mean, median, variance, stdev and CV of a revenue series via ``statistics``.

    Args:
        revenues: Sequence of revenue values.

    Returns:
        Dict with keys ``mean``, ``median``, ``variance``, ``stdev``, ``cv``.

    Raises:
        ValueError: If revenues is empty.
    """
    data = list(revenues)
    if not data:
        raise ValueError("revenues cannot be empty")
    mean = statistics.mean(data)
    if len(data) == 1:
        return {"mean": mean, "median": data[0], "variance": 0.0, "stdev": 0.0, "cv": 0.0}
    variance = statistics.variance(data)
    stdev = statistics.stdev(data)
    return {
        "mean": mean,
        "median": statistics.median(data),
        "variance": variance,
        "stdev": stdev,
        "cv": stdev / mean if mean != 0 else float("inf"),
    }
