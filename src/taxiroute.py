"""Reusable OOP logic for Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner.

Models a matatu association running three routes out of Kampala:

- ``Route``: passenger counts, fare, revenue and descriptive statistics.
- ``Forecaster`` and its subclasses: moving average, exponential smoothing,
  linear trend, and (extension) seasonal-naive demand forecasters, sharing a
  common ``fit``/``predict`` interface.
- Helper functions for supply-demand equilibrium, walk-forward backtesting,
  alpha grid search, fleet sizing, and synthetic weekly-pattern demand.
"""

from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from typing import Callable, Iterable

import numpy as np
from scipy.linalg import solve as linalg_solve


class Route:
    """A taxi route's daily passenger counts and fare.

    Attributes:
        name: Route name.
        passengers: NumPy array of daily passenger counts.
        fare: Fare per passenger, UGX.
    """

    def __init__(self, name: str, passengers: Iterable[float], fare: float) -> None:
        """Initialise a route, validating its passenger counts and fare.

        Args:
            name: Route name.
            passengers: Daily passenger counts (at least one day).
            fare: Fare per passenger, UGX (must be positive).

        Raises:
            ValueError: If passengers is empty, contains negative values, or fare <= 0.
        """
        arr = np.asarray(list(passengers), dtype=float)
        if arr.size == 0:
            raise ValueError("passengers cannot be empty")
        if np.any(arr < 0):
            raise ValueError("passenger counts cannot be negative")
        if fare <= 0:
            raise ValueError("fare must be positive")
        self.name = name
        self.passengers = arr
        self.fare = fare

    def __repr__(self) -> str:
        return f"Route({self.name!r}, fare={self.fare}, days={len(self.passengers)})"

    def __len__(self) -> int:
        return len(self.passengers)

    def daily_revenue(self) -> np.ndarray:
        """Daily revenue (UGX) = passengers * fare."""
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        """Total revenue over all recorded days, UGX."""
        return float(self.daily_revenue().sum())

    def stats(self) -> dict[str, float]:
        """Mean, median, variance and standard deviation of daily passengers via ``statistics``."""
        data = self.passengers.tolist()
        result = {"mean": statistics.mean(data), "median": statistics.median(data)}
        if len(data) > 1:
            result["variance"] = statistics.variance(data)
            result["stdev"] = statistics.stdev(data)
        else:
            result["variance"] = 0.0
            result["stdev"] = 0.0
        return result


def solve_equilibrium(a: float, b: float, c: float, d: float) -> tuple[float, float]:
    """Solve a linear supply-demand system for equilibrium price and quantity.

    Demand: Q = a - b*P.  Supply: Q = c + d*P.

    Args:
        a: Demand intercept.
        b: Demand slope (quantity falls by b per unit price).
        c: Supply intercept.
        d: Supply slope (quantity rises by d per unit price).

    Returns:
        A tuple ``(Q_star, P_star)`` of equilibrium quantity and price.

    Raises:
        ValueError: If b + d == 0 (parallel demand/supply lines, no unique equilibrium).
    """
    if b + d == 0:
        raise ValueError("b + d cannot be zero (demand and supply lines are parallel)")
    coefficients = np.array([[1.0, b], [1.0, -d]])
    rhs = np.array([a, c])
    q_star, p_star = linalg_solve(coefficients, rhs)
    return float(q_star), float(p_star)


class Forecaster(ABC):
    """Common interface for the demand forecasters below."""

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def fit(self, series: Iterable[float]) -> "Forecaster":
        """Fit the forecaster on a historical series."""

    @abstractmethod
    def predict(self, horizon: int = 1) -> np.ndarray:
        """Forecast the next ``horizon`` steps."""

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fit() before predict()")

    @staticmethod
    def _require_positive_horizon(horizon: int) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")


class MovingAverageForecaster(Forecaster):
    """Flat forecast: the mean of the last ``window`` observations."""

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._level: float | None = None

    def fit(self, series: Iterable[float]) -> "MovingAverageForecaster":
        """Store the mean of the last ``window`` observations as the forecast level.

        Raises:
            ValueError: If series has fewer than ``window`` points.
        """
        arr = np.asarray(list(series), dtype=float)
        if arr.size < self.window:
            raise ValueError(f"need at least {self.window} points to fit a {self.window}-day moving average")
        self._level = float(arr[-self.window:].mean())
        self._fitted = True
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Repeat the fitted moving-average level for ``horizon`` steps."""
        self._require_fitted()
        self._require_positive_horizon(horizon)
        return np.full(horizon, self._level)


class ExponentialSmoothingForecaster(Forecaster):
    """Flat forecast from simple exponential smoothing with a tunable alpha."""

    def __init__(self, alpha: float = 0.3) -> None:
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._level: float | None = None

    def fit(self, series: Iterable[float]) -> "ExponentialSmoothingForecaster":
        """Recursively smooth the series (level_t = alpha*x_t + (1-alpha)*level_{t-1}).

        Raises:
            ValueError: If series is empty.
        """
        arr = np.asarray(list(series), dtype=float)
        if arr.size < 1:
            raise ValueError("need at least 1 point to fit")
        level = arr[0]
        for x in arr[1:]:
            level = self.alpha * x + (1 - self.alpha) * level
        self._level = float(level)
        self._fitted = True
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Repeat the fitted smoothed level for ``horizon`` steps (flat forecast)."""
        self._require_fitted()
        self._require_positive_horizon(horizon)
        return np.full(horizon, self._level)


class LinearTrendForecaster(Forecaster):
    """Forecast by extrapolating a least-squares linear trend (``np.polyfit``)."""

    def __init__(self) -> None:
        super().__init__()
        self._coeffs: np.ndarray | None = None
        self._n: int | None = None

    def fit(self, series: Iterable[float]) -> "LinearTrendForecaster":
        """Fit a least-squares line (``np.polyfit``, degree 1) against the time index.

        Raises:
            ValueError: If series has fewer than 2 points.
        """
        arr = np.asarray(list(series), dtype=float)
        if arr.size < 2:
            raise ValueError("need at least 2 points to fit a linear trend")
        t = np.arange(arr.size)
        self._coeffs = np.polyfit(t, arr, 1)
        self._n = arr.size
        self._fitted = True
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Extrapolate the fitted line ``horizon`` steps past the last fitted index."""
        self._require_fitted()
        self._require_positive_horizon(horizon)
        slope, intercept = self._coeffs
        t_future = np.arange(self._n, self._n + horizon)
        return slope * t_future + intercept


class SeasonalNaiveForecaster(Forecaster):
    """Forecast day t as the value observed ``season_length`` steps before it."""

    def __init__(self, season_length: int = 7) -> None:
        super().__init__()
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        self.season_length = season_length
        self._series: np.ndarray | None = None

    def fit(self, series: Iterable[float]) -> "SeasonalNaiveForecaster":
        """Store the series for later seasonal look-back.

        Raises:
            ValueError: If series has fewer than ``season_length`` points.
        """
        arr = np.asarray(list(series), dtype=float)
        if arr.size < self.season_length:
            raise ValueError(f"need at least {self.season_length} points to fit a seasonal-naive model")
        self._series = arr
        self._fitted = True
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Forecast each step as the value ``season_length`` steps before it,
        chaining into its own forecasts once the horizon exceeds the season length."""
        self._require_fitted()
        self._require_positive_horizon(horizon)
        n = self._series.size
        out = np.empty(horizon)
        for h in range(horizon):
            idx = n + h - self.season_length
            if idx < 0:
                raise ValueError("not enough history to forecast this far ahead with this season_length")
            out[h] = self._series[idx] if idx < n else out[idx - n]
        return out


def walk_forward_mae(make_forecaster: Callable[[], Forecaster], series: Iterable[float], start: int) -> float:
    """Rolling-origin (walk-forward) one-step-ahead MAE.

    For each t in [start, len(series)), fits a fresh forecaster on
    series[:t] and scores its one-step forecast against series[t].

    Args:
        make_forecaster: A zero-argument callable returning a fresh, unfit Forecaster.
        series: The full historical series.
        start: First index to forecast (0-indexed); must be < len(series).

    Returns:
        Mean absolute error over all backtested steps.

    Raises:
        ValueError: If start >= len(series).
    """
    arr = np.asarray(list(series), dtype=float)
    if start >= arr.size:
        raise ValueError("start must be less than the series length")
    errors = []
    for t in range(start, arr.size):
        model = make_forecaster()
        model.fit(arr[:t])
        pred = model.predict(horizon=1)[0]
        errors.append(abs(pred - arr[t]))
    return float(np.mean(errors))


def grid_search_alpha(series: Iterable[float], start: int, alphas: Iterable[float]) -> tuple[float, float]:
    """Grid-search alpha for ExponentialSmoothingForecaster by walk-forward MAE.

    Args:
        series: The full historical series.
        start: First index to forecast (passed through to walk_forward_mae).
        alphas: Candidate alpha values to try.

    Returns:
        A tuple ``(best_alpha, best_mae)``.

    Raises:
        ValueError: If alphas is empty.
    """
    alphas = list(alphas)
    if not alphas:
        raise ValueError("alphas cannot be empty")
    best_alpha, best_mae = alphas[0], float("inf")
    for a in alphas:
        mae = walk_forward_mae(lambda a=a: ExponentialSmoothingForecaster(alpha=a), series, start)
        if mae < best_mae:
            best_alpha, best_mae = a, mae
    return best_alpha, best_mae


def vehicles_needed(forecast_passengers: float, trips_per_day: int = 8, seats: int = 14, buffer: float = 0.15) -> int:
    """Number of vehicles to deploy to cover a forecasted passenger count.

    Args:
        forecast_passengers: Forecast passengers for the day.
        trips_per_day: One-way trips each vehicle makes per day.
        seats: Seats per vehicle.
        buffer: Fractional safety buffer added to the forecast, e.g. 0.15 for 15%.

    Returns:
        Number of vehicles needed (rounded up).

    Raises:
        ValueError: If forecast_passengers is negative.
    """
    if forecast_passengers < 0:
        raise ValueError("forecast_passengers cannot be negative")
    capacity_per_vehicle = trips_per_day * seats
    return int(np.ceil(forecast_passengers * (1 + buffer) / capacity_per_vehicle))


DEFAULT_WEEKDAY_MULTIPLIERS = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.05, 4: 1.3, 5: 1.1, 6: 0.6}


def simulate_weekly_demand(
    base_mean: float,
    n_days: int = 60,
    noise_std: float = 5.0,
    seed: int = 0,
    weekday_multipliers: dict[int, float] | None = None,
) -> np.ndarray:
    """Simulate daily passenger demand with a weekly pattern and noise.

    Day 0 is treated as a Monday. Default multipliers make Friday (index 4)
    the busiest day and Sunday (index 6) the quietest, matching typical
    matatu ridership patterns.

    Args:
        base_mean: Baseline (Monday-equivalent) daily passenger count.
        n_days: Number of days to simulate.
        noise_std: Standard deviation of additive Gaussian noise.
        seed: Seed for ``np.random.default_rng`` (reproducibility).
        weekday_multipliers: Optional override of the default weekday multipliers.

    Returns:
        Array of length ``n_days`` with simulated (non-negative) passenger counts.

    Raises:
        ValueError: If n_days < 1 or base_mean < 0.
    """
    if n_days < 1:
        raise ValueError("n_days must be >= 1")
    if base_mean < 0:
        raise ValueError("base_mean cannot be negative")
    multipliers = weekday_multipliers or DEFAULT_WEEKDAY_MULTIPLIERS
    rng = np.random.default_rng(seed)
    weekday = np.arange(n_days) % 7
    mult = np.array([multipliers[int(d)] for d in weekday])
    values = base_mean * mult + rng.normal(0, noise_std, size=n_days)
    return np.clip(values, 0, None)
