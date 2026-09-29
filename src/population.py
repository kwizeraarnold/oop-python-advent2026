"""Reusable OOP logic for Mini-Project 1: UBOS District Population Forecaster.

Models district population time series and three forecasting approaches
sharing a common ``Forecaster`` interface:

- ``LinearTrendForecaster``: least-squares linear trend (``np.polyfit``).
- ``ExponentialCAGRForecaster``: compound-growth extrapolation from CAGR.
- ``FibonacciRatioForecaster``: scales the last value by successive
  Fibonacci ratios (a deliberately weak baseline, critiqued in the notebook).
"""

from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class DistrictPopulation:
    """A district's population time series.

    Attributes:
        name: District name.
        years: NumPy array of years.
        population_thousands: NumPy array of population, in thousands.
    """

    def __init__(self, name: str, years: Iterable[int], population_thousands: Iterable[float]) -> None:
        """Initialise a district population series, validating its shape and values.

        Args:
            name: District name.
            years: Years covered (must match population_thousands in length).
            population_thousands: Population estimates, in thousands.

        Raises:
            ValueError: If years and population_thousands differ in length,
                either is empty, or any population value is negative.
        """
        years_arr = np.asarray(list(years), dtype=int)
        pop_arr = np.asarray(list(population_thousands), dtype=float)
        if years_arr.size == 0 or pop_arr.size == 0:
            raise ValueError("years and population_thousands cannot be empty")
        if years_arr.shape != pop_arr.shape:
            raise ValueError("years and population_thousands must have equal length")
        if np.any(pop_arr < 0):
            raise ValueError("population cannot be negative")
        self.name = name
        self.years = years_arr
        self.population_thousands = pop_arr

    def __repr__(self) -> str:
        return (
            f"DistrictPopulation({self.name!r}, {self.years[0]}-{self.years[-1]}, "
            f"latest={self.population_thousands[-1]:.0f}k)"
        )

    def __len__(self) -> int:
        return len(self.years)

    def stats_statistics_module(self) -> dict[str, float]:
        """Mean, median, variance (sample, ddof=1) and stdev via the ``statistics`` module."""
        data = self.population_thousands.tolist()
        return {
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "variance": statistics.variance(data),
            "stdev": statistics.stdev(data),
        }

    def stats_numpy(self, ddof: int = 0) -> dict[str, float]:
        """Mean, median, variance and stdev via NumPy, with an explicit ``ddof``.

        Args:
            ddof: Delta degrees of freedom passed to ``np.var``/``np.std``.
                NumPy defaults to 0 (population variance/stdev); pass 1 to
                match ``statistics.variance``/``statistics.stdev`` (sample
                variance/stdev).
        """
        arr = self.population_thousands
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "variance": float(np.var(arr, ddof=ddof)),
            "stdev": float(np.std(arr, ddof=ddof)),
        }

    def year_over_year_growth(self) -> np.ndarray:
        """Year-on-year growth rates: (P[t] - P[t-1]) / P[t-1]."""
        return (self.population_thousands[1:] - self.population_thousands[:-1]) / self.population_thousands[:-1]

    def cagr(self) -> float:
        """Compound Annual Growth Rate over the full series."""
        n_periods = len(self.population_thousands) - 1
        if n_periods < 1:
            raise ValueError("need at least 2 data points to compute CAGR")
        start, end = self.population_thousands[0], self.population_thousands[-1]
        return (end / start) ** (1 / n_periods) - 1


def _fibonacci(n: int) -> list[int]:
    """First n Fibonacci numbers, F(1) = F(2) = 1."""
    if n < 1:
        raise ValueError("n must be >= 1")
    seq = [1, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


class Forecaster(ABC):
    """Common interface for the district-population forecasters below."""

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def fit(self, years: Iterable[int], values: Iterable[float]) -> "Forecaster":
        """Fit the forecaster on historical (years, values)."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the next ``horizon`` periods after the fitted data."""

    @abstractmethod
    def fitted_values(self) -> np.ndarray:
        """In-sample fitted values over the years passed to ``fit()``."""

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fit() before predict()")

    @staticmethod
    def _require_positive_horizon(horizon: int) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")


class LinearTrendForecaster(Forecaster):
    """Forecast by extrapolating a least-squares linear trend (``np.polyfit``)."""

    def __init__(self) -> None:
        super().__init__()
        self._coeffs: np.ndarray | None = None
        self._last_year: int | None = None
        self._fit_years: np.ndarray | None = None

    def fit(self, years: Iterable[int], values: Iterable[float]) -> "LinearTrendForecaster":
        """Fit a least-squares line (``np.polyfit``, degree 1) to (years, values).

        Raises:
            ValueError: If fewer than 2 points are given.
        """
        years_arr = np.asarray(list(years), dtype=float)
        values_arr = np.asarray(list(values), dtype=float)
        if years_arr.size < 2:
            raise ValueError("need at least 2 points to fit a linear trend")
        self._coeffs = np.polyfit(years_arr, values_arr, 1)
        self._last_year = int(years_arr[-1])
        self._fit_years = years_arr
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Extrapolate the fitted line ``horizon`` years past the last fitted year."""
        self._require_fitted()
        self._require_positive_horizon(horizon)
        slope, intercept = self._coeffs
        future_years = np.arange(self._last_year + 1, self._last_year + 1 + horizon)
        return slope * future_years + intercept

    def fitted_values(self) -> np.ndarray:
        """In-sample values predicted by the fitted line over the years used in ``fit()``."""
        self._require_fitted()
        slope, intercept = self._coeffs
        return slope * self._fit_years + intercept


class ExponentialCAGRForecaster(Forecaster):
    """Forecast by compounding the fitted period's CAGR forward from the last value."""

    def __init__(self) -> None:
        super().__init__()
        self._last_value: float | None = None
        self._cagr: float | None = None
        self._n_fit: int | None = None

    def fit(self, years: Iterable[int], values: Iterable[float]) -> "ExponentialCAGRForecaster":
        """Compute the CAGR implied by the first and last values, and store the last value.

        Raises:
            ValueError: If fewer than 2 points are given, or the first value is not positive.
        """
        values_arr = np.asarray(list(values), dtype=float)
        n_periods = values_arr.size - 1
        if n_periods < 1:
            raise ValueError("need at least 2 points to fit a CAGR model")
        if values_arr[0] <= 0:
            raise ValueError("first value must be positive to compute a growth rate")
        self._last_value = float(values_arr[-1])
        self._cagr = float((values_arr[-1] / values_arr[0]) ** (1 / n_periods) - 1)
        self._n_fit = int(values_arr.size)
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Compound the fitted CAGR forward from the last value for ``horizon`` steps."""
        self._require_fitted()
        self._require_positive_horizon(horizon)
        steps = np.arange(1, horizon + 1)
        return self._last_value * (1 + self._cagr) ** steps

    def fitted_values(self) -> np.ndarray:
        """Exponential curve through the fitted endpoints (exact at both ends,
        not an OLS fit — CAGR is defined only by the first and last values)."""
        self._require_fitted()
        steps = np.arange(self._n_fit) - (self._n_fit - 1)
        return self._last_value * (1 + self._cagr) ** steps


class FibonacciRatioForecaster(Forecaster):
    """Scales the last value by successive Fibonacci ratios F(k+1)/F(k).

    A deliberately weak baseline reproducing a prior cohort's approach:
    Fibonacci ratios converge quickly to the golden ratio (~1.618), so this
    model implies roughly 62% compounding annual growth regardless of the
    actual historical data — see the notebook's critical assessment.
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_value: float | None = None
        self._n_fit: int | None = None
        self._fit_values: np.ndarray | None = None

    def fit(self, years: Iterable[int], values: Iterable[float]) -> "FibonacciRatioForecaster":
        """Store the last value and the fitted length (no trend estimation is performed).

        Raises:
            ValueError: If values is empty.
        """
        values_arr = np.asarray(list(values), dtype=float)
        if values_arr.size < 1:
            raise ValueError("need at least 1 point to fit")
        self._last_value = float(values_arr[-1])
        self._n_fit = int(values_arr.size)
        self._fit_values = values_arr
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Scale the last value by successive Fibonacci ratios, ``horizon`` steps forward."""
        self._require_fitted()
        self._require_positive_horizon(horizon)
        fib = _fibonacci(self._n_fit + horizon + 1)
        preds = np.empty(horizon)
        value = self._last_value
        for h in range(horizon):
            idx = self._n_fit + h
            ratio = fib[idx] / fib[idx - 1]
            value = value * ratio
            preds[h] = value
        return preds

    def fitted_values(self) -> np.ndarray:
        """Naive 1-step-back reference curve (this model has no natural
        in-sample fit of its own: it is defined purely as a forward
        projection from the final observed value)."""
        self._require_fitted()
        return np.concatenate([self._fit_values[:1], self._fit_values[:-1]])


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Mean Absolute Error."""
    a, p = np.asarray(list(actual), dtype=float), np.asarray(list(predicted), dtype=float)
    if a.shape != p.shape or a.size == 0:
        raise ValueError("actual and predicted must be non-empty and the same shape")
    return float(np.mean(np.abs(a - p)))


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Root Mean Squared Error."""
    a, p = np.asarray(list(actual), dtype=float), np.asarray(list(predicted), dtype=float)
    if a.shape != p.shape or a.size == 0:
        raise ValueError("actual and predicted must be non-empty and the same shape")
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Mean Absolute Percentage Error, as a percentage.

    Raises:
        ValueError: If actual and predicted differ in shape/are empty, or
            any actual value is zero (MAPE is undefined for a zero actual).
    """
    a, p = np.asarray(list(actual), dtype=float), np.asarray(list(predicted), dtype=float)
    if a.shape != p.shape or a.size == 0:
        raise ValueError("actual and predicted must be non-empty and the same shape")
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when an actual value is zero")
    return float(np.mean(np.abs((a - p) / a)) * 100)


def classrooms_needed(population_thousands: float, school_age_fraction: float = 0.18, classroom_size: int = 53) -> int:
    """Number of classrooms needed to serve a population's school-age children.

    Args:
        population_thousands: Total population, in thousands.
        school_age_fraction: Fraction of population that is primary-school age.
        classroom_size: Pupils per classroom.

    Returns:
        Classrooms needed (rounded up).

    Raises:
        ValueError: If population_thousands is negative.
    """
    if population_thousands < 0:
        raise ValueError("population_thousands cannot be negative")
    pupils = population_thousands * 1000 * school_age_fraction
    return int(np.ceil(pupils / classroom_size))


def bootstrap_prediction_interval(
    residuals: Iterable[float],
    point_forecast: Iterable[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap a prediction interval around a point forecast from residuals.

    For each of ``n_resamples`` draws and each forecast step, resamples one
    residual (with replacement, independently per step) and adds it to that
    step's point forecast, then takes empirical percentiles across draws.
    This assumes residuals are i.i.d. across horizon steps (no
    autocorrelation modelling) — a simplification stated explicitly in the
    notebook.

    Args:
        residuals: In-sample residuals (actual - fitted) from the chosen model.
        point_forecast: The point forecast for each future step.
        n_resamples: Number of bootstrap resamples (>= 1000 recommended).
        ci: Confidence level, e.g. 0.95 for a 95% interval.
        seed: Seed for ``np.random.default_rng`` (reproducibility).

    Returns:
        A tuple ``(lower, upper)`` of arrays, one bound per forecast step.

    Raises:
        ValueError: If residuals or point_forecast is empty.
    """
    residuals_arr = np.asarray(list(residuals), dtype=float)
    forecast_arr = np.asarray(list(point_forecast), dtype=float)
    if residuals_arr.size == 0 or forecast_arr.size == 0:
        raise ValueError("residuals and point_forecast cannot be empty")
    rng = np.random.default_rng(seed)
    horizon = forecast_arr.size
    draws = rng.choice(residuals_arr, size=(n_resamples, horizon), replace=True)
    paths = forecast_arr[np.newaxis, :] + draws
    alpha = 1 - ci
    lower = np.percentile(paths, 100 * alpha / 2, axis=0)
    upper = np.percentile(paths, 100 * (1 - alpha / 2), axis=0)
    return lower, upper
