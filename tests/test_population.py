"""Pytest unit tests for Mini-Project 1 (src/population.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.population import (
    DistrictPopulation,
    ExponentialCAGRForecaster,
    FibonacciRatioForecaster,
    LinearTrendForecaster,
    bootstrap_prediction_interval,
    classrooms_needed,
    mae,
    mape,
    rmse,
)

YEARS = list(range(2015, 2025))
KAMPALA = [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800]


def test_district_population_len_and_repr() -> None:
    d = DistrictPopulation("Kampala", YEARS, KAMPALA)
    assert len(d) == 10
    assert "Kampala" in repr(d)


def test_district_population_rejects_mismatched_lengths() -> None:
    """Edge case: years and population lengths must match."""
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", YEARS, KAMPALA[:5])


def test_district_population_rejects_negative_population() -> None:
    bad = KAMPALA.copy()
    bad[0] = -1
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", YEARS, bad)


def test_stats_statistics_vs_numpy_ddof() -> None:
    d = DistrictPopulation("Kampala", YEARS, KAMPALA)
    stats_module = d.stats_statistics_module()
    numpy_pop = d.stats_numpy(ddof=0)
    numpy_sample = d.stats_numpy(ddof=1)
    # statistics.variance is sample variance (ddof=1); should match np.var(ddof=1), not ddof=0.
    assert stats_module["variance"] == pytest.approx(numpy_sample["variance"])
    assert stats_module["variance"] != pytest.approx(numpy_pop["variance"])


def test_year_over_year_growth() -> None:
    d = DistrictPopulation("Kampala", YEARS, KAMPALA)
    growth = d.year_over_year_growth()
    assert len(growth) == 9
    assert growth[0] == pytest.approx((1250 - 1200) / 1200)


def test_cagr_matches_hand_calculation() -> None:
    d = DistrictPopulation("Kampala", YEARS, KAMPALA)
    expected = (1800 / 1200) ** (1 / 9) - 1
    assert d.cagr() == pytest.approx(expected)


def test_cagr_requires_two_points() -> None:
    """Edge case: a single-point series cannot compute CAGR."""
    d = DistrictPopulation("OnePoint", [2020], [500])
    with pytest.raises(ValueError):
        d.cagr()


def test_linear_trend_forecaster_matches_hand_calculation() -> None:
    years = [2015, 2016, 2017, 2018]
    values = [100, 110, 120, 130]  # perfectly linear, slope=10/yr
    model = LinearTrendForecaster().fit(years, values)
    forecast = model.predict(horizon=2)
    assert forecast[0] == pytest.approx(140.0)
    assert forecast[1] == pytest.approx(150.0)


def test_exponential_cagr_forecaster() -> None:
    years = [2015, 2016, 2017]
    values = [100, 110, 121]  # exactly 10% growth per year
    model = ExponentialCAGRForecaster().fit(years, values)
    forecast = model.predict(horizon=1)
    assert forecast[0] == pytest.approx(121 * 1.10, rel=1e-6)


def test_linear_trend_fitted_values_matches_actual_on_perfect_line() -> None:
    years = [2015, 2016, 2017, 2018]
    values = [100, 110, 120, 130]
    model = LinearTrendForecaster().fit(years, values)
    assert np.allclose(model.fitted_values(), values)


def test_exponential_cagr_fitted_values_matches_endpoints() -> None:
    years = [2015, 2016, 2017]
    values = [100, 110, 121]
    model = ExponentialCAGRForecaster().fit(years, values)
    fitted = model.fitted_values()
    assert fitted[0] == pytest.approx(100.0)
    assert fitted[-1] == pytest.approx(121.0)


def test_fibonacci_ratio_forecaster_matches_hand_calculation() -> None:
    years = [2015, 2016, 2017]
    values = [10, 20, 30]  # n_fit = 3
    model = FibonacciRatioForecaster().fit(years, values)
    forecast = model.predict(horizon=2)
    # fib = [1,1,2,3,5,...]; idx=3 -> fib[3]/fib[2] = 3/2 = 1.5; idx=4 -> fib[4]/fib[3] = 5/3
    assert forecast[0] == pytest.approx(30 * 1.5)
    assert forecast[1] == pytest.approx(30 * 1.5 * (5 / 3))


def test_predict_before_fit_raises() -> None:
    """Edge case: predicting an unfit forecaster must raise."""
    with pytest.raises(RuntimeError):
        LinearTrendForecaster().predict(1)


def test_mae_rmse_mape_known_values() -> None:
    actual = [100, 200, 300]
    predicted = [110, 190, 300]
    assert mae(actual, predicted) == pytest.approx((10 + 10 + 0) / 3)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt((100 + 100 + 0) / 3))
    assert mape(actual, predicted) == pytest.approx(((10 / 100) + (10 / 200) + 0) / 3 * 100)


def test_mape_rejects_zero_actual() -> None:
    """Edge case: MAPE is undefined when an actual value is zero."""
    with pytest.raises(ValueError):
        mape([0, 100], [10, 100])


def test_classrooms_needed_matches_hand_calculation() -> None:
    # 1000 thousand -> 1,000,000 people * 0.18 = 180,000 pupils / 53 = 3396.2 -> 3397
    assert classrooms_needed(1000, school_age_fraction=0.18, classroom_size=53) == 3397


def test_classrooms_needed_rejects_negative_population() -> None:
    with pytest.raises(ValueError):
        classrooms_needed(-1)


def test_bootstrap_prediction_interval_reproducible_and_bounds_forecast() -> None:
    residuals = [-5, -3, -1, 1, 3, 5]
    point_forecast = [100, 110, 120]
    lower_a, upper_a = bootstrap_prediction_interval(residuals, point_forecast, n_resamples=1000, seed=42)
    lower_b, upper_b = bootstrap_prediction_interval(residuals, point_forecast, n_resamples=1000, seed=42)
    assert np.array_equal(lower_a, lower_b)
    assert np.all(lower_a <= point_forecast)
    assert np.all(upper_a >= point_forecast)


def test_bootstrap_prediction_interval_rejects_empty_residuals() -> None:
    """Edge case: empty residuals cannot be bootstrapped."""
    with pytest.raises(ValueError):
        bootstrap_prediction_interval([], [100])
