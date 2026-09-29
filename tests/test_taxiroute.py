"""Pytest unit tests for Mini-Project 5 (src/taxiroute.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.taxiroute import (
    ExponentialSmoothingForecaster,
    LinearTrendForecaster,
    MovingAverageForecaster,
    Route,
    SeasonalNaiveForecaster,
    grid_search_alpha,
    simulate_weekly_demand,
    solve_equilibrium,
    vehicles_needed,
    walk_forward_mae,
)

NTINDA = [35, 40, 42, 50, 55, 60, 48, 52, 47, 45]


def test_route_daily_and_total_revenue() -> None:
    route = Route("Ntinda", NTINDA, fare=2000)
    daily = route.daily_revenue()
    assert daily[0] == pytest.approx(35 * 2000)
    assert route.total_revenue() == pytest.approx(sum(NTINDA) * 2000)
    assert len(route) == 10


def test_route_stats() -> None:
    route = Route("Ntinda", NTINDA, fare=2000)
    stats = route.stats()
    assert stats["mean"] == pytest.approx(sum(NTINDA) / len(NTINDA))


def test_route_rejects_empty_passengers() -> None:
    """Edge case: an empty passenger series must raise."""
    with pytest.raises(ValueError):
        Route("Bad", [], fare=2000)


def test_route_rejects_negative_fare() -> None:
    with pytest.raises(ValueError):
        Route("Bad", NTINDA, fare=0)


def test_solve_equilibrium_matches_hand_calculation() -> None:
    # Qd = 120 - 0.02P, Qs = 10 + 0.03P => P* = 110/0.05 = 2200, Q* = 120-0.02*2200 = 76
    q_star, p_star = solve_equilibrium(a=120, b=0.02, c=10, d=0.03)
    assert p_star == pytest.approx(2200.0)
    assert q_star == pytest.approx(76.0)


def test_solve_equilibrium_rejects_parallel_lines() -> None:
    """Edge case: b + d == 0 means no unique equilibrium."""
    with pytest.raises(ValueError):
        solve_equilibrium(a=100, b=0.01, c=10, d=-0.01)


def test_moving_average_forecaster() -> None:
    model = MovingAverageForecaster(window=3).fit([10, 20, 30])
    assert model.predict(1)[0] == pytest.approx(20.0)


def test_moving_average_forecaster_rejects_insufficient_data() -> None:
    """Edge case: fewer points than the window must raise."""
    with pytest.raises(ValueError):
        MovingAverageForecaster(window=3).fit([10, 20])


def test_exponential_smoothing_forecaster_constant_series() -> None:
    model = ExponentialSmoothingForecaster(alpha=0.5).fit([50, 50, 50, 50])
    assert model.predict(1)[0] == pytest.approx(50.0)


def test_exponential_smoothing_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError):
        ExponentialSmoothingForecaster(alpha=0.0)
    with pytest.raises(ValueError):
        ExponentialSmoothingForecaster(alpha=1.5)


def test_linear_trend_forecaster_matches_hand_calculation() -> None:
    model = LinearTrendForecaster().fit([10, 20, 30, 40])
    forecast = model.predict(horizon=2)
    assert forecast[0] == pytest.approx(50.0)
    assert forecast[1] == pytest.approx(60.0)


def test_linear_trend_rejects_insufficient_data() -> None:
    """Edge case: fewer than 2 points cannot fit a line."""
    with pytest.raises(ValueError):
        LinearTrendForecaster().fit([10])


def test_seasonal_naive_forecaster() -> None:
    series = list(range(14))  # 0..13
    model = SeasonalNaiveForecaster(season_length=7).fit(series)
    forecast = model.predict(horizon=1)
    assert forecast[0] == pytest.approx(series[-7])


def test_predict_before_fit_raises() -> None:
    """Edge case: predicting an unfit forecaster must raise."""
    with pytest.raises(RuntimeError):
        MovingAverageForecaster().predict(1)


def test_walk_forward_mae_zero_for_constant_series() -> None:
    constant = [42.0] * 10
    mae = walk_forward_mae(lambda: MovingAverageForecaster(window=3), constant, start=3)
    assert mae == pytest.approx(0.0)


def test_grid_search_alpha_returns_valid_alpha() -> None:
    series = NTINDA
    best_alpha, best_mae = grid_search_alpha(series, start=3, alphas=[0.1, 0.3, 0.5, 0.9])
    assert best_alpha in [0.1, 0.3, 0.5, 0.9]
    assert best_mae >= 0


def test_vehicles_needed_matches_hand_calculation() -> None:
    # 100 passengers, 15% buffer -> 115 / (8*14=112) -> ceil(1.0268) = 2
    assert vehicles_needed(100, trips_per_day=8, seats=14, buffer=0.15) == 2


def test_vehicles_needed_rejects_negative_forecast() -> None:
    """Edge case: negative forecast passengers must raise."""
    with pytest.raises(ValueError):
        vehicles_needed(-5)


def test_simulate_weekly_demand_reproducible_and_shaped() -> None:
    a = simulate_weekly_demand(base_mean=50, n_days=60, seed=1)
    b = simulate_weekly_demand(base_mean=50, n_days=60, seed=1)
    assert np.array_equal(a, b)
    assert len(a) == 60
    assert np.all(a >= 0)


def test_simulate_weekly_demand_friday_busier_than_sunday_on_average() -> None:
    values = simulate_weekly_demand(base_mean=50, n_days=700, noise_std=1.0, seed=2)
    weekday = np.arange(700) % 7
    friday_mean = values[weekday == 4].mean()
    sunday_mean = values[weekday == 6].mean()
    assert friday_mean > sunday_mean
