"""Pytest unit tests for Mini-Project 2 (src/microgrid.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microgrid import (
    HybridMicroGrid,
    MicroGrid,
    generate_demand_csv,
    load_demand_csv,
    read_demand_interactive,
    read_positive_float,
)


@pytest.fixture
def grid() -> MicroGrid:
    return MicroGrid()


def test_solve_day_matches_hand_solution(grid: MicroGrid) -> None:
    """3x+2y=D1, 4x+y=D2 with D1=160, D2=190 has the exact solution x=44, y=14."""
    x, y = grid.solve_day(160, 190)
    assert x == pytest.approx(44.0)
    assert y == pytest.approx(14.0)


def test_solve_many_loop_and_vectorised_agree(grid: MicroGrid) -> None:
    demand = np.array([[160.0, 100.0, 200.0], [190.0, 80.0, 150.0]])
    loop_result = grid.solve_many_loop(demand)
    vec_result = grid.solve_many_vectorised(demand)
    assert np.allclose(loop_result, vec_result)


def test_determinant_and_condition_number(grid: MicroGrid) -> None:
    # det([[3,2],[4,1]]) = 3*1 - 2*4 = -5
    assert grid.determinant == pytest.approx(-5.0)
    assert grid.condition_number > 0
    assert "well-posed" in grid.describe_conditioning()


def test_detect_infeasible_flags_negative_dispatch(grid: MicroGrid) -> None:
    dispatch = np.array([[10.0, -1.0, 5.0], [3.0, 2.0, -0.5]])
    mask = grid.detect_infeasible(dispatch)
    assert list(mask) == [False, True, True]


def test_fix_infeasible_nnls_removes_negatives(grid: MicroGrid) -> None:
    # A demand pair that pushes the raw linear solve negative.
    demand = np.array([[5.0, 160.0], [40.0, 190.0]])
    dispatch = grid.solve_many_vectorised(demand)
    corrected, flagged = grid.fix_infeasible_nnls(demand, dispatch)
    assert len(flagged) >= 1
    assert np.all(corrected >= -1e-9)


def test_usage_stats_normal_case(grid: MicroGrid) -> None:
    stats = grid.usage_stats(np.array([10.0, 20.0, 30.0]))
    assert stats["mean"] == pytest.approx(20.0)
    assert stats["variance"] == pytest.approx(100.0)
    assert stats["stdev"] == pytest.approx(10.0)


def test_usage_stats_empty_raises(grid: MicroGrid) -> None:
    """Edge case: empty input must raise, not silently return garbage."""
    with pytest.raises(ValueError):
        grid.usage_stats(np.array([]))


def test_daily_and_monthly_cost(grid: MicroGrid) -> None:
    x = np.array([10.0, 20.0])
    y = np.array([5.0, 0.0])
    daily = grid.daily_cost(x, y)
    assert daily[0] == pytest.approx(10 * 150 + 5 * 450)
    assert grid.monthly_cost(x, y) == pytest.approx(daily.sum())


def test_hybrid_microgrid_solves_3x3_system() -> None:
    hybrid = HybridMicroGrid()
    x, y, z = hybrid.solve_day(160, 190, 90)
    solved = hybrid.coefficients @ np.array([x, y, z])
    assert np.allclose(solved, [160, 190, 90])


def test_hybrid_microgrid_requires_d3() -> None:
    hybrid = HybridMicroGrid()
    with pytest.raises(ValueError):
        hybrid.solve_day(160, 190)


def test_read_positive_float_rejects_then_accepts() -> None:
    responses = iter(["", "abc", "-5", "12.5"])
    value = read_positive_float("prompt: ", input_func=lambda _: next(responses))
    assert value == pytest.approx(12.5)


def test_read_demand_interactive_returns_both_values() -> None:
    responses = iter(["100", "80"])
    d1, d2 = read_demand_interactive(input_func=lambda _: next(responses))
    assert (d1, d2) == (100.0, 80.0)


def test_generate_and_load_demand_csv_roundtrip(tmp_path) -> None:
    csv_path = tmp_path / "demand_30days.csv"
    generated = generate_demand_csv(csv_path, n_days=30, seed=7)
    loaded = load_demand_csv(csv_path)
    assert len(generated) == 30
    assert list(loaded.columns) == ["day", "weekday", "D1", "D2"]
    assert (loaded["D1"] > 0).all()
    assert (loaded["D2"] > 0).all()


def test_load_demand_csv_missing_file_raises(tmp_path) -> None:
    """Edge case: loading a nonexistent CSV should raise, not fail silently."""
    with pytest.raises(FileNotFoundError):
        load_demand_csv(tmp_path / "does_not_exist.csv")


def test_monte_carlo_sensitivity_reproducible(grid: MicroGrid) -> None:
    result_a = grid.monte_carlo_sensitivity(160, 190, seed=42, n_draws=500)
    result_b = grid.monte_carlo_sensitivity(160, 190, seed=42, n_draws=500)
    assert np.array_equal(result_a["x_samples"], result_b["x_samples"])
    assert len(result_a["x_samples"]) == 500
