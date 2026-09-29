"""Pytest unit tests for Mini-Project 3 (src/fishstock.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fishstock import FishStock, PriceModel, RiskAssessor, fibonacci_stock, revenue_stats


def test_fibonacci_stock_normal_case() -> None:
    seq = fibonacci_stock(10)
    assert list(seq[:6]) == [1.0, 1.0, 2.0, 3.0, 5.0, 8.0]
    assert len(seq) == 10


def test_fibonacci_stock_rejects_invalid_n() -> None:
    """Edge case: n < 1 should raise, not silently return an empty/garbage array."""
    with pytest.raises(ValueError):
        fibonacci_stock(0)


@pytest.fixture
def stock() -> FishStock:
    return FishStock(r=0.4, K=10_000.0, N0=4_000.0)


def test_fishstock_first_step_matches_hand_calculation(stock: FishStock) -> None:
    # N1 = N0 + r*N0*(1-N0/K) - h*N0, with h=0: N1 = 4000 + 0.4*4000*(1-0.4) = 4960
    trajectory, harvest = stock.simulate(weeks=1, h=0.0)
    assert trajectory[0] == pytest.approx(4000.0)
    assert trajectory[1] == pytest.approx(4960.0)
    assert harvest[0] == pytest.approx(0.0)


def test_fishstock_harvest_reduces_stock_relative_to_no_harvest(stock: FishStock) -> None:
    no_harvest, _ = stock.simulate(weeks=52, h=0.0)
    harvested, _ = stock.simulate(weeks=52, h=0.20)
    assert harvested[-1] < no_harvest[-1]


def test_fishstock_msy_property(stock: FishStock) -> None:
    assert stock.msy == pytest.approx(0.4 * 10_000 / 4)


def test_fishstock_rejects_invalid_params() -> None:
    """Edge cases: non-positive K, negative N0, negative r must all raise."""
    with pytest.raises(ValueError):
        FishStock(K=0)
    with pytest.raises(ValueError):
        FishStock(N0=-1)
    with pytest.raises(ValueError):
        FishStock(r=-0.1)


def test_fishstock_closed_weeks_zero_harvest(stock: FishStock) -> None:
    _trajectory, harvest = stock.simulate(weeks=10, h=0.5, closed_weeks={0, 1, 2})
    assert harvest[0] == pytest.approx(0.0)
    assert harvest[1] == pytest.approx(0.0)
    assert harvest[2] == pytest.approx(0.0)
    assert harvest[3] > 0.0


def test_fishstock_rejects_invalid_h(stock: FishStock) -> None:
    with pytest.raises(ValueError):
        stock.simulate(h=1.5)


def test_price_model_respects_bounds() -> None:
    model = PriceModel(start_price=12_000.0, low=9_000.0, high=16_000.0, step_std=2_000.0)
    rng = np.random.default_rng(1)
    prices = model.simulate(weeks=200, rng=rng)
    assert np.all(prices >= 9_000.0)
    assert np.all(prices <= 16_000.0)


def test_price_model_reproducible_with_same_seed() -> None:
    model = PriceModel()
    prices_a = model.simulate(weeks=52, rng=np.random.default_rng(7))
    prices_b = model.simulate(weeks=52, rng=np.random.default_rng(7))
    assert np.array_equal(prices_a, prices_b)


def test_price_model_rejects_start_price_outside_bounds() -> None:
    """Edge case: an invalid start price must raise at construction time."""
    with pytest.raises(ValueError):
        PriceModel(start_price=20_000.0, low=9_000.0, high=16_000.0)


def test_risk_assessor_classify_thresholds() -> None:
    assessor = RiskAssessor(low_cv=0.15, high_cv=0.35)
    assert assessor.classify(0.05) == "Low"
    assert assessor.classify(0.20) == "Moderate"
    assert assessor.classify(0.50) == "High"


def test_risk_assessor_classify_rejects_negative_cv() -> None:
    assessor = RiskAssessor()
    with pytest.raises(ValueError):
        assessor.classify(-0.1)


def test_risk_assessor_monte_carlo_reproducible_and_sized(stock: FishStock) -> None:
    price_model = PriceModel()
    assessor = RiskAssessor()
    revenues_a = assessor.simulate_annual_revenue(stock, price_model, h=0.10, n_sims=500, seed=42)
    revenues_b = assessor.simulate_annual_revenue(stock, price_model, h=0.10, n_sims=500, seed=42)
    assert len(revenues_a) == 500
    assert np.array_equal(revenues_a, revenues_b)


def test_value_at_risk_known_array() -> None:
    revenues = np.arange(1, 101, dtype=float)  # 1..100
    var = RiskAssessor.value_at_risk(revenues, pct=0.05)
    assert var == pytest.approx(np.percentile(revenues, 5))


def test_value_at_risk_empty_raises() -> None:
    """Edge case: VaR of an empty revenue array should raise."""
    with pytest.raises(ValueError):
        RiskAssessor.value_at_risk(np.array([]))


def test_revenue_stats_normal_case() -> None:
    stats = revenue_stats([100.0, 200.0, 300.0])
    assert stats["mean"] == pytest.approx(200.0)
    assert stats["median"] == pytest.approx(200.0)
    assert stats["cv"] == pytest.approx(stats["stdev"] / stats["mean"])


def test_revenue_stats_empty_raises() -> None:
    """Edge case: empty revenue series should raise, not silently return garbage."""
    with pytest.raises(ValueError):
        revenue_stats([])
