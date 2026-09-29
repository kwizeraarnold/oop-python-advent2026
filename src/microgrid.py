"""Reusable OOP logic for Mini-Project 2: Solar Micro-Grid Dispatch Planner.

Models a rural health-centre micro-grid (Kasese) that must split two daily
loads between solar panels (x) and batteries (y):

    3x + 2y = D1   (daytime load, kWh)
    4x +  y = D2   (critical-equipment load, kWh)

``MicroGrid`` wraps the 2x2 linear system; ``HybridMicroGrid`` extends it
with a diesel generator (z) and a third constraint, solving a 3x3 system.
"""

from __future__ import annotations

import statistics
import timeit
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.linalg import solve as linalg_solve
from scipy.optimize import nnls


class MicroGrid:
    """A two-source (solar + battery) micro-grid dispatch model.

    The grid must satisfy two daily load constraints by choosing how much
    energy to draw from solar (``x``) and battery (``y``):

        3x + 2y = D1  (daytime load, kWh)
        4x +  y = D2  (critical-equipment load, kWh)

    Attributes:
        coefficients: The 2x2 coefficient matrix A in A @ [x, y] = [D1, D2].
        solar_price: Cost of solar energy in UGX/kWh.
        battery_price: Cost of battery energy in UGX/kWh.
    """

    def __init__(self, solar_price: float = 150.0, battery_price: float = 450.0) -> None:
        """Initialise the micro-grid with its fixed coefficient matrix and prices.

        Args:
            solar_price: UGX per kWh drawn from solar.
            battery_price: UGX per kWh drawn from battery.
        """
        self.coefficients = np.array([[3.0, 2.0], [4.0, 1.0]])
        self.solar_price = solar_price
        self.battery_price = battery_price

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(solar_price={self.solar_price}, "
            f"battery_price={self.battery_price})"
        )

    @property
    def determinant(self) -> float:
        """Determinant of the coefficient matrix."""
        return float(np.linalg.det(self.coefficients))

    @property
    def condition_number(self) -> float:
        """2-norm condition number of the coefficient matrix."""
        return float(np.linalg.cond(self.coefficients))

    def describe_conditioning(self) -> str:
        """Human-readable explanation of well-posedness for this system."""
        det = self.determinant
        cond = self.condition_number
        posed = "well-posed (unique solution exists)" if abs(det) > 1e-9 else "singular (no unique solution)"
        sensitivity = (
            "well-conditioned: small changes in demand cause proportionally small "
            "changes in the dispatch" if cond < 10 else
            "poorly conditioned: small changes in demand could be amplified in the dispatch"
        )
        return (
            f"det(A) = {det:.4f} -> the system is {posed}. "
            f"cond(A) = {cond:.4f} -> the system is {sensitivity}."
        )

    def solve_day(self, d1: float, d2: float) -> tuple[float, float]:
        """Solve for (solar, battery) dispatch on a single day.

        Args:
            d1: Daytime load in kWh.
            d2: Critical-equipment load in kWh.

        Returns:
            A ``(x, y)`` tuple of solar and battery kWh.
        """
        rhs = np.array([d1, d2], dtype=float)
        x, y = linalg_solve(self.coefficients, rhs)
        return float(x), float(y)

    def solve_many_loop(self, demand: np.ndarray) -> np.ndarray:
        """Solve a 2xN block of daily demands one day at a time (Python loop).

        Args:
            demand: Array of shape (2, N) with rows [D1, D2].

        Returns:
            Array of shape (2, N) with rows [x, y].
        """
        n_days = demand.shape[1]
        result = np.empty((2, n_days))
        for i in range(n_days):
            result[:, i] = linalg_solve(self.coefficients, demand[:, i])
        return result

    def solve_many_vectorised(self, demand: np.ndarray) -> np.ndarray:
        """Solve a 2xN block of daily demands in a single vectorised call.

        Args:
            demand: Array of shape (2, N) with rows [D1, D2].

        Returns:
            Array of shape (2, N) with rows [x, y].
        """
        return linalg_solve(self.coefficients, demand)

    def time_solve_methods(self, demand: np.ndarray, number: int = 100) -> dict[str, float]:
        """Time the loop-based and vectorised solvers on the same demand block.

        Args:
            demand: Array of shape (2, N).
            number: Number of repeats passed to ``timeit``.

        Returns:
            Dict with ``loop_seconds`` and ``vectorised_seconds`` (per ``number`` runs).
        """
        loop_time = timeit.timeit(lambda: self.solve_many_loop(demand), number=number)
        vec_time = timeit.timeit(lambda: self.solve_many_vectorised(demand), number=number)
        return {"loop_seconds": loop_time, "vectorised_seconds": vec_time}

    @staticmethod
    def detect_infeasible(dispatch: np.ndarray) -> np.ndarray:
        """Flag days where either x or y is negative (physically infeasible).

        Args:
            dispatch: Array of shape (2, N) with rows [x, y].

        Returns:
            Boolean array of shape (N,), True where the day is infeasible.
        """
        return (dispatch[0] < 0) | (dispatch[1] < 0)

    def fix_infeasible_nnls(self, demand: np.ndarray, dispatch: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """Re-solve infeasible days with non-negative least squares.

        Days with a negative solar or battery dispatch are re-solved with
        ``scipy.optimize.nnls``, which finds the closest non-negative (x, y)
        that minimises the residual against the coefficient matrix. This is
        preferred over naive clipping because it keeps the *best available*
        non-negative solution rather than an arbitrary truncation.

        Args:
            demand: Array of shape (2, N) with rows [D1, D2].
            dispatch: Array of shape (2, N) with rows [x, y] from a linear solve.

        Returns:
            A tuple of (corrected dispatch array of shape (2, N), list of
            flagged day indices).
        """
        corrected = dispatch.copy()
        infeasible_mask = self.detect_infeasible(dispatch)
        flagged_days = list(np.flatnonzero(infeasible_mask))
        for day in flagged_days:
            solution, _residual = nnls(self.coefficients, demand[:, day])
            corrected[:, day] = solution
        return corrected, flagged_days

    @staticmethod
    def usage_stats(values: np.ndarray) -> dict[str, float]:
        """Mean, variance and standard deviation of a usage series via ``statistics``.

        Args:
            values: 1-D array of kWh values.

        Returns:
            Dict with ``mean``, ``variance`` and ``stdev``.

        Raises:
            ValueError: If ``values`` is empty.
        """
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            raise ValueError("usage_stats requires at least one value")
        data = values.tolist()
        if len(data) == 1:
            return {"mean": data[0], "variance": 0.0, "stdev": 0.0}
        return {
            "mean": statistics.mean(data),
            "variance": statistics.variance(data),
            "stdev": statistics.stdev(data),
        }

    def daily_cost(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Daily energy cost in UGX for given solar/battery dispatch arrays."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return x * self.solar_price + y * self.battery_price

    def monthly_cost(self, x: np.ndarray, y: np.ndarray) -> float:
        """Total energy cost in UGX summed over the given dispatch arrays."""
        return float(np.sum(self.daily_cost(x, y)))

    def monte_carlo_sensitivity(
        self,
        d1: float,
        d2: float,
        pct: float = 0.05,
        n_draws: int = 1000,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """Perturb (D1, D2) by +/- pct and re-solve to see how (x, y) move.

        Args:
            d1: Baseline daytime load, kWh.
            d2: Baseline critical-equipment load, kWh.
            pct: Fractional perturbation applied uniformly, e.g. 0.05 for +/-5%.
            n_draws: Number of Monte Carlo draws.
            seed: Seed for the random number generator (reproducibility).

        Returns:
            Dict with arrays ``x_samples`` and ``y_samples`` of length ``n_draws``.
        """
        rng = np.random.default_rng(seed)
        d1_samples = d1 * (1 + rng.uniform(-pct, pct, size=n_draws))
        d2_samples = d2 * (1 + rng.uniform(-pct, pct, size=n_draws))
        demand = np.vstack([d1_samples, d2_samples])
        dispatch = self.solve_many_vectorised(demand)
        return {"x_samples": dispatch[0], "y_samples": dispatch[1]}


class HybridMicroGrid(MicroGrid):
    """A three-source micro-grid that adds a diesel generator (z).

    Extends the base 2x2 system with a third source and a third constraint,
    a total daily energy-balance equation:

        3x + 2y +  z = D1  (daytime load, kWh)
        4x +  y + 2z = D2  (critical-equipment load, kWh; diesel weighted 2x
                            because the generator is assumed twice as
                            efficient at meeting spiky critical-equipment
                            demand as it is at meeting the daytime load)
         x +  y +  z = D3  (total daily energy balance, kWh)
    """

    def __init__(self, solar_price: float = 150.0, battery_price: float = 450.0, diesel_price: float = 900.0) -> None:
        """Initialise the hybrid grid with its 3x3 coefficient matrix and prices.

        Args:
            solar_price: UGX per kWh drawn from solar.
            battery_price: UGX per kWh drawn from battery.
            diesel_price: UGX per kWh drawn from diesel (illustrative, most expensive).
        """
        super().__init__(solar_price=solar_price, battery_price=battery_price)
        self.diesel_price = diesel_price
        self.coefficients = np.array([
            [3.0, 2.0, 1.0],
            [4.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
        ])

    def solve_day(self, d1: float, d2: float, d3: float | None = None) -> tuple[float, float, float]:
        """Solve for (solar, battery, diesel) dispatch on a single day.

        Args:
            d1: Daytime load in kWh.
            d2: Critical-equipment load in kWh.
            d3: Total daily energy balance in kWh. Required for the hybrid
                system (unlike the base class, this is not optional in practice).

        Returns:
            An ``(x, y, z)`` tuple of solar, battery and diesel kWh.

        Raises:
            ValueError: If ``d3`` is not provided.
        """
        if d3 is None:
            raise ValueError("HybridMicroGrid.solve_day requires d3 (total energy balance)")
        rhs = np.array([d1, d2, d3], dtype=float)
        x, y, z = linalg_solve(self.coefficients, rhs)
        return float(x), float(y), float(z)

    def daily_cost(self, x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
        """Daily energy cost in UGX, including diesel if ``z`` is given."""
        base = super().daily_cost(x, y)
        if z is not None:
            base = base + np.asarray(z, dtype=float) * self.diesel_price
        return base


def read_positive_float(prompt: str, input_func: Callable[[str], str] = input) -> float:
    """Read a positive float from the console, re-prompting on bad input.

    Rejects empty input, non-numeric input, and negative values.

    Args:
        prompt: Text shown to the user.
        input_func: Injectable input function (defaults to built-in ``input``,
            overridden in tests to avoid real console interaction).

    Returns:
        A validated, non-negative float.
    """
    while True:
        raw = input_func(prompt).strip()
        if not raw:
            print("Input cannot be empty. Try again.")
            continue
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a numeric value. Try again.")
            continue
        if value < 0:
            print("Value cannot be negative. Try again.")
            continue
        return value


def read_demand_interactive(input_func: Callable[[str], str] = input) -> tuple[float, float]:
    """Prompt the user for D1 and D2 with validation and re-prompting.

    Args:
        input_func: Injectable input function (defaults to built-in ``input``).

    Returns:
        A ``(d1, d2)`` tuple of validated demand values.
    """
    d1 = read_positive_float("Enter daytime load D1 (kWh): ", input_func=input_func)
    d2 = read_positive_float("Enter critical-equipment load D2 (kWh): ", input_func=input_func)
    return d1, d2


def generate_demand_csv(path: str | Path, n_days: int = 30, seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic 30-day demand CSV with a weekly pattern and noise.

    D1 (daytime load) and D2 (critical-equipment load) both get a higher
    baseline on weekdays (clinic busier, more procedures) and a lower one on
    weekends, plus Gaussian noise.

    Args:
        path: Destination CSV path.
        n_days: Number of days to generate.
        seed: Seed for ``np.random.default_rng`` (reproducibility).

    Returns:
        The generated DataFrame (also written to ``path``).
    """
    rng = np.random.default_rng(seed)
    days = np.arange(1, n_days + 1)
    weekday = (days - 1) % 7  # 0=Mon ... 6=Sun

    is_weekend = weekday >= 5
    d1_base = np.where(is_weekend, 30.0, 45.0)
    d2_base = np.where(is_weekend, 20.0, 26.0)

    d1 = d1_base + rng.normal(0, 4.0, size=n_days)
    d2 = d2_base + rng.normal(0, 2.5, size=n_days)
    d1 = np.clip(d1, 5, None)
    d2 = np.clip(d2, 5, None)

    df = pd.DataFrame({"day": days, "weekday": weekday, "D1": d1.round(2), "D2": d2.round(2)})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_demand_csv(path: str | Path) -> pd.DataFrame:
    """Load a demand CSV produced by :func:`generate_demand_csv`.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with columns ``day``, ``weekday``, ``D1``, ``D2``.

    Raises:
        ValueError: If the file is empty or missing required columns.
    """
    df = pd.read_csv(path)
    required = {"day", "D1", "D2"}
    if df.empty:
        raise ValueError(f"{path} contains no data rows")
    if not required.issubset(df.columns):
        raise ValueError(f"{path} is missing required columns {required - set(df.columns)}")
    return df
