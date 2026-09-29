# OOP with Python — Assignment 2, Advent 2026

MSCS & MSDS OOP with Python mini-projects (see `Assignment_1_Advent_2026_MiniProjects.pdf` for the full brief and rubric).

## Status

| Mini-project | Notebook | Status |
|---|---|---|
| 1 — UBOS District Population Forecaster | `project1_population.ipynb` | Done |
| 2 — Solar Micro-Grid Dispatch Planner | `project2_microgrid.ipynb` | Done |
| 3 — Lake Victoria Fish Stock & Export Risk Model | `project3_fishstock.ipynb` | Done |
| 4 — Rainfall Pattern & Crop Suitability Analyser | `project4_rainfall.ipynb` | Done |
| 5 — Taxi Route Revenue, Pricing & Fleet Planner | `project5_taxiroute.ipynb` | Done |

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Running

```bash
# Launch any notebook, e.g.
jupyter notebook project1_population.ipynb
# then Kernel -> Restart & Run All (repeat for project2_microgrid.ipynb, project3_fishstock.ipynb,
# project4_rainfall.ipynb, project5_taxiroute.ipynb)

# Run tests
pytest tests/ -v
```

Reusable domain logic lives in `src/` (one module per mini-project) and is imported into the corresponding notebook. Each notebook is self-contained and reproducible: random generators are seeded (`np.random.default_rng(seed)`), and `Restart & Run All` runs top to bottom without errors.

## Mini-Project 1: UBOS District Population Forecaster — summary of findings

`src/population.py` implements `DistrictPopulation`, a `Forecaster` ABC with `LinearTrendForecaster`, `ExponentialCAGRForecaster` and `FibonacciRatioForecaster` subclasses, plus `mae`/`rmse`/`mape`, `classrooms_needed` and `bootstrap_prediction_interval` helpers. Five districts are modelled: Kampala, Wakiso and Gulu (the assignment's illustrative figures) plus two added here, Mbale and Jinja (also illustrative). Key results from `project1_population.ipynb`:

- The backtest (train 2015–2021, test 2022–2024) split cleanly by growth shape: exponential/CAGR won for the three accelerating districts (Kampala, Wakiso, Gulu — MAPE under 2.1%), linear trend won for the two near-linear ones (Mbale, Jinja — MAPE under 0.7%).
- The Fibonacci-ratio model failed catastrophically on every district (MAPE 145–168%) — an empirical confirmation, not just a theoretical claim, of the notebook's critique that Fibonacci ratios converge to ~62% annual compounding, unrelated to any real demographic growth mechanism.
- Forecast variance was lower than historical variance for every district, illustrating that forecast-series variance reflects a model's smoothness, not its uncertainty — which is why the bootstrap prediction intervals (1,000 resamples) exist as a separate, explicit uncertainty estimate.
- Planning impact varies widely: Wakiso needs the most additional classrooms by 2029 (2,088, driven by its highest CAGR at 6.47%/year) while Jinja needs the fewest (132).

Full analysis, assumptions and limitations are in the notebook's **Findings & Limitations** section.

## Mini-Project 2: Solar Micro-Grid Dispatch Planner — summary of findings

`src/microgrid.py` implements `MicroGrid` (2x2 solar/battery dispatch system) and `HybridMicroGrid` (3x3 extension with a diesel generator). Key results from `project2_microgrid.ipynb`:

- The base 2x2 system is well-posed (det = -5) and moderately well-conditioned (cond(A) ≈ 5.83). A Monte Carlo sensitivity analysis (±5% demand noise, 1,000 draws) showed the recovered solar dispatch amplifies input noise by roughly the condition-number bound (~6.5x), while battery dispatch is amplified far less (~2x) — so the solar split is the more demand-sensitive of the two recommendations.
- A single vectorised `scipy.linalg.solve` call over 30 days is faster than solving day-by-day in a Python loop.
- 2 of 30 synthetic days produced a physically infeasible (negative) raw solar dispatch; these are corrected with `scipy.optimize.nnls` rather than naive clipping.
- Comparing solar and battery volatility by raw standard deviation vs. coefficient of variation gives opposite answers — a reminder that variance/stdev on series with very different means is not directly comparable.
- The `HybridMicroGrid` extension shows that a badly chosen third constraint (linearly dependent on the first two) makes the 3x3 system singular and the diesel dispatch undetermined.

Full analysis, assumptions and limitations are in the notebook's **Findings & Limitations** section.

## Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model — summary of findings

`src/fishstock.py` implements `FishStock` (discrete logistic growth with harvesting), `PriceModel` (bounded random-walk export price) and `RiskAssessor` (CV-based risk classification + Monte Carlo VaR). Key results from `project3_fishstock.ipynb`:

- Simulated equilibrium stock at every tested harvest rate matches the model's analytic fixed point `N* = K(1 - h/r)` almost exactly — a correctness check against a second (closed-form) method.
- h=0.20 harvests closest to the theoretical maximum sustainable yield (MSY = rK/4) while settling at a stable equilibrium of K/2; h=0.30 overharvests — average weekly catch is *lower* than at h=0.20 because the depleted equilibrium stock leaves less to harvest a fraction of.
- The original brief's flat "variance > 50,000" risk rule is meaningless because variance carries squared, scale-dependent units; replacing it with the dimensionless coefficient of variation (CV) gives a comparable, justified "Low/Moderate/High" classification instead.
- In this model, revenue risk (CV ≈ 0.11–0.12 across all harvest rates) is driven entirely by price randomness, since harvest is deterministic given h — a real fishery would also have stochastic catch, so this understates true risk.
- An 8-week/year closed season cuts 5-year revenue by ~14% but raises the final stock by ~33%, a defensible short-term-income-for-long-term-stock trade-off.

Full analysis, assumptions and limitations are in the notebook's **Findings & Limitations** section.

## Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser — summary of findings

`src/rainfall.py` implements `Region` and `CropRule`, plus a corrected `cosine_similarity` (replacing a prior cohort's flawed use of `math.cos()`, which computes the cosine of a scalar angle, not vector similarity), `similarity_matrices`, and `find_rainy_seasons` (peak detection via `scipy.signal.find_peaks`). This project uses **real 14-year (2010–2023) monthly rainfall from NASA POWER** (`power.larc.nasa.gov`, parameter `PRECTOTCORR`) for Kampala, Gulu and Mbarara from the start, not just in the extension — see `scripts/fetch_rainfall_data.py` for the fetch/regeneration script and `data/rainfall_2010_2023.csv` for the committed data. Key results from `project4_rainfall.ipynb`:

- Cosine similarity and Pearson correlation can disagree sharply on the same pair of regions (Kampala vs Gulu: cosine ≈ 0.89 "very similar" vs Pearson ≈ 0.47 "weakly correlated") — a concrete demonstration of why cosine similarity alone can be misleading.
- Peak detection classified **all three regions as bimodal**, contradicting the textbook "unimodal north" simplification for Gulu — though Gulu's inter-peak dip is much shallower than Kampala's, so it behaves more like one long wet season with two mild humps.
- Year-to-year reliability varies hugely by month: Gulu's January rainfall has an interannual CV of 1.89 (std exceeds the mean) vs 0.38 in its wettest month (May) — dry months are far less predictable across years, a risk invisible in a single illustrative year of data.
- Crop-suitability thresholds (maize, beans, coffee) are stated explicitly with their FAO-derived rationale rather than left unjustified.

Full analysis, assumptions and limitations are in the notebook's **Findings & Limitations** section.

## Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner — summary of findings

`src/taxiroute.py` implements `Route`, a `Forecaster` ABC with `MovingAverageForecaster`, `ExponentialSmoothingForecaster`, `LinearTrendForecaster` and `SeasonalNaiveForecaster` subclasses, plus `solve_equilibrium`, `walk_forward_mae`, `grid_search_alpha`, `vehicles_needed` and `simulate_weekly_demand` helpers. Key results from `project5_taxiroute.ipynb`:

- The Ntinda route's UGX 2,000 fare sits below its modelled UGX 2,200 supply-demand equilibrium — a ~10 passenger/trip-hour shortage consistent with matatus filling up quickly on that route.
- Walk-forward backtesting (days 4–10) picked exponential smoothing as the best model on all three routes, but the alpha grid search hit the tested boundary (α=0.9) every time — flagged explicitly as a sign the true optimum may lie outside the tested range, not just accepted at face value.
- Day-11 forecasts and fleet sizing (8 trips/day × 14 seats + 15% buffer) show all three illustrative routes only need 1 vehicle — a function of the small illustrative passenger volumes, not a general result.
- The extension is the clearest finding: over a 60-day simulation with a real weekly pattern, a 7-day seasonal-naive forecaster (MAE 4.04) clearly beat both the moving average (7.80) and exponential smoothing (7.07) — the 10-day core-task series is too short to ever show this, since it contains less than two weeks of data.

Full analysis, assumptions and limitations are in the notebook's **Findings & Limitations** section.

## AI use declaration

Claude Code (Anthropic, Claude Sonnet 5) was used as an AI coding assistant for all five mini-projects: to help design the class structures (`DistrictPopulation`/`Forecaster` subclasses; `MicroGrid`/`HybridMicroGrid`; `FishStock`/`PriceModel`/`RiskAssessor`; `Region`/`CropRule`; `Route`/`Forecaster` subclasses), write the data generators/fetchers and numerical helper logic (bootstrap prediction intervals; NNLS infeasibility handling; logistic-growth/random-walk/Monte-Carlo simulations; cosine similarity/Pearson/Euclidean pairwise matrices; rainy-season peak detection; walk-forward backtesting and alpha grid search), fetch and convert the real NASA POWER rainfall data, draft the pytest test suites, and help write the notebooks' narrative/markdown cells. All generated code was reviewed, run, and understood before inclusion; numerical results (MAE/RMSE/MAPE, determinant, condition number, CVs, VaR, equilibrium stock/price, costs, similarity matrices, backtest MAE) were produced by actually executing each notebook, not invented — including catching and correcting a wrong assumption in an early draft (project 4's Task 5 discussion and advisory note originally assumed Gulu was unimodal; the executed peak-detection output showed it is bimodal, and the text was rewritten to match) and refactoring project 1's forecasters to expose a public `fitted_values()` method instead of the notebook reaching into private attributes.

## Repository structure

```
oop/
  src/                       # reusable, tested logic per mini-project
    population.py            # Mini-Project 1
    microgrid.py             # Mini-Project 2
    fishstock.py             # Mini-Project 3
    rainfall.py               # Mini-Project 4
    taxiroute.py              # Mini-Project 5
  scripts/
    fetch_rainfall_data.py   # regenerates data/rainfall_2010_2023.csv from NASA POWER
  data/
    demand_30days.csv        # synthetic, seeded demand data for Mini-Project 2
    rainfall_2010_2023.csv   # real NASA POWER monthly rainfall for Mini-Project 4
  tests/
    test_population.py       # pytest suite for Mini-Project 1
    test_microgrid.py        # pytest suite for Mini-Project 2
    test_fishstock.py        # pytest suite for Mini-Project 3
    test_rainfall.py         # pytest suite for Mini-Project 4
    test_taxiroute.py        # pytest suite for Mini-Project 5
  project1_population.ipynb
  project2_microgrid.ipynb
  project3_fishstock.ipynb
  project4_rainfall.ipynb
  project5_taxiroute.ipynb
  requirements.txt
  README.md
```
