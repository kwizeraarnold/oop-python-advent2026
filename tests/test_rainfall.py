"""Pytest unit tests for Mini-Project 4 (src/rainfall.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rainfall import (
    CropRule,
    Region,
    cosine_similarity,
    find_rainy_seasons,
    similarity_matrices,
    suitability_table,
)

KAMPALA = [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130]
GULU = [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15]


def test_region_annual_total_and_mean() -> None:
    region = Region("Kampala", KAMPALA)
    assert region.annual_total() == pytest.approx(sum(KAMPALA))
    assert region.mean() == pytest.approx(sum(KAMPALA) / 12)
    assert len(region) == 12


def test_region_wettest_and_driest_month() -> None:
    region = Region("Kampala", KAMPALA)
    assert region.wettest_month() == ("May", 220.0)
    assert region.driest_month() == ("Sep", 60.0)


def test_region_rejects_wrong_length() -> None:
    """Edge case: a non-12-length rainfall series must raise."""
    with pytest.raises(ValueError):
        Region("Bad", [1, 2, 3])


def test_region_rejects_negative_rainfall() -> None:
    """Edge case: negative rainfall must raise."""
    bad = KAMPALA.copy()
    bad[0] = -5
    with pytest.raises(ValueError):
        Region("Bad", bad)


def test_croprule_classify_bands() -> None:
    maize = CropRule("maize", drought_max_mm=60, waterlog_min_mm=180, source="test")
    assert maize.classify(30) == "Drought risk for maize"
    assert maize.classify(120) == "Good for maize"
    assert maize.classify(200) == "Waterlogging risk for maize"


def test_croprule_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        CropRule("maize", drought_max_mm=100, waterlog_min_mm=50, source="test")


def test_croprule_rejects_negative_rainfall() -> None:
    """Edge case: classifying a negative rainfall value must raise."""
    maize = CropRule("maize", drought_max_mm=60, waterlog_min_mm=180, source="test")
    with pytest.raises(ValueError):
        maize.classify(-1)


def test_suitability_table_shape_and_content() -> None:
    regions = [Region("Kampala", KAMPALA), Region("Gulu", GULU)]
    crops = [CropRule("maize", 60, 180, "test"), CropRule("beans", 50, 150, "test")]
    table = suitability_table(regions, crops)
    assert len(table) == 2 * 12 * 2
    assert set(table["region"]) == {"Kampala", "Gulu"}
    assert set(table["crop"]) == {"maize", "beans"}


def test_cosine_similarity_matches_scipy() -> None:
    a, b = np.array(KAMPALA, dtype=float), np.array(GULU, dtype=float)
    ours = cosine_similarity(a, b)
    theirs = 1 - scipy_cosine(a, b)
    assert ours == pytest.approx(theirs, rel=1e-9)


def test_cosine_similarity_identical_vectors_is_one() -> None:
    a = np.array(KAMPALA, dtype=float)
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_scaled_vector_is_still_one() -> None:
    """A region that is exactly a scaled-up copy of another should be rated
    maximally 'similar' by cosine similarity, even though totals differ hugely."""
    a = np.array(KAMPALA, dtype=float)
    b = a * 3.0
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_similarity_rejects_zero_vector() -> None:
    """Edge case: a zero vector makes cosine similarity undefined."""
    with pytest.raises(ValueError):
        cosine_similarity(np.zeros(12), np.array(KAMPALA, dtype=float))


def test_similarity_matrices_diagonals() -> None:
    regions = {"Kampala": np.array(KAMPALA, dtype=float), "Gulu": np.array(GULU, dtype=float)}
    matrices = similarity_matrices(regions)
    assert np.allclose(np.diag(matrices["cosine"].values), 1.0)
    assert np.allclose(np.diag(matrices["pearson"].values), 1.0)
    assert np.allclose(np.diag(matrices["euclidean"].values), 0.0)


def test_find_rainy_seasons_detects_bimodal() -> None:
    # Two clear separated peaks (Mar/Apr and Sep/Oct), low elsewhere.
    bimodal = [10, 10, 100, 90, 10, 10, 10, 10, 100, 90, 10, 10]
    result = find_rainy_seasons(bimodal, prominence=30)
    assert result["pattern"] == "Bimodal"
    assert len(result["peak_months"]) == 2


def test_find_rainy_seasons_detects_unimodal() -> None:
    unimodal = [10, 10, 20, 60, 100, 90, 60, 20, 10, 10, 10, 10]
    result = find_rainy_seasons(unimodal, prominence=30)
    assert result["pattern"] == "Unimodal"
    assert result["peak_months"] == [5]


def test_find_rainy_seasons_rejects_wrong_length() -> None:
    """Edge case: a non-12-length series must raise."""
    with pytest.raises(ValueError):
        find_rainy_seasons([1, 2, 3])
