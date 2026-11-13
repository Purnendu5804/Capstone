"""Tests for scorers (S1, S2), baselines, registry, and timing harness."""

import numpy as np
import pytest

from capstone.contracts.types import Tile
from capstone.scorers.base import Scorer, timed_score
from capstone.scorers.baselines import FifoScorer, RandomScorer
from capstone.scorers.edge import S2EdgeScorer
from capstone.scorers.registry import build_scorer, list_scorers, register_scorer
from capstone.scorers.timing import benchmark_scorer
from capstone.scorers.variance import S1VarianceScorer


def create_tile(pixels: np.ndarray, band_names: tuple[str, ...] = ("B02", "B03", "B04", "B08"), **kwargs) -> Tile:
    """Helper to create a Tile with specified pixel array."""
    data = {
        "tile_id": "test_tile",
        "strip_id": "test_strip",
        "row": 0,
        "col": 0,
        "pixels": pixels,
        "band_names": band_names,
        "timestamp": 1.0,
    }
    data.update(kwargs)
    return Tile(**data)


class TestS1VarianceScorer:
    """Tests for S1VarianceScorer."""

    def test_protocol_conformance(self) -> None:
        scorer = S1VarianceScorer()
        assert isinstance(scorer, Scorer)
        assert scorer.name == "s1"

    def test_constant_tile_variance_is_zero(self) -> None:
        # Constant tile: all values identical across all channels
        pixels = np.full((32, 32, 4), 5000, dtype=np.uint16)
        tile = create_tile(pixels)
        scorer = S1VarianceScorer(band="B08")
        score = scorer.score(tile)
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_noisy_tile_variance_is_larger(self) -> None:
        rng = np.random.default_rng(42)
        const_pixels = np.full((32, 32, 4), 5000, dtype=np.uint16)
        noisy_pixels = np.clip(
            rng.normal(5000.0, 500.0, size=(32, 32, 4)), 0, 65535
        ).astype(np.uint16)

        const_tile = create_tile(const_pixels)
        noisy_tile = create_tile(noisy_pixels)

        scorer = S1VarianceScorer()
        assert scorer.score(const_tile) == pytest.approx(0.0, abs=1e-9)
        assert scorer.score(noisy_tile) > 10000.0

    def test_determinism_for_fixed_tile(self) -> None:
        rng = np.random.default_rng(123)
        pixels = rng.integers(1000, 10000, size=(32, 32, 4), dtype=np.uint16)
        tile = create_tile(pixels)

        scorer = S1VarianceScorer()
        score1 = scorer.score(tile)
        score2 = scorer.score(tile)
        assert score1 == score2

    def test_band_selection_and_fallback(self) -> None:
        pixels = np.zeros((16, 16, 4), dtype=np.uint16)
        # Assign distinct variance to channel 2 ("B04")
        pixels[:, :, 2] = np.tile([1000, 9000], (16, 8)).astype(np.uint16)

        tile = create_tile(pixels, band_names=("B02", "B03", "B04", "B08"))

        scorer_b04 = S1VarianceScorer(band="B04")
        scorer_b02 = S1VarianceScorer(band="B02")
        scorer_missing = S1VarianceScorer(band="B99")  # fallback to channel 0 ("B02")

        assert scorer_b04.score(tile) > 0.0
        assert scorer_b02.score(tile) == pytest.approx(0.0, abs=1e-9)
        assert scorer_missing.score(tile) == scorer_b02.score(tile)


class TestS2EdgeScorer:
    """Tests for S2EdgeScorer."""

    def test_protocol_conformance(self) -> None:
        scorer = S2EdgeScorer()
        assert isinstance(scorer, Scorer)
        assert scorer.name == "s2"

    def test_constant_tile_edge_is_zero(self) -> None:
        pixels = np.full((32, 32, 4), 4000, dtype=np.uint16)
        tile = create_tile(pixels)
        scorer = S2EdgeScorer(band="B08")
        score = scorer.score(tile)
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_sharp_edge_is_larger(self) -> None:
        const_pixels = np.full((32, 32, 4), 4000, dtype=np.uint16)
        edge_pixels = np.zeros((32, 32, 4), dtype=np.uint16)
        edge_pixels[:, :16, :] = 1000
        edge_pixels[:, 16:, :] = 9000

        const_tile = create_tile(const_pixels)
        edge_tile = create_tile(edge_pixels)

        scorer = S2EdgeScorer()
        assert scorer.score(const_tile) == pytest.approx(0.0, abs=1e-9)
        assert scorer.score(edge_tile) > 0.02
        assert scorer.score(edge_tile) > scorer.score(const_tile)

    def test_determinism_for_fixed_tile(self) -> None:
        pixels = np.zeros((32, 32, 4), dtype=np.uint16)
        pixels[:16, :, :] = 8000
        tile = create_tile(pixels)

        scorer = S2EdgeScorer()
        score1 = scorer.score(tile)
        score2 = scorer.score(tile)
        assert score1 == score2

    def test_scale_normalisation(self) -> None:
        pixels = np.zeros((32, 32, 4), dtype=np.uint16)
        pixels[:, :16, :] = 1000
        pixels[:, 16:, :] = 8000
        tile = create_tile(pixels)

        scorer_10k = S2EdgeScorer(scale=10000.0)
        scorer_20k = S2EdgeScorer(scale=20000.0)

        # Doubling the normalization constant should halve the resulting gradient magnitude
        assert scorer_10k.score(tile) == pytest.approx(2.0 * scorer_20k.score(tile), rel=1e-5)

    def test_invalid_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            S2EdgeScorer(scale=0.0)


class TestTimedScore:
    """Tests for timed_score helper."""

    def test_timed_score_measures_cost(self) -> None:
        pixels = np.full((16, 16, 4), 2000, dtype=np.uint16)
        tile = create_tile(pixels, tile_id="tile_perf")
        scorer = S1VarianceScorer(name="s1_test")

        res = timed_score(scorer, tile)
        assert res.tile_id == "tile_perf"
        assert res.scorer_name == "s1_test"
        assert res.score == pytest.approx(0.0, abs=1e-9)
        assert res.cost_seconds >= 0.0


class TestBaselines:
    """Tests for RandomScorer and FifoScorer."""

    def test_random_scorer_respects_seed(self) -> None:
        pixels = np.zeros((16, 16, 4), dtype=np.uint16)
        tile = create_tile(pixels)

        scorer_a1 = RandomScorer(seed=42)
        scorer_a2 = RandomScorer(seed=42)
        scorer_b = RandomScorer(seed=999)

        scores_a1 = [scorer_a1.score(tile) for _ in range(5)]
        scores_a2 = [scorer_a2.score(tile) for _ in range(5)]
        scores_b = [scorer_b.score(tile) for _ in range(5)]

        assert scores_a1 == scores_a2
        assert scores_a1 != scores_b

    def test_fifo_scorer_earlier_timestamp_higher_score(self) -> None:
        pixels = np.zeros((16, 16, 4), dtype=np.uint16)
        early_tile = create_tile(pixels, tile_id="early", timestamp=10.0)
        late_tile = create_tile(pixels, tile_id="late", timestamp=25.0)

        scorer = FifoScorer()
        early_score = scorer.score(early_tile)
        late_score = scorer.score(late_tile)

        # Higher score means more valuable; earlier timestamp must receive higher score
        assert early_score > late_score
        assert early_score == -10.0
        assert late_score == -25.0


class TestRegistry:
    """Tests for scorer registry."""

    def test_build_builtin_scorers(self) -> None:
        s1 = build_scorer("s1")
        assert isinstance(s1, S1VarianceScorer)

        s2 = build_scorer("s2")
        assert isinstance(s2, S2EdgeScorer)

        rnd = build_scorer("random", seed=10)
        assert isinstance(rnd, RandomScorer)

        fifo = build_scorer("fifo")
        assert isinstance(fifo, FifoScorer)

    def test_unknown_scorer_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown scorer"):
            build_scorer("non_existent_scorer")

    def test_register_custom_scorer(self) -> None:
        class DummyScorer:
            name = "dummy"

            def score(self, tile: Tile) -> float:
                return 42.0

        register_scorer("dummy", DummyScorer)
        custom = build_scorer("dummy")
        assert isinstance(custom, DummyScorer)

        pixels = np.zeros((8, 8, 4), dtype=np.uint16)
        assert custom.score(create_tile(pixels)) == 42.0
        assert "dummy" in list_scorers()


class TestTimingHarness:
    """Tests for benchmark_scorer."""

    def test_benchmark_returns_expected_keys(self) -> None:
        pixels = np.full((16, 16, 4), 1000, dtype=np.uint16)
        tiles = [create_tile(pixels, tile_id=f"t_{i}") for i in range(3)]
        scorer = S1VarianceScorer()

        res = benchmark_scorer(scorer, tiles, warmup=2, repeats=5)

        for key in ["median", "p95", "mean", "samples", "per_tile_samples", "peak_memory"]:
            assert key in res

        assert len(res["samples"]) == 3 * 5
        assert res["mean"] > 0.0
        assert res["median"] > 0.0
        assert res["peak_memory"] >= 0

    def test_benchmark_empty_tiles_raises(self) -> None:
        scorer = S1VarianceScorer()
        with pytest.raises(ValueError, match="Cannot benchmark on empty tiles"):
            benchmark_scorer(scorer, [])
