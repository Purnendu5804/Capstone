"""Tests for scorers (S1, S2), baselines, registry, and timing harness."""

import warnings

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

    def test_strict_mode_raises_on_missing_band(self) -> None:
        pixels = np.zeros((16, 16, 4), dtype=np.uint16)
        tile = create_tile(pixels, band_names=("B02", "B03", "B04", "B08"))
        scorer = S1VarianceScorer(band="B99", strict=True)
        with pytest.raises(ValueError, match="Band 'B99' not found in tile.band_names"):
            scorer.score(tile)

    def test_non_strict_warns_once_and_falls_back(self) -> None:
        pixels = np.zeros((16, 16, 4), dtype=np.uint16)
        pixels[:, :, 0] = 500  # channel 0 has variance 0
        tile = create_tile(pixels, band_names=("B02", "B03", "B04", "B08"))

        scorer = S1VarianceScorer(band="B99", strict=False)

        # First call emits warning
        with pytest.warns(UserWarning, match="Band 'B99' not found"):
            score = scorer.score(tile)
        assert score == pytest.approx(0.0, abs=1e-9)

        # Second call should NOT emit another warning
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            score2 = scorer.score(tile)
            assert score2 == pytest.approx(0.0, abs=1e-9)


class TestS2EdgeScorer:
    """Tests for S2EdgeScorer."""

    def test_protocol_conformance(self) -> None:
        scorer = S2EdgeScorer()
        assert isinstance(scorer, Scorer)
        assert scorer.name == "s2"

    def test_constant_tile_is_zero_in_both_modes(self) -> None:
        pixels = np.full((32, 32, 4), 4000, dtype=np.uint16)
        tile = create_tile(pixels)

        density_scorer = S2EdgeScorer(mode="density", threshold=0.02)
        mag_scorer = S2EdgeScorer(mode="mean_magnitude")

        assert density_scorer.score(tile) == pytest.approx(0.0, abs=1e-9)
        assert mag_scorer.score(tile) == pytest.approx(0.0, abs=1e-9)

    def test_checkerboard_has_higher_density_than_single_edge(self) -> None:
        single_edge = np.zeros((32, 32, 4), dtype=np.uint16)
        single_edge[:, 16:, :] = 8000
        single_edge_tile = create_tile(single_edge)

        # 4x4 checkerboard with many edges
        grid_y, grid_x = np.ogrid[:32, :32]
        checker_2d = np.where(((grid_y // 4) + (grid_x // 4)) % 2 == 1, 8000, 0).astype(np.uint16)
        checker_pixels = np.repeat(checker_2d[:, :, None], 4, axis=2)
        checker_tile = create_tile(checker_pixels)

        scorer = S2EdgeScorer(mode="density", threshold=0.02)
        single_density = scorer.score(single_edge_tile)
        checker_density = scorer.score(checker_tile)

        assert 0.0 <= single_density <= 1.0
        assert 0.0 <= checker_density <= 1.0
        assert checker_density > single_density

    def test_density_bounded_in_zero_one(self) -> None:
        rng = np.random.default_rng(99)
        pixels = rng.integers(0, 65535, size=(32, 32, 4), dtype=np.uint16)
        tile = create_tile(pixels)

        scorer = S2EdgeScorer(mode="density", threshold=0.02)
        score = scorer.score(tile)
        assert 0.0 <= score <= 1.0

    def test_threshold_monotonicity(self) -> None:
        # A higher threshold must never give a higher edge density
        pixels = np.zeros((32, 32, 4), dtype=np.uint16)
        pixels[:16, :, :] = 8000
        tile = create_tile(pixels)

        thresholds = [0.01, 0.02, 0.05, 0.10, 0.20]
        densities = [
            S2EdgeScorer(mode="density", threshold=t).score(tile) for t in thresholds
        ]

        for i in range(len(densities) - 1):
            assert densities[i] >= densities[i + 1]

    def test_strict_mode_and_fallback(self) -> None:
        pixels = np.zeros((16, 16, 4), dtype=np.uint16)
        tile = create_tile(pixels, band_names=("B02", "B03", "B04", "B08"))

        scorer_strict = S2EdgeScorer(band="B99", strict=True)
        with pytest.raises(ValueError, match="Band 'B99' not found"):
            scorer_strict.score(tile)

        scorer_lenient = S2EdgeScorer(band="B99", strict=False)
        with pytest.warns(UserWarning, match="Band 'B99' not found"):
            score = scorer_lenient.score(tile)
        assert score == pytest.approx(0.0, abs=1e-9)

        # Subsequent call does not warn again
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            assert scorer_lenient.score(tile) == pytest.approx(0.0, abs=1e-9)

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            S2EdgeScorer(scale=0.0)
        with pytest.raises(ValueError, match="non-negative"):
            S2EdgeScorer(threshold=-0.1)
        with pytest.raises(ValueError, match="mode must be"):
            S2EdgeScorer(mode="invalid_mode")


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

    def test_random_scorer_pure_function_of_seed_and_tile_id(self) -> None:
        pixels = np.zeros((16, 16, 4), dtype=np.uint16)
        tile_a = create_tile(pixels, tile_id="tile_A")
        tile_b = create_tile(pixels, tile_id="tile_B")
        tile_c = create_tile(pixels, tile_id="tile_C")

        scorer = RandomScorer(seed=42)

        # Call in one order
        score_a1 = scorer.score(tile_a)
        score_b1 = scorer.score(tile_b)
        score_c1 = scorer.score(tile_c)

        # Call in a completely different order
        score_c2 = scorer.score(tile_c)
        score_a2 = scorer.score(tile_a)
        score_b2 = scorer.score(tile_b)

        # Same tile must produce exact same score regardless of call order
        assert score_a1 == score_a2
        assert score_b1 == score_b2
        assert score_c1 == score_c2

        # Check values are in [0, 1)
        for s in [score_a1, score_b1, score_c1]:
            assert 0.0 <= s < 1.0

        # Different seed produces different score
        scorer_other = RandomScorer(seed=999)
        assert scorer_other.score(tile_a) != score_a1

    def test_random_scorer_has_no_reset(self) -> None:
        scorer = RandomScorer()
        assert not hasattr(scorer, "reset")

    def test_fifo_scorer_earlier_timestamp_higher_score(self) -> None:
        pixels = np.zeros((16, 16, 4), dtype=np.uint16)
        early_tile = create_tile(pixels, tile_id="early", timestamp=10.0)
        late_tile = create_tile(pixels, tile_id="late", timestamp=25.0)

        scorer = FifoScorer()
        early_score = scorer.score(early_tile)
        late_score = scorer.score(late_tile)

        assert early_score > late_score
        assert early_score == -10.0
        assert late_score == -25.0


class TestRegistry:
    """Tests for scorer registry."""

    def test_list_scorers_returns_only_canonical_names(self) -> None:
        canonical = list_scorers()
        expected = ["fifo", "random", "s1", "s2", "s2_mean"]
        assert canonical == expected

    def test_build_builtin_scorers(self) -> None:
        s1 = build_scorer("s1")
        assert isinstance(s1, S1VarianceScorer)

        s2 = build_scorer("s2")
        assert isinstance(s2, S2EdgeScorer)
        assert s2.mode == "density"

        s2_mean = build_scorer("s2_mean")
        assert isinstance(s2_mean, S2EdgeScorer)
        assert s2_mean.mode == "mean_magnitude"

        rnd = build_scorer("random", seed=10)
        assert isinstance(rnd, RandomScorer)

        fifo = build_scorer("fifo")
        assert isinstance(fifo, FifoScorer)

    def test_aliases_resolve_correctly(self) -> None:
        s1_alias = build_scorer("s1_variance")
        assert isinstance(s1_alias, S1VarianceScorer)

        s2_alias = build_scorer("sobel")
        assert isinstance(s2_alias, S2EdgeScorer)

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

    def test_benchmark_returns_expected_keys_and_shape(self) -> None:
        pixels = np.full((16, 16, 4), 1000, dtype=np.uint16)
        tiles = [create_tile(pixels, tile_id=f"t_{i}") for i in range(3)]
        scorer = S1VarianceScorer()

        res = benchmark_scorer(scorer, tiles, warmup=2, repeats=5)

        # Expected keys
        for key in ["median", "p95", "mean", "samples_s", "peak_memory_bytes"]:
            assert key in res

        # Verify old keys are removed
        for old_key in ["samples", "per_tile_samples", "peak_memory"]:
            assert old_key not in res

        # Verify samples_s shape is (repeats, num_tiles)
        samples_s = res["samples_s"]
        assert isinstance(samples_s, np.ndarray)
        assert samples_s.shape == (5, 3)

        assert res["mean"] > 0.0
        assert res["median"] > 0.0
        assert res["peak_memory_bytes"] >= 0

    def test_benchmark_empty_tiles_raises(self) -> None:
        scorer = S1VarianceScorer()
        with pytest.raises(ValueError, match="Cannot benchmark on empty tiles"):
            benchmark_scorer(scorer, [])
