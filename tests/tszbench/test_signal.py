from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import welch

from tszbench.recordings import GRID_UV, RECORDINGS, Gap, RecordingSpec
from tszbench.signal import (
    BUMP_POWER,
    DIGITAL_GAP,
    DIGITAL_LIMIT,
    Synthesizer,
)

RATE_HZ = 512
BLOCK_SAMPLES = 16_384


def make_spec(
    channels: int = 3,
    duration_s: int = 30,
    gaps: tuple[Gap, ...] = (),
    **changes: object,
) -> RecordingSpec:
    return replace(
        RECORDINGS["R2"],
        id="test",
        channels=channels,
        duration_s=duration_s,
        gaps=gaps,
        **changes,
    )


def collect(spec: RecordingSpec, block_samples: int = BLOCK_SAMPLES, workers=1):
    return np.concatenate(
        list(Synthesizer(spec, workers).blocks(block_samples))
    )


@pytest.fixture(scope="module")
def long_run() -> np.ndarray:
    return collect(make_spec(channels=8, duration_s=600))


def test_numpy_draws_are_sequential():
    first = np.random.default_rng(5)
    split = np.concatenate(
        [first.standard_normal(7), first.standard_normal(13)]
    )
    whole = np.random.default_rng(5).standard_normal(20)
    np.testing.assert_array_equal(split, whole)


@pytest.mark.parametrize("block_samples", [1, 777, 4096, 15_360, 20_000])
def test_every_block_length_gives_the_same_samples(block_samples):
    spec = make_spec(
        duration_s=3, gaps=(Gap(channel=1, start_s=1.0, duration_s=0.5),)
    )
    np.testing.assert_array_equal(
        collect(spec, block_samples), collect(spec, 4096)
    )


def test_blocks_hold_the_requested_rows_and_the_remainder_last():
    spec = make_spec(channels=2, duration_s=3)
    shapes = [b.shape for b in Synthesizer(spec).blocks(1000)]
    assert sum(shape[0] for shape in shapes) == spec.samples_per_channel
    assert all(shape == (1000, 2) for shape in shapes[:-1])
    assert shapes[-1] == (spec.samples_per_channel % 1000, 2)


def test_blocks_are_float32_rows_by_channels():
    block = next(iter(Synthesizer(make_spec(channels=2)).blocks(100)))
    assert block.dtype == np.float32
    assert block.shape == (100, 2)


def test_a_cut_gives_the_first_samples_of_the_full_recording():
    full = make_spec(duration_s=30, gaps=(Gap(1, 1.0, 0.5), Gap(2, 20.0, 2.0)))
    cut = full.cut(10, "10s")
    np.testing.assert_array_equal(
        collect(cut), collect(full)[: cut.samples_per_channel]
    )


def test_a_channel_does_not_depend_on_the_channel_count():
    narrow = collect(make_spec(channels=2, duration_s=5))
    wide = collect(make_spec(channels=5, duration_s=5))
    np.testing.assert_array_equal(narrow, wide[:, :2])


def test_two_instances_give_identical_output():
    spec = make_spec(duration_s=5)
    np.testing.assert_array_equal(collect(spec), collect(spec))


def test_one_instance_restarts_from_sample_zero_on_each_call():
    synthesizer = Synthesizer(make_spec(duration_s=2))
    first = np.concatenate(list(synthesizer.blocks(500)))
    second = np.concatenate(list(synthesizer.blocks(500)))
    np.testing.assert_array_equal(first, second)


def test_worker_count_does_not_change_the_output():
    spec = make_spec(channels=6, duration_s=5)
    np.testing.assert_array_equal(
        collect(spec, workers=1), collect(spec, workers=4)
    )


def test_a_different_seed_gives_different_samples():
    spec = make_spec(channels=1, duration_s=2)
    other = replace(spec, seed=spec.seed + 1)
    assert not np.array_equal(collect(spec), collect(other))


def test_rms_matches_the_spec(long_run):
    rms = np.sqrt(np.mean(long_run.astype(np.float64) ** 2))
    assert rms == pytest.approx(RECORDINGS["R2"].rms_uv, rel=0.05)


def test_the_first_seconds_are_stationary():
    spec = make_spec(channels=512, duration_s=1)
    start = collect(spec).astype(np.float64)
    assert np.mean(start**2) == pytest.approx(spec.rms_uv**2, rel=0.08)


def test_the_bump_stands_above_the_neighboring_bands(long_run):
    freqs, power = welch(long_run, fs=RATE_HZ, nperseg=4096, axis=0)
    density = power.mean(axis=1)

    def band(low, high):
        return density[(freqs >= low) & (freqs < high)].mean()

    assert band(9, 11) > 2 * band(4, 6)
    assert band(9, 11) > 4 * band(15, 20)


def test_the_spectrum_falls_as_one_over_frequency(long_run):
    freqs, power = welch(long_run, fs=RATE_HZ, nperseg=4096, axis=0)
    density = power.mean(axis=1)
    fitted = (freqs >= 1) & (freqs <= 100) & ~((freqs >= 6) & (freqs <= 15))
    slope = np.polyfit(np.log(freqs[fitted]), np.log(density[fitted]), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.1)


def test_the_bump_carries_its_share_of_the_variance(long_run):
    freqs, power = welch(long_run, fs=RATE_HZ, nperseg=4096, axis=0)
    density = power.mean(axis=1)
    in_band = density[(freqs >= 7) & (freqs <= 13)].sum()
    assert in_band / density.sum() > BUMP_POWER * 0.8


def test_channels_are_uncorrelated(long_run):
    increments = np.diff(long_run.astype(np.float64), axis=0)
    off_diagonal = ~np.eye(long_run.shape[1], dtype=bool)
    assert np.abs(np.corrcoef(increments.T)[off_diagonal]).max() < 0.05
    assert np.abs(np.corrcoef(long_run.T)[off_diagonal]).max() < 0.4


def test_every_finite_value_is_a_multiple_of_the_grid(long_run):
    scaled = long_run.astype(np.float64) / GRID_UV
    assert np.isfinite(scaled).all()
    np.testing.assert_array_equal(scaled, np.rint(scaled))


def test_values_clip_to_the_digital_limit():
    spec = make_spec(channels=2, duration_s=5, rms_uv=1e5)
    samples = collect(spec)
    limit = DIGITAL_LIMIT * GRID_UV
    assert samples.max() == limit
    assert samples.min() == -limit


GAPS = (
    Gap(channel=1, start_s=2.0, duration_s=0.5),
    Gap(channel=2, start_s=10.0, duration_s=1.0),
    Gap(channel=1, start_s=29.0, duration_s=5.0),
)


@pytest.mark.parametrize("block_samples", [1000, BLOCK_SAMPLES])
def test_gaps_are_nan_on_the_gap_channel_and_nowhere_else(block_samples):
    spec = make_spec(gaps=GAPS)
    expected = np.zeros((spec.samples_per_channel, spec.channels), dtype=bool)
    for gap in GAPS:
        start, stop = gap.sample_range(spec.rate_hz)
        expected[start:stop, gap.channel] = True
    np.testing.assert_array_equal(
        np.isnan(collect(spec, block_samples)), expected
    )


def test_the_signal_under_a_gap_does_not_change_the_other_samples():
    gapped = make_spec(gaps=GAPS)
    plain = make_spec()
    samples = collect(gapped)
    finite = np.isfinite(samples)
    np.testing.assert_array_equal(samples[finite], collect(plain)[finite])


def test_digital_blocks_match_the_float_blocks():
    spec = make_spec(duration_s=5, gaps=GAPS[:2])
    synthesizer = Synthesizer(spec)
    digital = np.concatenate(list(synthesizer.digital_blocks(700)))
    floats = np.concatenate(list(synthesizer.blocks(700)))
    assert digital.dtype == np.int16
    gap = digital == DIGITAL_GAP
    np.testing.assert_array_equal(gap, np.isnan(floats))
    np.testing.assert_array_equal(
        digital[~gap].astype(np.float32) * np.float32(GRID_UV), floats[~gap]
    )
    assert digital[~gap].min() >= -DIGITAL_LIMIT


def test_a_gap_on_a_missing_channel_is_rejected():
    spec = make_spec(
        channels=2, gaps=(Gap(channel=2, start_s=1, duration_s=1),)
    )
    with pytest.raises(ValueError, match="channel 2"):
        Synthesizer(spec)


def test_a_worker_count_below_one_is_rejected():
    with pytest.raises(ValueError, match="workers"):
        Synthesizer(make_spec(), workers=0)


def test_a_block_length_below_one_is_rejected():
    with pytest.raises(ValueError, match="block_samples"):
        next(iter(Synthesizer(make_spec()).blocks(0)))
