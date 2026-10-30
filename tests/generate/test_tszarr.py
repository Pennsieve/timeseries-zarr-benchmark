import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from timeseries_zarr import MultiChannelSource, WriteOpts, open_bundle

from generate.tszarr import (
    BLOCK_ROWS,
    START_US,
    ZSTD_LEVEL,
    SynthesizedSource,
    write_tszarr,
)
from tszbench.recordings import RECORDINGS, START_TIME, Gap, RecordingSpec
from tszbench.signal import Synthesizer

CHANNELS = 4
RATE_HZ = 2000
DURATION_S = 60
BINS_PER_LEVEL_ONE = 4
SPEC = replace(
    RECORDINGS["R3"],
    id="R3-test",
    channels=CHANNELS,
    rate_hz=RATE_HZ,
    duration_s=DURATION_S,
    gaps=(
        Gap(channel=0, start_s=3.0, duration_s=1.0),
        Gap(channel=0, start_s=10.0005, duration_s=0.25),
        Gap(channel=2, start_s=40.0, duration_s=0.5),
    ),
)


def synthesize(spec: RecordingSpec) -> np.ndarray:
    return np.concatenate(list(Synthesizer(spec).blocks(BLOCK_ROWS)))


@pytest.fixture(scope="module")
def expected() -> np.ndarray:
    return synthesize(SPEC)


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tszarr") / "bundle.zarr"
    write_tszarr(SPEC, path)
    return path


def array_codecs(node: object) -> list[dict[str, object]]:
    """Return every codec dictionary nested anywhere in a metadata document."""
    found: list[dict[str, object]] = []
    if isinstance(node, dict):
        if "name" in node and "configuration" in node:
            found.append(node)
        for value in node.values():
            found.extend(array_codecs(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(array_codecs(value))
    return found


def test_the_source_is_a_multi_channel_source():
    assert isinstance(SynthesizedSource(SPEC), MultiChannelSource)


def test_the_source_describes_each_channel():
    channels = SynthesizedSource(SPEC).channels()
    assert [channel.id for channel in channels] == [
        f"ch{index:03d}" for index in range(CHANNELS)
    ]
    assert [channel.name for channel in channels] == [
        channel.id for channel in channels
    ]
    for channel in channels:
        assert channel.unit == "uV"
        assert channel.rate_hz() == RATE_HZ
        assert channel.start_us() == START_US
        assert channel.num_samples() == SPEC.samples_per_channel


def test_the_start_time_is_in_microseconds_since_the_unix_epoch():
    assert round(START_TIME.timestamp() * 1_000_000) == START_US


def test_the_windows_hold_the_synthesized_samples_by_channel(expected):
    windows = list(SynthesizedSource(SPEC).iter_windows())
    columns = [
        np.concatenate([window[channel] for window in windows])
        for channel in range(CHANNELS)
    ]
    np.testing.assert_array_equal(np.stack(columns, axis=1), expected)
    assert all(len(window) == CHANNELS for window in windows)


def test_a_window_of_the_widest_recording_fits_in_32_mebibytes():
    assert BLOCK_ROWS * RECORDINGS["R4"].channels * 4 <= 32 * 2**20


def test_the_default_write_options_match_the_plan():
    opts = WriteOpts()
    assert opts.inner_len == 8192
    assert opts.target_shard_bytes == 16 * 2**20
    assert opts.min_bins == 1024


def test_raw_holds_the_synthesized_samples(bundle_path, expected):
    with open_bundle(bundle_path) as bundle:
        assert len(bundle.continuous) == CHANNELS
        for key, channel in bundle.continuous.items():
            np.testing.assert_array_equal(channel.raw[:], expected[:, key])


def test_the_channels_carry_their_metadata(bundle_path):
    with open_bundle(bundle_path) as bundle:
        for key, channel in bundle.continuous.items():
            assert channel.id == f"ch{key:03d}"
            assert channel.name == f"ch{key:03d}"
            assert channel.unit == "uV"
            assert channel.rate_hz == RATE_HZ
            assert channel.offset_us == 0
            assert channel.num_samples == SPEC.samples_per_channel


def test_time_counts_from_the_start_time(bundle_path):
    with open_bundle(bundle_path) as bundle:
        assert bundle.recording is not None
        assert bundle.recording.reference_time == START_TIME
        assert bundle.recording.session_start_time == START_TIME


def test_every_level_holds_minmax_mean_var_and_count(bundle_path):
    with open_bundle(bundle_path) as bundle:
        for channel in bundle.continuous.values():
            assert len(channel.levels) == 3
            for level in channel.levels.values():
                assert level.minmax is not None
                assert level.mean is not None
                assert level.var is not None
                assert level.count is not None


def test_level_one_matches_a_reduction_of_the_samples(bundle_path, expected):
    with open_bundle(bundle_path) as bundle:
        for key, channel in bundle.continuous.items():
            level = channel.levels[1]
            assert level.samples_per_bin == BINS_PER_LEVEL_ONE
            bins = (
                expected[:, key]
                .astype(np.float64)
                .reshape(-1, BINS_PER_LEVEL_ONE)
            )
            counts = np.isfinite(bins).sum(axis=1)
            np.testing.assert_array_equal(level.count[:], counts)
            full = counts == BINS_PER_LEVEL_ONE
            minmax = level.minmax[:]
            np.testing.assert_allclose(minmax[full, 0], bins[full].min(axis=1))
            np.testing.assert_allclose(minmax[full, 1], bins[full].max(axis=1))
            np.testing.assert_allclose(
                level.mean[:][full],
                bins[full].mean(axis=1),
                rtol=1e-5,
                atol=1e-4,
            )
            np.testing.assert_allclose(
                level.var[:][full], bins[full].var(axis=1), rtol=1e-4, atol=1e-3
            )
            assert np.isnan(minmax[~full]).all()
            assert np.isnan(level.mean[:][~full]).all()
            assert np.isnan(level.var[:][~full]).all()


def test_a_bin_with_a_gap_sample_is_nan_and_keeps_its_count(bundle_path):
    first, _ = SPEC.gaps[1].sample_range(RATE_HZ)
    assert first % BINS_PER_LEVEL_ONE != 0
    index = first // BINS_PER_LEVEL_ONE
    with open_bundle(bundle_path) as bundle:
        level = bundle.continuous[0].levels[1]
        assert level.count[index] == first % BINS_PER_LEVEL_ONE
        assert np.isnan(level.minmax[index]).all()


def test_every_level_counts_the_finite_samples(bundle_path, expected):
    with open_bundle(bundle_path) as bundle:
        for key, channel in bundle.continuous.items():
            finite = int(np.isfinite(expected[:, key]).sum())
            for level in channel.levels.values():
                assert int(level.count[:].sum()) == finite


def test_the_root_holds_consolidated_metadata(bundle_path):
    root = json.loads((bundle_path / "zarr.json").read_text())
    consolidated = root["consolidated_metadata"]
    assert consolidated["kind"] == "inline"
    channels = {key for key in consolidated["metadata"] if "/" not in key}
    assert channels == {str(index) for index in range(CHANNELS)}
    assert "0/raw" in consolidated["metadata"]


def test_every_array_uses_zstd_level_3_and_raw_uses_8192_element_chunks(
    bundle_path,
):
    arrays = 0
    for metadata_path in bundle_path.rglob("zarr.json"):
        metadata = json.loads(metadata_path.read_text())
        if metadata["node_type"] != "array":
            continue
        arrays += 1
        codecs = array_codecs(metadata["codecs"])
        zstd = [codec for codec in codecs if codec["name"] == "zstd"]
        assert [codec["configuration"]["level"] for codec in zstd] == [
            ZSTD_LEVEL
        ]
        if metadata_path.parent.name == "raw":
            sharding = [c for c in codecs if c["name"] == "sharding_indexed"]
            assert sharding[0]["configuration"]["chunk_shape"] == [8192]
    assert arrays == CHANNELS * (1 + 3 * 4)


def test_the_worker_count_does_not_change_the_bundle(tmp_path, bundle_path):
    path = tmp_path / "parallel.zarr"
    write_tszarr(SPEC, path, workers=3)
    with open_bundle(path) as parallel, open_bundle(bundle_path) as serial:
        for key, channel in parallel.continuous.items():
            np.testing.assert_array_equal(
                channel.raw[:], serial.continuous[key].raw[:]
            )
