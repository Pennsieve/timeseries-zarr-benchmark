from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from timeseries_zarr import Bundle

from generate.tszarr import write_tszarr
from resampling_benchmark.check import (
    arrays_equal,
    check_recording,
    edge_samples,
    gap_ranges,
    main,
    nominal_entries,
    plan_window,
    windows_for,
)
from resampling_benchmark.load.generator import ServerRequester
from resampling_benchmark.load.orchestrate import WINDOWS_S
from resampling_benchmark.load.precomputed import PrecomputedReader
from resampling_benchmark.service.app import Settings, create_app
from resampling_benchmark.service.store import open_recording
from tests.resampling_benchmark.conftest import (
    TINY_ID,
    TINY_SPEC,
    upload_directory,
)
from tszbench.csvmeta import read_header
from tszbench.fs import CountingFileSystem, StoreRoots, make_filesystem
from tszbench.minmax import COLUMNS
from tszbench.recordings import RECORDINGS, Gap, RecordingSpec, sample_range
from tszbench.schemas import CHECK, CHECK_FILE, relative_path

DECIMATION = 4


def expected_level(spec: RecordingSpec, window_s: float) -> int:
    """Return the coarsest level with at least COLUMNS bins in [0, window_s)."""
    for level in range(spec.levels, 0, -1):
        if nominal_entries(window_s, spec.rate_hz, level) >= COLUMNS:
            return level
    return 0


def all_windows() -> list[tuple[str, int]]:
    return [
        (name, window)
        for name, spec in RECORDINGS.items()
        for window in windows_for(spec)
    ]


def test_windows_follow_the_plan() -> None:
    assert windows_for(RECORDINGS["R3"]) == list(WINDOWS_S)
    assert windows_for(RECORDINGS["R4"]) == [10, 60, 600, 3_600]
    assert windows_for(RECORDINGS["R4"], [60, 86_400]) == [60]


def test_the_levels_of_r3_match_the_plan() -> None:
    spec = RECORDINGS["R3"]
    levels = [expected_level(spec, window) for window in WINDOWS_S]
    assert levels == [1, 2, 4, 5, 8]


@pytest.mark.parametrize(("name", "window_s"), all_windows())
def test_check_window_holds_whole_bins_on_a_bin_edge(
    name: str, window_s: int
) -> None:
    spec = RECORDINGS[name]
    level = expected_level(spec, window_s)
    window = plan_window(spec, window_s, level)
    bin_samples = DECIMATION**level
    count = window.stop - window.first
    assert count == COLUMNS * window.bins_per_column * bin_samples
    assert window.first % bin_samples == 0
    assert window.first % edge_samples(spec.rate_hz, bin_samples) == 0
    assert window.first >= 0
    assert window.stop <= spec.samples_per_channel
    assert sample_range(window.start_us, window.end_us, spec.rate_hz) == (
        window.first,
        window.stop,
    )


@pytest.mark.parametrize(("name", "window_s"), all_windows())
def test_check_window_holds_the_gap_it_is_centered_on(
    name: str, window_s: int
) -> None:
    spec = RECORDINGS[name]
    window = plan_window(spec, window_s, expected_level(spec, window_s))
    count = window.stop - window.first
    gaps = gap_ranges(spec)
    shorter = [gap for gap in gaps if gap[1] - gap[0] < count]
    chosen = (
        max(shorter, key=lambda gap: gap[1] - gap[0])
        if shorter
        else min(gaps, key=lambda gap: gap[1] - gap[0])
    )
    if shorter:
        assert window.first <= chosen[0]
        assert chosen[1] <= window.stop
    else:
        assert chosen[0] <= window.first < window.stop <= chosen[1]


def test_bins_per_column_rounds_the_nominal_bins() -> None:
    spec = RECORDINGS["R3"]
    window = plan_window(spec, 600, 4)
    assert window.bins_per_column == 2
    assert window.stop - window.first == 2000 * 2 * 256
    window = plan_window(spec, 86_400, 8)
    assert window.bins_per_column == 1


def test_a_raw_check_window_holds_columns_times_rounded_samples() -> None:
    spec = RECORDINGS["R1"]
    window = plan_window(spec, 10, 0)
    assert window.level == 0
    assert window.stop - window.first == COLUMNS * round(2560 / COLUMNS)


def test_a_recording_too_short_for_the_rounded_bins_gets_fewer_per_column() -> (
    None
):
    window = plan_window(RECORDINGS["R1"], 86_400, 6)
    assert window.bins_per_column == 2


def test_a_recording_with_fewer_bins_than_columns_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        plan_window(RECORDINGS["R1"].cut(3_600, "1h"), 86_400, 7)


def test_arrays_equal_treats_nan_as_equal_and_checks_shape() -> None:
    left = np.array([[[np.nan, 1.0]]], dtype=np.float32)
    assert arrays_equal(left, left.copy())
    assert not arrays_equal(left, np.array([[[0.0, 1.0]]], dtype=np.float32))
    assert not arrays_equal(left, np.zeros((1, 2, 2), dtype=np.float32))


def test_check_recording_reports_equal_and_unequal_rows(
    tiny_spec: RecordingSpec,
) -> None:
    ok = np.zeros((4, COLUMNS, 2), dtype=np.float32)
    rows = check_recording(
        tiny_spec,
        [5],
        lambda start, end: 1,
        lambda start, end: ok,
        lambda start, end, level: ok if level == 1 else ok + 1,
    )
    assert len(rows) == 1
    assert rows[0]["equal"] == "true"
    assert rows[0]["rows"] == COLUMNS
    assert set(rows[0]) == set(CHECK)
    wrong = check_recording(
        tiny_spec,
        [5],
        lambda start, end: 1,
        lambda start, end: ok,
        lambda start, end, level: ok + 1,
    )
    assert wrong[0]["equal"] == "false"


def test_a_window_that_raises_gets_an_error_row_and_the_others_still_run(
    tiny_spec: RecordingSpec,
) -> None:
    ok = np.zeros((4, COLUMNS, 2), dtype=np.float32)

    def server(start_us: int, end_us: int) -> np.ndarray:
        if end_us - start_us > 30_000_000:
            raise httpx.ReadTimeout("slow")
        return ok

    rows = check_recording(
        tiny_spec,
        [5, 60, 10],
        lambda start, end: 1,
        server,
        lambda start, end, level: ok,
    )
    assert [row["window_s"] for row in rows] == ["5", "60", "10"]
    assert [row["equal"] for row in rows] == [
        "true",
        "error:ReadTimeout",
        "true",
    ]
    assert rows[1]["start_us"] != ""
    assert set(rows[1]) == set(CHECK)


@pytest.fixture(scope="module")
def reader_parts(
    tiny_roots: StoreRoots,
) -> Iterator[tuple[Bundle, CountingFileSystem, ThreadPoolExecutor]]:
    filesystem = make_filesystem("s3", tiny_roots)
    bundle = open_recording(filesystem, tiny_roots, TINY_ID)
    with ThreadPoolExecutor(max_workers=4) as pool:
        yield bundle, filesystem, pool
    bundle.close()
    filesystem.close()


@pytest.fixture(scope="module")
def reader(
    reader_parts: tuple[Bundle, CountingFileSystem, ThreadPoolExecutor],
) -> PrecomputedReader:
    return PrecomputedReader(*reader_parts)


@pytest.fixture(scope="module")
def requester(tiny_settings: Settings) -> Iterator[ServerRequester]:
    with TestClient(create_app(tiny_settings)) as client:
        yield ServerRequester(client, TINY_ID)


@pytest.mark.network
def test_the_two_methods_agree_on_every_check_window(
    tiny_spec: RecordingSpec,
    reader: PrecomputedReader,
    requester: ServerRequester,
) -> None:
    windows = [1, 5, 10, 30, 60]
    rows = check_recording(
        tiny_spec,
        windows,
        reader.choose_level,
        requester.fetch,
        lambda start, end, level: reader.read(start, end, level).array,
    )
    assert [row["equal"] for row in rows] == ["true"] * len(windows)
    assert {row["level"] for row in rows} == {0, 1, 2}
    assert all(row["rows"] == COLUMNS for row in rows)


@pytest.mark.network
def test_check_windows_include_the_nan_runs_of_the_gap_channel(
    tiny_spec: RecordingSpec,
    reader: PrecomputedReader,
    requester: ServerRequester,
) -> None:
    for window_s in (10, 60):
        level = reader.choose_level(0, window_s * 1_000_000)
        window = plan_window(tiny_spec, window_s, level)
        served = requester.fetch(window.start_us, window.end_us)
        read = reader.read(window.start_us, window.end_us, window.level).array
        assert np.isnan(served[0]).any()
        assert not np.isnan(served[1]).any()
        np.testing.assert_array_equal(served, read)


@pytest.mark.network
def test_the_two_methods_bound_the_same_range_on_an_arbitrary_window(
    reader: PrecomputedReader, requester: ServerRequester
) -> None:
    start_us, end_us = 31_337_001, 41_337_003
    served = requester.fetch(start_us, end_us)
    result = reader.read(start_us, end_us)
    assert result.level >= 1
    assert result.array.shape == served.shape
    assert result.counts.requests > 0
    assert np.isfinite(result.array[1]).all()
    assert np.isfinite(served[1]).all()
    assert result.array[1, :, 0].min() <= served[1, :, 0].min()
    assert result.array[1, :, 1].max() >= served[1, :, 1].max()


@pytest.mark.network
def test_the_precomputed_read_picks_the_coarsest_level_with_enough_bins(
    reader: PrecomputedReader,
) -> None:
    for window_s in (5, 10, 30, 60):
        level = reader.choose_level(0, window_s * 1_000_000)
        assert level == expected_level(TINY_SPEC, window_s)


@pytest.mark.network
def test_a_window_too_short_for_any_level_is_read_raw(
    reader: PrecomputedReader,
) -> None:
    result = reader.read(0, 500_000)
    assert result.level == 0
    assert result.array.shape == (4, 1000, 2)
    assert result.counts.bytes > 0
    assert result.read_seconds > 0
    assert result.reduce_seconds > 0
    assert np.isfinite(result.array[1]).all()


@pytest.mark.network
def test_the_precomputed_read_of_a_level_fetches_fewer_bytes_than_raw(
    reader: PrecomputedReader,
) -> None:
    raw = reader.read(0, 30_000_000, 0)
    binned = reader.read(0, 30_000_000, 3)
    assert binned.counts.bytes < raw.counts.bytes
    assert binned.level == 3


@pytest.mark.network
def test_the_command_line_check_passes_and_writes_rows(
    tiny_roots: StoreRoots,
    server_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(RECORDINGS, TINY_ID, TINY_SPEC)
    monkeypatch.setattr(
        "resampling_benchmark.check.WINDOWS_S", (1, 5, 10, 30, 60)
    )
    out = tmp_path / CHECK_FILE
    status = main(
        [
            "--server-url",
            server_url,
            "--s3-root",
            tiny_roots.s3,
            "--s3-endpoint-url",
            str(tiny_roots.s3_endpoint_url),
            "--recordings",
            TINY_ID,
            "--results",
            str(tmp_path),
        ]
    )
    assert status == 0
    lines = [
        line
        for line in out.read_text().splitlines()
        if not line.startswith("#")
    ]
    assert lines[0].split(",") == list(CHECK)
    assert len(lines) == 6
    assert all(line.endswith(",true") for line in lines[1:])
    assert "date" in read_header(out)


ODD_SPEC = replace(
    TINY_SPEC,
    id="T2",
    channels=2,
    rate_hz=512,
    duration_s=150,
    levels=3,
    gaps=(Gap(channel=0, start_s=60.0, duration_s=1.5),),
)
"""A rate whose sample period is not a whole number of microseconds."""


@pytest.mark.network
def test_the_methods_agree_when_the_sample_period_is_not_whole_microseconds(
    tmp_path: Path, s3_endpoint: str
) -> None:
    path = tmp_path / relative_path(ODD_SPEC.id, "tszarr")
    path.parent.mkdir(parents=True)
    write_tszarr(ODD_SPEC, path)
    roots = upload_directory(tmp_path, s3_endpoint)
    settings = Settings(roots=roots, recordings=(ODD_SPEC.id,), processes=2)
    filesystem = make_filesystem("s3", roots)
    bundle = open_recording(filesystem, roots, ODD_SPEC.id)
    with (
        TestClient(create_app(settings)) as client,
        ThreadPoolExecutor(max_workers=4) as pool,
    ):
        reader = PrecomputedReader(bundle, filesystem, pool)
        server = ServerRequester(client, ODD_SPEC.id)
        rows = check_recording(
            ODD_SPEC,
            [1, 5, 10, 60],
            reader.choose_level,
            server.fetch,
            lambda start, end, level: reader.read(start, end, level).array,
        )
    bundle.close()
    filesystem.close()
    assert [row["equal"] for row in rows] == ["true"] * 4
    assert {row["level"] for row in rows} >= {0, 1}
