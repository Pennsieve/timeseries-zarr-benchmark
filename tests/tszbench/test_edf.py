import tracemalloc
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from fsspec.implementations.local import LocalFileSystem

from tszbench import edf
from tszbench.edf import (
    EdfRangeReader,
    parse_header,
    plan_requests,
    write_edf,
)
from tszbench.recordings import GRID_UV, RECORDINGS, RecordingSpec, sample_range

DIGITAL_GAP = -32768

LAYOUTS = {
    "r1": replace(
        RECORDINGS["R1"], id="t1", channels=1, duration_s=20, gaps=()
    ),
    "r2": replace(
        RECORDINGS["R2"], id="t2", channels=4, duration_s=12, gaps=()
    ),
    "r3": replace(RECORDINGS["R3"], id="t3", channels=5, duration_s=8, gaps=()),
}

WINDOWS = {
    "inside one record": (200_000, 700_000),
    "across records": (700_000, 3_300_000),
    "file start": (0, 1_000_000),
    "file end": (-1_500_000, None),
    "past the end": (-1_000_000, 50_000_000),
    "single sample": (3_906, 3_907),
}


class RecordingFileSystem(LocalFileSystem):  # type: ignore[misc]
    """Local filesystem that records every byte-range read."""

    def __init__(self) -> None:
        super().__init__(skip_instance_cache=True)
        self.reads: list[tuple[int, int]] = []

    def cat_file(self, path, start=None, end=None, **kwargs):
        self.reads.append((start, end))
        return super().cat_file(path, start, end, **kwargs)


def digital_data(spec: RecordingSpec, seed: int = 0) -> np.ndarray:
    """Return random (samples, channels) int16 data with runs of gap markers."""
    rng = np.random.default_rng(seed)
    data = rng.integers(
        -32767,
        32768,
        size=(spec.samples_per_channel, spec.channels),
        dtype=np.int16,
    )
    rows = spec.samples_per_channel
    data[rows // 3 : rows // 3 + spec.rate_hz // 2, 0] = DIGITAL_GAP
    data[rows - 7 :, -1] = DIGITAL_GAP
    return data


def expected_physical(data: np.ndarray) -> np.ndarray:
    physical = (data.astype(np.float64) * GRID_UV).astype(np.float32)
    physical[data == DIGITAL_GAP] = np.nan
    return physical


def write_layout(
    tmp_path: Path, spec: RecordingSpec, block_rows: int | None = None
) -> tuple[Path, np.ndarray]:
    data = digital_data(spec)
    path = tmp_path / f"{spec.id}.edf"
    step = block_rows or data.shape[0]
    write_edf(
        path,
        spec,
        (data[row : row + step] for row in range(0, data.shape[0], step)),
    )
    return path, data


def resolve_window(spec: RecordingSpec, window: tuple[int, int | None]):
    start_us, end_us = window
    if start_us < 0:
        start_us += spec.duration_us
    if end_us is None:
        end_us = spec.duration_us
    elif end_us < 0:
        end_us += spec.duration_us
    return start_us, end_us


def reader_for(path: Path) -> EdfRangeReader:
    return EdfRangeReader(RecordingFileSystem(), str(path))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_header_round_trips_the_spec(tmp_path, layout):
    spec = LAYOUTS[layout]
    path, _ = write_layout(tmp_path, spec)
    raw = path.read_bytes()
    header = parse_header(lambda start, end: raw[start:end])
    per_record = round(spec.edf_record_s * spec.rate_hz)
    assert header.header_bytes == 256 * (spec.channels + 1)
    assert header.records == spec.samples_per_channel // per_record
    assert header.record_s == spec.edf_record_s
    assert header.start == datetime(2026, 1, 1)
    assert header.labels == tuple(f"ch{i:03d}" for i in range(spec.channels))
    assert header.samples_per_record == (per_record,) * spec.channels
    assert header.physical_min == (-4096.0,) * spec.channels
    assert header.physical_max == (4095.875,) * spec.channels
    assert header.digital_min == (-32768,) * spec.channels
    assert header.digital_max == (32767,) * spec.channels
    assert header.record_bytes == 2 * spec.channels * per_record
    assert (
        len(raw) == header.header_bytes + header.records * header.record_bytes
    )


def test_header_is_plain_ascii_edf_text(tmp_path):
    spec = LAYOUTS["r3"]
    path, _ = write_layout(tmp_path, spec)
    raw = path.read_bytes()
    assert raw[:8] == b"0       "
    assert raw[168:184] == b"01.01.2600.00.00"
    assert raw[236:252] == b"125     0.064   "
    assert raw[256 : 256 + 16] == b"ch000           "
    assert raw[: 256 * (spec.channels + 1)].isascii()
    physical_max_at = 256 + spec.channels * (16 + 80 + 8 + 8)
    assert raw[physical_max_at : physical_max_at + 8] == b"4095.875"


def test_header_reads_are_two_calls():
    calls = []

    def read(start, end):
        calls.append((start, end))
        return b""

    with pytest.raises(ValueError, match="must hold at least 256"):
        parse_header(read)
    assert calls == [(0, 256)]


def test_reader_fetches_the_header_in_two_reads(tmp_path):
    path, _ = write_layout(tmp_path, LAYOUTS["r2"])
    filesystem = RecordingFileSystem()
    EdfRangeReader(filesystem, str(path))
    assert filesystem.reads == [(0, 256), (256, 256 * 5)]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("window", WINDOWS)
def test_all_channel_window_equals_digital_times_grid(tmp_path, layout, window):
    spec = LAYOUTS[layout]
    path, data = write_layout(tmp_path, spec)
    start_us, end_us = resolve_window(spec, WINDOWS[window])
    first, stop = sample_range(start_us, end_us, spec.rate_hz)
    stop = min(stop, spec.samples_per_channel)
    result = reader_for(path).read(start_us, end_us)
    assert result.dtype == np.float32
    assert result.shape == (stop - first, spec.channels)
    np.testing.assert_array_equal(result, expected_physical(data[first:stop]))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("window", WINDOWS)
def test_one_channel_window_equals_digital_times_grid(tmp_path, layout, window):
    spec = LAYOUTS[layout]
    path, data = write_layout(tmp_path, spec)
    start_us, end_us = resolve_window(spec, WINDOWS[window])
    first, stop = sample_range(start_us, end_us, spec.rate_hz)
    stop = min(stop, spec.samples_per_channel)
    reader = reader_for(path)
    for channel in {0, spec.channels - 1}:
        result = reader.read(start_us, end_us, channel)
        assert result.dtype == np.float32
        assert result.shape == (stop - first,)
        np.testing.assert_array_equal(
            result, expected_physical(data[first:stop, channel])
        )


def test_gap_markers_read_as_nan_and_nothing_else_does(tmp_path):
    spec = LAYOUTS["r2"]
    path, data = write_layout(tmp_path, spec)
    result = reader_for(path).read(0, spec.duration_us)
    assert np.isnan(result).sum() == (data == DIGITAL_GAP).sum() > 0
    assert (result[~np.isnan(result)] >= -4095.875).all()


def test_window_beyond_the_file_is_empty(tmp_path):
    spec = LAYOUTS["r1"]
    path, _ = write_layout(tmp_path, spec)
    reader = reader_for(path)
    after = spec.duration_us
    assert reader.read(after, after + 5_000_000).shape == (0, 1)
    assert reader.read(after, after + 5_000_000, 0).shape == (0,)
    assert reader.read(5_000_000, 5_000_000, 0).shape == (0,)


def test_rate_is_samples_per_record_over_record_duration(tmp_path):
    path, _ = write_layout(tmp_path, LAYOUTS["r3"])
    assert reader_for(path).rate_hz == 2000


def test_rejects_a_channel_outside_the_file(tmp_path):
    path, _ = write_layout(tmp_path, LAYOUTS["r2"])
    with pytest.raises(ValueError, match=r"channel 4: must be in \[0, 4\)"):
        reader_for(path).read(0, 1_000_000, 4)


@pytest.mark.parametrize("block_rows", [1, 37, 256, 1_000])
def test_blocks_of_any_length_write_the_same_file(tmp_path, block_rows):
    spec = LAYOUTS["r3"]
    whole, _ = write_layout(tmp_path, spec)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    cut, _ = write_layout(other_dir, spec, block_rows=block_rows)
    assert cut.read_bytes() == whole.read_bytes()


def test_writer_batches_records_below_the_batch_size(tmp_path, monkeypatch):
    spec = LAYOUTS["r2"]
    whole, data = write_layout(tmp_path, spec)
    monkeypatch.setattr(edf, "WRITE_BATCH_BYTES", 3 * 4 * 2 * 512 + 5)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    batched, _ = write_layout(other_dir, spec, block_rows=700)
    assert batched.read_bytes() == whole.read_bytes()


def test_writer_stores_each_channel_contiguously_in_a_record(tmp_path):
    spec = LAYOUTS["r2"]
    path, data = write_layout(tmp_path, spec)
    raw = path.read_bytes()[256 * (spec.channels + 1) :]
    record = np.frombuffer(raw[: 2 * 4 * 512], dtype="<i2").reshape(4, 512)
    np.testing.assert_array_equal(record, data[:512].T)


def test_writer_rejects_blocks_that_do_not_total_the_samples(tmp_path):
    spec = LAYOUTS["r1"]
    path = tmp_path / "short.edf"
    short = np.zeros((spec.samples_per_channel - 1, 1), dtype=np.int16)
    with pytest.raises(ValueError, match="must total"):
        write_edf(path, spec, [short])
    assert not path.exists()
    long = np.zeros((spec.samples_per_channel + 1, 1), dtype=np.int16)
    with pytest.raises(ValueError, match="must total"):
        write_edf(path, spec, [long])
    assert not path.exists()


@pytest.mark.parametrize(
    "block",
    [
        np.zeros((256, 1), dtype=np.int32),
        np.zeros((256,), dtype=np.int16),
        np.zeros((256, 2), dtype=np.int16),
    ],
)
def test_writer_rejects_a_block_of_the_wrong_shape_or_type(tmp_path, block):
    with pytest.raises(ValueError, match="must"):
        write_edf(tmp_path / "bad.edf", LAYOUTS["r1"], [block])


def test_writer_rejects_a_spec_without_a_record_duration(tmp_path):
    spec = replace(RECORDINGS["R4"], duration_s=1, gaps=())
    with pytest.raises(ValueError, match="no EDF record duration"):
        write_edf(tmp_path / "r4.edf", spec, [])


def test_writer_rejects_a_duration_that_is_not_whole_records(tmp_path):
    spec = replace(LAYOUTS["r3"], duration_s=1)
    with pytest.raises(ValueError, match="whole number"):
        write_edf(tmp_path / "bad.edf", spec, [])


def test_writer_rejects_a_record_that_is_not_whole_samples(tmp_path):
    spec = replace(LAYOUTS["r1"], edf_record_s=0.001)
    with pytest.raises(ValueError, match="whole number of samples"):
        write_edf(tmp_path / "bad.edf", spec, [])


def test_parser_rejects_signals_of_different_rates(tmp_path):
    spec = LAYOUTS["r2"]
    path, _ = write_layout(tmp_path, spec)
    raw = bytearray(path.read_bytes())
    samples_at = 256 + spec.channels * (16 + 80 + 8 * 5 + 80)
    raw[samples_at : samples_at + 8] = b"256     "
    with pytest.raises(ValueError, match="one positive sample rate"):
        parse_header(lambda start, end: bytes(raw[start:end]))


def test_parser_names_the_field_that_does_not_parse(tmp_path):
    spec = LAYOUTS["r1"]
    path, _ = write_layout(tmp_path, spec)
    raw = bytearray(path.read_bytes())
    raw[236:244] = b"many    "
    with pytest.raises(ValueError, match="field records"):
        parse_header(lambda start, end: bytes(raw[start:end]))


KIB = 1 << 10
MIB = 1 << 20


@pytest.mark.parametrize(
    ("ranges", "expected"),
    [
        ([], []),
        ([(0, 100)], [(0, 100)]),
        ([(0, 100), (100, 200)], [(0, 200)]),
        ([(0, 100), (100 + MIB - 1, 200 + MIB)], [(0, 200 + MIB)]),
        (
            [(0, 100), (100 + MIB, 200 + MIB)],
            [(0, 100), (100 + MIB, 200 + MIB)],
        ),
        ([(0, 100), (50, 80), (60, 300)], [(0, 300)]),
        ([(500, 600), (0, 100)], [(0, 600)]),
        (
            [(2 * MIB, 2 * MIB + 5), (0, 100)],
            [(0, 100), (2 * MIB, 2 * MIB + 5)],
        ),
        ([(0, 100), (0, 0), (900, 900)], [(0, 100)]),
    ],
)
def test_plan_requests_merges_ranges_closer_than_the_gap(ranges, expected):
    assert plan_requests(ranges) == expected


def test_plan_requests_splits_a_range_above_the_maximum():
    limit = edf.MAX_REQUEST_BYTES
    assert plan_requests([(10, 10 + limit)]) == [(10, 10 + limit)]
    assert plan_requests([(10, 10 + 2 * limit + 5)]) == [
        (10, 10 + limit),
        (10 + limit, 10 + 2 * limit),
        (10 + 2 * limit, 10 + 2 * limit + 5),
    ]


def test_plan_requests_splits_after_merging():
    limit = edf.MAX_REQUEST_BYTES
    ranges = [(0, limit), (limit + 10, limit + 20)]
    assert plan_requests(ranges) == [(0, limit), (limit, limit + 20)]


def test_plan_requests_covers_every_range_in_at_most_the_maximum():
    rng = np.random.default_rng(3)
    starts = np.sort(rng.integers(0, 200 * MIB, size=300))
    ranges = [(int(s), int(s) + int(rng.integers(1, 3 * MIB))) for s in starts]
    requests = plan_requests(ranges)
    assert all(end - start <= edf.MAX_REQUEST_BYTES for start, end in requests)
    for start, end in ranges:
        assert any(lo <= start and end <= hi for lo, hi in requests) or (
            sum(
                min(end, hi) - max(start, lo)
                for lo, hi in requests
                if lo < end and start < hi
            )
            == end - start
        )


def test_all_channel_window_is_one_request_after_the_header(tmp_path):
    spec = LAYOUTS["r3"]
    path, _ = write_layout(tmp_path, spec)
    filesystem = RecordingFileSystem()
    reader = EdfRangeReader(filesystem, str(path))
    filesystem.reads.clear()
    reader.read(700_000, 3_300_000)
    record_bytes = reader.header.record_bytes
    assert filesystem.reads == [
        (
            reader.header.header_bytes + 10 * record_bytes,
            reader.header.header_bytes + 52 * record_bytes,
        )
    ]


def test_one_channel_of_a_narrow_file_merges_into_one_request(tmp_path):
    path, _ = write_layout(tmp_path, LAYOUTS["r2"])
    filesystem = RecordingFileSystem()
    reader = EdfRangeReader(filesystem, str(path))
    filesystem.reads.clear()
    reader.read(0, 5_000_000, 1)
    assert len(filesystem.reads) == 1


def test_one_channel_of_a_wide_file_reads_one_range_per_record(tmp_path):
    spec = replace(
        RECORDINGS["R3"],
        id="wide",
        channels=300,
        rate_hz=2000,
        duration_s=4,
        edf_record_s=1.0,
        gaps=(),
    )
    path, data = write_layout(tmp_path, spec)
    filesystem = RecordingFileSystem()
    reader = EdfRangeReader(filesystem, str(path))
    filesystem.reads.clear()
    result = reader.read(500_000, 3_500_000, 7)
    assert len(filesystem.reads) == 4
    assert {end - start for start, end in filesystem.reads} == {4_000}
    np.testing.assert_array_equal(
        result, expected_physical(data[1_000:7_000, 7])
    )


def test_requests_are_fetched_by_one_range_call(tmp_path):
    spec = replace(
        RECORDINGS["R3"],
        id="wide",
        channels=300,
        duration_s=4,
        edf_record_s=1.0,
        gaps=(),
    )
    path, _ = write_layout(tmp_path, spec)
    calls = []

    class Spy(RecordingFileSystem):
        def cat_ranges(self, paths, starts, ends, **kwargs):
            calls.append(len(paths))
            return super().cat_ranges(paths, starts, ends, **kwargs)

    reader = EdfRangeReader(Spy(), str(path))
    reader.read(0, 4_000_000, 3)
    assert calls == [4]


@pytest.mark.parametrize("layout", ["r2", "r3"])
@pytest.mark.parametrize("channel", [None, 2])
def test_a_request_split_inside_a_record_still_decodes(
    tmp_path, monkeypatch, layout, channel
):
    spec = LAYOUTS[layout]
    path, data = write_layout(tmp_path, spec)
    monkeypatch.setattr(edf, "MAX_REQUEST_BYTES", 1_001)
    filesystem = RecordingFileSystem()
    reader = EdfRangeReader(filesystem, str(path))
    filesystem.reads.clear()
    first, stop = sample_range(700_000, 3_300_000, spec.rate_hz)
    result = reader.read(700_000, 3_300_000, channel)
    assert len(filesystem.reads) > 1
    assert all(end - start <= 1_001 for start, end in filesystem.reads)
    expected = (
        data[first:stop] if channel is None else data[first:stop, channel]
    )
    np.testing.assert_array_equal(result, expected_physical(expected))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_pyedflib_reads_the_same_physical_values(tmp_path, layout):
    pyedflib = pytest.importorskip("pyedflib")
    spec = LAYOUTS[layout]
    path, data = write_layout(tmp_path, spec)
    expected = expected_physical(data)
    reader = pyedflib.EdfReader(str(path))
    try:
        assert reader.signals_in_file == spec.channels
        assert (
            reader.getNSamples().tolist()
            == [spec.samples_per_channel] * spec.channels
        )
        for channel in range(spec.channels):
            other = reader.readSignal(channel)
            ours = expected[:, channel]
            present = ~np.isnan(ours)
            np.testing.assert_array_equal(other[present], ours[present])
            assert (other[~present] == -4096.0).all()
    finally:
        reader.close()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_float32_decode_equals_the_float64_decode(tmp_path, layout):
    spec = LAYOUTS[layout]
    path, data = write_layout(tmp_path, spec)
    reader = reader_for(path)
    gain = reader._gain
    shift = reader._shift
    wide = data.astype(np.float64) * gain + shift
    wide[data == DIGITAL_GAP] = np.nan
    result = reader.read(0, spec.duration_us)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, wide.astype(np.float32))


def test_all_channel_read_peaks_below_four_times_the_int16_payload(tmp_path):
    spec = replace(
        RECORDINGS["R3"], id="t4", channels=8, duration_s=16, gaps=()
    )
    path, _ = write_layout(tmp_path, spec)
    reader = reader_for(path)
    payload = spec.samples_per_channel * spec.channels * 2
    tracemalloc.start()
    try:
        result = reader.read(0, spec.duration_us)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.shape == (spec.samples_per_channel, spec.channels)
    assert peak < 4 * payload
