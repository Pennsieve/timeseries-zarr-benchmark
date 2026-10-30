from dataclasses import replace
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest
import zarr
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO

from generate.nwb import (
    BUFFER_BYTES,
    CHANNEL_CHUNK_ROWS,
    CONVERSION,
    SERIES_NAME,
    TIME_CHUNK_BYTES,
    ZSTD_LEVEL,
    ChunkLayout,
    SynthesizedIterator,
    buffer_rows,
    chunk_shape,
    write_nwb_hdf5,
    write_nwb_zarr,
)
from tszbench.recordings import RECORDINGS, START_TIME, Gap, RecordingSpec
from tszbench.signal import Synthesizer

ZSTD_FILTER_ID = 32015
CHANNELS = 3
RATE_HZ = 2048
DURATION_S = 60
SPEC = replace(
    RECORDINGS["R3"],
    id="tiny",
    channels=CHANNELS,
    rate_hz=RATE_HZ,
    duration_s=DURATION_S,
    gaps=(
        Gap(channel=0, start_s=5.0, duration_s=1.0),
        Gap(channel=0, start_s=30.25, duration_s=0.5),
    ),
)
TIME_ROWS = TIME_CHUNK_BYTES // (4 * CHANNELS)
DATA_PATH = f"acquisition/{SERIES_NAME}/data"


@pytest.fixture(scope="module")
def expected() -> np.ndarray:
    return np.concatenate(list(Synthesizer(SPEC).blocks(TIME_ROWS)))


@pytest.fixture(scope="module", params=["time", "channel"])
def layout(request: pytest.FixtureRequest) -> ChunkLayout:
    return cast(ChunkLayout, request.param)


@pytest.fixture(scope="module")
def hdf5_path(
    tmp_path_factory: pytest.TempPathFactory, layout: ChunkLayout
) -> Path:
    path = tmp_path_factory.mktemp("hdf5") / "file.nwb"
    write_nwb_hdf5(SPEC, path, layout, workers=2)
    return path


@pytest.fixture(scope="module")
def zarr_path(
    tmp_path_factory: pytest.TempPathFactory, layout: ChunkLayout
) -> Path:
    path = tmp_path_factory.mktemp("zarr") / "file.nwb.zarr"
    write_nwb_zarr(SPEC, path, layout, workers=2)
    return path


def expected_chunks(layout: ChunkLayout) -> tuple[int, int]:
    return (TIME_ROWS, CHANNELS) if layout == "time" else (8192, 1)


def test_a_time_chunk_holds_every_channel_in_about_one_mebibyte():
    for name in ("R1", "R2", "R3", "R4"):
        spec = RECORDINGS[name]
        rows, columns = chunk_shape(spec, "time")
        assert columns == spec.channels
        assert rows == max(1, 2**20 // (4 * spec.channels))
        assert 4 * rows * columns <= 2**20


def test_a_channel_chunk_holds_8192_rows_of_one_channel():
    assert chunk_shape(RECORDINGS["R3"], "channel") == (8192, 1)
    assert CHANNEL_CHUNK_ROWS == 8192


def test_a_recording_shorter_than_a_chunk_gets_a_chunk_of_its_length():
    spec = replace(SPEC, duration_s=1)
    assert chunk_shape(spec, "time") == (RATE_HZ, CHANNELS)
    assert chunk_shape(spec, "channel") == (RATE_HZ, 1)
    assert buffer_rows(spec, "time") == RATE_HZ


@pytest.mark.parametrize("name", ["R1", "R2", "R3", "R4"])
@pytest.mark.parametrize("layout", ["time", "channel"])
def test_a_buffer_holds_whole_chunk_rows_within_the_buffer_size(
    name: str, layout: ChunkLayout
):
    spec = RECORDINGS[name]
    chunk_rows = chunk_shape(spec, layout)[0]
    rows = buffer_rows(spec, layout)
    assert rows % chunk_rows == 0
    assert rows >= chunk_rows
    assert (
        4 * rows * spec.channels <= BUFFER_BYTES
        or rows == chunk_rows
        or rows == spec.samples_per_channel
    )


def test_the_iterator_writes_every_chunk_whole_and_in_sample_order(
    expected: np.ndarray, layout: ChunkLayout
):
    iterator = SynthesizedIterator(SPEC, layout)
    chunk_rows = expected_chunks(layout)[0]
    rows_seen = 0
    for chunk in iterator:
        selection = chunk.selection
        assert selection[0].start == rows_seen
        assert selection[0].start % chunk_rows == 0
        assert selection[1] == slice(0, CHANNELS)
        if selection[0].stop < SPEC.samples_per_channel:
            assert (selection[0].stop - selection[0].start) % chunk_rows == 0
        np.testing.assert_array_equal(chunk.data, expected[selection])
        rows_seen = selection[0].stop
    assert rows_seen == SPEC.samples_per_channel


def test_the_iterator_reports_the_shape_and_type_of_the_series():
    iterator = SynthesizedIterator(SPEC, "time")
    assert iterator.maxshape == (SPEC.samples_per_channel, CHANNELS)
    assert iterator.dtype == np.float32


def test_the_hdf5_series_holds_the_synthesized_samples(
    hdf5_path: Path, expected: np.ndarray
):
    with h5py.File(hdf5_path) as file:
        data = file[DATA_PATH][()]
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data, expected)


def test_the_hdf5_series_is_chunked_and_compressed_with_zstd(
    hdf5_path: Path, layout: ChunkLayout
):
    with h5py.File(hdf5_path) as file:
        dataset = file[DATA_PATH]
        assert dataset.shape == (SPEC.samples_per_channel, CHANNELS)
        assert dataset.chunks == expected_chunks(layout)
        properties = dataset.id.get_create_plist()
        assert properties.get_nfilters() == 1
        identifier, _, values, _ = properties.get_filter(0)
        assert identifier == ZSTD_FILTER_ID
        assert values[0] == ZSTD_LEVEL


def test_the_hdf5_series_stores_microvolts_at_the_spec_rate(hdf5_path: Path):
    with NWBHDF5IO(str(hdf5_path), mode="r") as io:
        nwbfile = io.read()
        series = nwbfile.acquisition[SERIES_NAME]
        assert nwbfile.identifier == SPEC.id
        assert nwbfile.session_start_time == START_TIME
        assert series.unit == "volts"
        assert series.conversion == CONVERSION
        assert series.rate == RATE_HZ
        assert series.starting_time == 0.0
        assert series.data.shape == (SPEC.samples_per_channel, CHANNELS)
        assert len(nwbfile.electrodes) == CHANNELS
        assert len(nwbfile.devices) == 1
        assert len(nwbfile.electrode_groups) == 1
        assert list(series.electrodes.data[:]) == list(range(CHANNELS))


def test_the_zarr_series_holds_the_synthesized_samples(
    zarr_path: Path, expected: np.ndarray
):
    array = zarr.open_group(str(zarr_path), mode="r")[DATA_PATH]
    assert array.dtype == np.float32
    np.testing.assert_array_equal(array[:], expected)


def test_the_zarr_series_is_chunked_and_compressed_with_zstd(
    zarr_path: Path, layout: ChunkLayout
):
    array = zarr.open_group(str(zarr_path), mode="r")[DATA_PATH]
    assert array.metadata.zarr_format == 3
    assert array.shape == (SPEC.samples_per_channel, CHANNELS)
    assert array.chunks == expected_chunks(layout)
    assert array.shards is None
    assert len(array.compressors) == 1
    codec = array.compressors[0]
    assert type(codec).__name__ == "ZstdCodec"
    assert codec.level == ZSTD_LEVEL


def test_the_zarr_file_writes_one_object_per_chunk(
    zarr_path: Path, layout: ChunkLayout
):
    chunks = expected_chunks(layout)
    expected_objects = -(-SPEC.samples_per_channel // chunks[0]) * (
        CHANNELS // chunks[1]
    )
    data_directory = zarr_path / DATA_PATH / "c"
    found = [path for path in data_directory.rglob("*") if path.is_file()]
    assert len(found) == expected_objects


def test_the_zarr_root_holds_consolidated_metadata(zarr_path: Path):
    root = zarr.open_group(str(zarr_path), mode="r")
    assert root.metadata.consolidated_metadata is not None


def test_the_zarr_series_stores_microvolts_at_the_spec_rate(zarr_path: Path):
    with NWBZarrIO(str(zarr_path), mode="r") as io:
        nwbfile = io.read()
        series = nwbfile.acquisition[SERIES_NAME]
        assert nwbfile.identifier == SPEC.id
        assert nwbfile.session_start_time == START_TIME
        assert series.unit == "volts"
        assert series.conversion == CONVERSION
        assert series.rate == RATE_HZ
        assert series.starting_time == 0.0
        assert len(nwbfile.electrodes) == CHANNELS
        assert len(nwbfile.devices) == 1


def test_a_recording_of_one_channel_writes_to_both_backends(
    tmp_path: Path,
):
    spec: RecordingSpec = replace(SPEC, channels=1, duration_s=2, gaps=())
    expected = np.concatenate(list(Synthesizer(spec).blocks(4096)))
    write_nwb_hdf5(spec, tmp_path / "a.nwb", "time")
    write_nwb_zarr(spec, tmp_path / "a.nwb.zarr", "channel")
    with h5py.File(tmp_path / "a.nwb") as file:
        np.testing.assert_array_equal(file[DATA_PATH][()], expected)
    array = zarr.open_group(str(tmp_path / "a.nwb.zarr"), mode="r")[DATA_PATH]
    np.testing.assert_array_equal(array[:], expected)
