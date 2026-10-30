from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest
import zarr
from fsspec.implementations.local import LocalFileSystem
from timeseries_zarr import open_bundle

from generate.formats import formats_of, write_format, write_one_byte_object
from generate.nwb import SERIES_NAME, TIME_CHUNK_BYTES, ZSTD_LEVEL
from tszbench import schemas
from tszbench.edf import EdfRangeReader
from tszbench.recordings import RECORDINGS, Gap
from tszbench.signal import Synthesizer

ZSTD_FILTER_ID = 32015
CHANNELS = 3
RATE_HZ = 512
DURATION_S = 40
SPEC = replace(
    RECORDINGS["R3"],
    id="tiny",
    channels=CHANNELS,
    rate_hz=RATE_HZ,
    duration_s=DURATION_S,
    edf_record_s=1.0,
    gaps=(
        Gap(channel=0, start_s=5.0, duration_s=1.0),
        Gap(channel=0, start_s=20.25, duration_s=0.5),
    ),
)
DATA_PATH = f"acquisition/{SERIES_NAME}/data"
# One chunk of the time layout holds more rows than the recording, so a second
# recording with a higher rate covers a partial last chunk.
LONG_SPEC = replace(SPEC, id="long", rate_hz=2048, duration_s=60)

Reader = Callable[[Path], npt.NDArray[np.float32]]


def read_edf(path: Path) -> npt.NDArray[np.float32]:
    reader = EdfRangeReader(LocalFileSystem(), str(path))
    return reader.read(0, SPEC.duration_us)


def read_hdf5(path: Path) -> npt.NDArray[np.float32]:
    with h5py.File(path) as file:
        data: npt.NDArray[np.float32] = file[DATA_PATH][()]
    return data


def read_zarr(path: Path) -> npt.NDArray[np.float32]:
    array = zarr.open_group(str(path), mode="r")[DATA_PATH]
    data: npt.NDArray[np.float32] = array[:]
    return data


def read_tszarr(path: Path) -> npt.NDArray[np.float32]:
    with open_bundle(path) as bundle:
        columns = [
            bundle.continuous[key].raw[:] for key in sorted(bundle.continuous)
        ]
    return np.stack(columns, axis=1)


READERS: dict[str, Reader] = {
    "edf": read_edf,
    "nwb-hdf5-time": read_hdf5,
    "nwb-hdf5-channel": read_hdf5,
    "nwb-zarr-time": read_zarr,
    "nwb-zarr-channel": read_zarr,
    "tszarr": read_tszarr,
}


@pytest.fixture(scope="module")
def expected() -> npt.NDArray[np.float32]:
    return np.concatenate(list(Synthesizer(SPEC).blocks(4096)))


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("formats")
    return {
        name: write_format(SPEC, name, root, workers=2)
        for name in schemas.FORMATS
    }


def test_every_format_has_a_reader():
    assert set(READERS) == set(schemas.FORMATS)


def test_the_path_is_the_relative_path_under_the_root(tmp_path: Path):
    for name in ("edf", "tszarr"):
        path = write_format(SPEC, name, tmp_path)
        assert path == tmp_path / schemas.relative_path(SPEC.id, name)
        assert path.exists()


def test_the_synthesized_samples_hold_the_planted_gaps(
    written: dict[str, Path], expected: npt.NDArray[np.float32]
):
    assert np.isnan(expected[:, 0]).sum() == 1.5 * RATE_HZ
    assert not np.isnan(expected[:, 1:]).any()


@pytest.mark.parametrize("name", schemas.FORMATS)
def test_every_format_returns_the_same_float32_samples(
    name: str,
    written: dict[str, Path],
    expected: npt.NDArray[np.float32],
):
    samples = READERS[name](written[name])
    assert samples.dtype == np.float32
    assert samples.shape == expected.shape
    np.testing.assert_array_equal(samples, expected)


def test_the_edf_file_is_not_compressed(written: dict[str, Path]):
    header = 256 * (CHANNELS + 1)
    size = written["edf"].stat().st_size
    assert size == header + 2 * CHANNELS * SPEC.samples_per_channel


def test_the_nwb_hdf5_files_have_the_specified_chunks_and_codec(
    written: dict[str, Path],
):
    time_rows = TIME_CHUNK_BYTES // (4 * CHANNELS)
    expected_chunks = {
        "nwb-hdf5-time": (min(time_rows, SPEC.samples_per_channel), CHANNELS),
        "nwb-hdf5-channel": (8192, 1),
    }
    for name, chunks in expected_chunks.items():
        with h5py.File(written[name]) as file:
            dataset = file[DATA_PATH]
            assert dataset.chunks == chunks
            identifier, _, values, _ = dataset.id.get_create_plist().get_filter(
                0
            )
            assert identifier == ZSTD_FILTER_ID
            assert values[0] == ZSTD_LEVEL


def test_the_nwb_zarr_files_have_the_specified_chunks_and_codec(
    written: dict[str, Path],
):
    time_rows = TIME_CHUNK_BYTES // (4 * CHANNELS)
    expected_chunks = {
        "nwb-zarr-time": (min(time_rows, SPEC.samples_per_channel), CHANNELS),
        "nwb-zarr-channel": (8192, 1),
    }
    for name, chunks in expected_chunks.items():
        array = zarr.open_group(str(written[name]), mode="r")[DATA_PATH]
        assert array.chunks == chunks
        assert [type(codec).__name__ for codec in array.compressors] == [
            "ZstdCodec"
        ]
        assert array.compressors[0].level == ZSTD_LEVEL


def test_the_tszarr_bundle_holds_every_level_of_every_channel(
    written: dict[str, Path],
):
    with open_bundle(written["tszarr"]) as bundle:
        assert len(bundle.continuous) == CHANNELS
        for channel in bundle.continuous.values():
            assert channel.rate_hz == RATE_HZ
            assert channel.num_samples == SPEC.samples_per_channel


def test_a_longer_recording_gives_the_same_samples_in_every_format(
    tmp_path: Path,
):
    expected = np.concatenate(list(Synthesizer(LONG_SPEC).blocks(8192)))
    for name in ("nwb-hdf5-time", "nwb-zarr-time", "nwb-zarr-channel"):
        path = write_format(LONG_SPEC, name, tmp_path)
        np.testing.assert_array_equal(READERS[name](path), expected)


def test_a_second_write_replaces_the_file(
    tmp_path: Path, expected: npt.NDArray[np.float32]
):
    for name in ("nwb-zarr-channel", "tszarr", "edf", "nwb-hdf5-time"):
        write_format(SPEC, name, tmp_path)
        path = write_format(SPEC, name, tmp_path)
        np.testing.assert_array_equal(READERS[name](path), expected)


def test_edf_is_refused_for_a_recording_without_a_record_duration(
    tmp_path: Path,
):
    spec = replace(SPEC, edf_record_s=None)
    with pytest.raises(ValueError, match="edf"):
        write_format(spec, "edf", tmp_path)
    assert not list(tmp_path.rglob("*"))


def test_an_unknown_format_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="nwb-parquet"):
        write_format(SPEC, "nwb-parquet", tmp_path)


def test_a_failed_write_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fail(*args: object, **kwargs: object) -> None:
        (tmp_path / SPEC.id / schemas.FILE_NAMES["tszarr"]).mkdir(parents=True)
        raise RuntimeError("disk full")

    monkeypatch.setattr("generate.formats.write_tszarr", fail)
    with pytest.raises(RuntimeError, match="disk full"):
        write_format(SPEC, "tszarr", tmp_path)
    assert not (tmp_path / SPEC.id / schemas.FILE_NAMES["tszarr"]).exists()


def test_the_formats_of_a_recording_leave_out_edf_without_a_record_duration():
    assert formats_of(RECORDINGS["R3"]) == schemas.FORMATS
    assert formats_of(RECORDINGS["R4"]) == schemas.FORMATS[1:]


def test_the_one_byte_object_holds_one_byte(tmp_path: Path):
    path = write_one_byte_object(tmp_path)
    assert path == tmp_path / schemas.ONE_BYTE_OBJECT
    assert path.stat().st_size == 1
    assert write_one_byte_object(tmp_path) == path
    assert path.stat().st_size == 1
