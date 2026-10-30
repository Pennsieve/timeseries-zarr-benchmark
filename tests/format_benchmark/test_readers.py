import gc
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import numpy.typing as npt
import pytest

import generate.nwb
from format_benchmark.positions import positions
from format_benchmark.readers import (
    READERS,
    SERIES_NAME,
    TszarrReader,
    make_reader,
    relative_path,
)
from tests.format_benchmark.conftest import TINY_SPEC
from tszbench import schemas
from tszbench.fs import StoreRoots, make_filesystem
from tszbench.recordings import sample_range

NWB_DATA_PATH = f"acquisition/{SERIES_NAME}/data"
WINDOWS_US = [
    (0, 1_000_000, 0),
    (4_500_000, 6_500_000, 0),
    (12_345_678, 13_345_678, 2),
    (20_000_000, 21_000_000, 0),
    (39_000_000, 40_000_000, 1),
]
"""Start, end, and channel of windows that cover a gap, an end of the recording,
and a start between two samples."""


def test_series_name_matches_the_files_that_generate_writes() -> None:
    assert SERIES_NAME == generate.nwb.SERIES_NAME


def test_every_row_of_the_plan_has_a_reader() -> None:
    assert set(READERS) == set(schemas.FORMAT_ROWS)


def test_make_reader_rejects_an_unknown_format(tiny_root: Path) -> None:
    roots = StoreRoots(local=tiny_root, http="", s3="")
    filesystem = make_filesystem("local", roots)
    try:
        with pytest.raises(ValueError, match="must be one of"):
            make_reader("parquet", filesystem, "local", roots, TINY_SPEC)
    finally:
        filesystem.close()


def test_zarr_direct_reads_the_tszarr_file() -> None:
    assert relative_path("zarr-direct", "R1") == schemas.relative_path(
        "R1", "tszarr"
    )


def test_overhead_reads_the_one_byte_object() -> None:
    assert relative_path("overhead", "R1") == schemas.ONE_BYTE_OBJECT


def reference(tiny_root: Path) -> npt.NDArray[np.float32]:
    path = tiny_root / schemas.relative_path("tiny", "nwb-hdf5-time")
    with h5py.File(path) as file:
        data: npt.NDArray[np.float32] = file[NWB_DATA_PATH][()]
    return data


def expected_window(
    full: npt.NDArray[np.float32],
    start_us: int,
    end_us: int,
    channel: int | None,
) -> npt.NDArray[np.float32]:
    first, stop = sample_range(start_us, end_us, TINY_SPEC.rate_hz)
    return full[first:stop] if channel is None else full[first:stop, channel]


def test_every_reader_returns_the_same_samples(
    store: str, roots: StoreRoots, tiny_root: Path
) -> None:
    full = reference(tiny_root)
    filesystem = make_filesystem(store, roots)
    try:
        for file_format in schemas.FORMAT_ROWS:
            if file_format == "overhead":
                continue
            reader = make_reader(
                file_format, filesystem, store, roots, TINY_SPEC
            )
            reader.open()
            try:
                for start_us, end_us, channel in WINDOWS_US:
                    for selected in (channel, None):
                        expected = expected_window(
                            full, start_us, end_us, selected
                        )
                        result = reader.read(start_us, end_us, selected)
                        assert result.dtype == np.float32, file_format
                        assert result.shape == expected.shape, file_format
                        np.testing.assert_array_equal(
                            result,
                            expected,
                            err_msg=f"{file_format} {start_us} {selected}",
                        )
            finally:
                reader.close()
    finally:
        filesystem.close()


def test_the_planned_windows_read_the_same_samples_in_every_format(
    store: str, roots: StoreRoots, tiny_root: Path
) -> None:
    full = reference(tiny_root)
    reads = positions(TINY_SPEC, 10, 2)
    filesystem = make_filesystem(store, roots)
    try:
        for file_format in ("edf", "nwb-zarr-channel", "zarr-direct", "tszarr"):
            reader = make_reader(
                file_format, filesystem, store, roots, TINY_SPEC
            )
            reader.open()
            try:
                for read in reads:
                    end_us = read.start_us + 10_000_000
                    result = reader.read(read.start_us, end_us, read.channel)
                    np.testing.assert_array_equal(
                        result,
                        expected_window(
                            full, read.start_us, end_us, read.channel
                        ),
                    )
            finally:
                reader.close()
    finally:
        filesystem.close()


def test_the_tszarr_reader_returns_the_samples_of_the_window(
    tiny_root: Path,
) -> None:
    roots = StoreRoots(local=tiny_root, http="", s3="")
    filesystem = make_filesystem("local", roots)
    reader = TszarrReader(
        filesystem,
        str(tiny_root / schemas.relative_path("tiny", "tszarr")),
        TINY_SPEC,
    )
    reader.open()
    try:
        aligned = reader.read(0, 1_000_000, 0)
        shifted = reader.read(1000, 1_001_000, 0)
        stacked = reader.read(1000, 1_001_000, None)
        assert len(aligned) == 512
        assert len(shifted) == 512
        assert stacked.shape == (512, TINY_SPEC.channels)
        np.testing.assert_array_equal(stacked[:, 0], shifted)
    finally:
        reader.close()
        filesystem.close()


def test_the_overhead_reader_returns_no_samples_and_counts_one_request(
    store: str, roots: StoreRoots
) -> None:
    filesystem = make_filesystem(store, roots)
    reader = make_reader("overhead", filesystem, store, roots, TINY_SPEC)
    try:
        reader.open()
        before = filesystem.counts()
        result = reader.read(0, 1_000_000, 0)
        used = filesystem.counts() - before
        assert result.size == 0
        assert used.requests == 1
        assert used.bytes == 1
    finally:
        reader.close()
        filesystem.close()


def test_a_one_channel_read_fetches_fewer_bytes_than_every_channel(
    roots: StoreRoots, store: str
) -> None:
    filesystem = make_filesystem(store, roots)
    reader = make_reader(
        "nwb-zarr-channel", filesystem, store, roots, TINY_SPEC
    )
    reader.open()
    try:
        before = filesystem.counts()
        reader.read(0, 10_000_000, 1)
        one = filesystem.counts() - before
        before = filesystem.counts()
        reader.read(0, 10_000_000, None)
        every = filesystem.counts() - before
        assert 0 < one.bytes < every.bytes
        assert 0 < one.requests < every.requests
    finally:
        reader.close()
        filesystem.close()


def test_the_hdf5_reader_fetches_the_chunks_again_on_a_repeated_read(
    store: str, roots: StoreRoots
) -> None:
    filesystem = make_filesystem(store, roots)
    reader = make_reader("nwb-hdf5-time", filesystem, store, roots, TINY_SPEC)
    reader.open()
    try:
        used = []
        for _ in range(2):
            before = filesystem.counts()
            reader.read(0, 1_000_000, 1)
            used.append(filesystem.counts() - before)
        assert all(count.requests > 0 and count.bytes > 0 for count in used)
    finally:
        reader.close()
        filesystem.close()


def test_the_hdf5_reader_holds_garbage_collection_off_while_open(
    tiny_root: Path,
) -> None:
    roots = StoreRoots(local=tiny_root, http="", s3="")
    filesystem = make_filesystem("local", roots)
    reader = make_reader("nwb-hdf5-time", filesystem, "local", roots, TINY_SPEC)
    assert gc.isenabled()
    reader.open()
    try:
        assert not gc.isenabled()
    finally:
        reader.close()
        filesystem.close()
    assert gc.isenabled()
