"""One reader per row of the format benchmark, behind one interface.

Every reader works through a `CountingFileSystem`. Only the store differs
between two runs of one format.
"""

import abc
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

import h5py
import hdf5plugin  # noqa: F401  (registers the zstd filter of the NWB files)
import numpy as np
import numpy.typing as npt
import zarr
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO
from timeseries_zarr import Bundle, open_bundle

from tszbench import schemas
from tszbench.edf import EdfRangeReader
from tszbench.fs import (
    CountingFileSystem,
    StoreRoots,
    location,
    zarr_store,
)
from tszbench.recordings import RecordingSpec, sample_range

SERIES_NAME = "ElectricalSeries"
"""Name of the series under `acquisition` in the NWB files."""

HDF5_CHUNK_CACHE_BYTES = 0
"""Size of the HDF5 chunk cache of an open file. The default of 1 MiB holds a
whole chunk of R1. A later read of that chunk then sends no request."""

STACK_WORKERS = 32
"""Threads of an all-channel read of TS-Zarr and of `zarr-direct`."""

SampleArray = npt.NDArray[Any]


class Reader(abc.ABC):
    """Reads the raw samples of one window of one file.

    Call `open` once, then `read` any number of times, then `close`.
    """

    file_format: ClassVar[str]

    def __init__(
        self, filesystem: CountingFileSystem, path: str, spec: RecordingSpec
    ) -> None:
        """Bind the reader to a file.

        Args:
            filesystem: The filesystem every request goes through.
            path: Location of the file, as `location` returns it.
            spec: The recording the file holds.
        """
        self._fs = filesystem
        self._path = path
        self._spec = spec

    @abc.abstractmethod
    def open(self) -> None:
        """Fetch the metadata a read needs and keep the file open."""

    @abc.abstractmethod
    def read(
        self, start_us: int, end_us: int, channel: int | None
    ) -> SampleArray:
        """Return the samples of the window `[start_us, end_us)`.

        The window is in microseconds from sample 0. One channel gives shape
        `(samples,)`, and a channel of None reads every channel and gives
        `(samples, channels)`. The samples are float32 microvolts. Every
        reader returns the samples of `tszbench.recordings.sample_range`.
        """

    def close(self) -> None:  # noqa: B027
        """Release what `open` holds. The default holds nothing."""

    def _samples(self, start_us: int, end_us: int) -> tuple[int, int]:
        """Return the half-open sample range of a window."""
        return sample_range(start_us, end_us, self._spec.rate_hz)


def _slice(
    data: Any, first: int, stop: int, channel: int | None
) -> SampleArray:
    """Return one selection of a `(samples, channels)` dataset."""
    if channel is None:
        return np.asarray(data[first:stop, :])
    return np.asarray(data[first:stop, channel])


def _raw_array(group: zarr.Group, key: str) -> "zarr.Array[Any]":
    """Return the `raw` array of the channel group named key."""
    array = group[f"{key}/raw"]
    if not isinstance(array, zarr.Array):
        raise TypeError(f"{key}/raw: is a group, expected an array")
    return array


class EdfReader(Reader):
    """`edf`: the range reader of the package."""

    file_format = "edf"

    def open(self) -> None:
        """Read the EDF header."""
        self._reader = EdfRangeReader(self._fs, self._path)

    def read(
        self, start_us: int, end_us: int, channel: int | None
    ) -> SampleArray:
        """Return the window, as `Reader.read` describes."""
        return self._reader.read(start_us, end_us, channel)


class NwbHdf5Reader(Reader):
    """`nwb-hdf5-time` and `nwb-hdf5-channel`: pynwb over h5py.

    h5py reads through a file object of the filesystem. The file object has no
    read-ahead, and every h5py read is one request. The file keeps no chunk
    cache, and every read fetches the chunks it covers.

    Cyclic garbage collection stays off between `open` and `close`. An h5py
    call holds the HDF5 lock while it waits for the filesystem thread. A
    collection that runs on that thread frees an h5py object and waits for the
    same lock. The process then hangs.
    """

    file_format = "nwb-hdf5-time"

    def open(self) -> None:
        """Open the file, read the HDF5 superblock and the NWB structure."""
        gc.collect()
        self._collect_again = gc.isenabled()
        gc.disable()
        try:
            self._file = h5py.File(
                self._fs.open(self._path, "rb"),
                "r",
                rdcc_nbytes=HDF5_CHUNK_CACHE_BYTES,
            )
            self._io = NWBHDF5IO(file=self._file, mode="r")
            nwb = self._io.read()
            self._data = nwb.acquisition[SERIES_NAME].data
        except BaseException:
            if self._collect_again:
                gc.enable()
            raise

    def read(
        self, start_us: int, end_us: int, channel: int | None
    ) -> SampleArray:
        """Return the window, as `Reader.read` describes."""
        first, stop = self._samples(start_us, end_us)
        return _slice(self._data, first, stop, channel)

    def close(self) -> None:
        """Close the NWB reader and the file, then collect garbage."""
        self._io.close()
        self._file.close()
        gc.collect()
        if self._collect_again:
            gc.enable()


class NwbZarrReader(Reader):
    """`nwb-zarr-time` and `nwb-zarr-channel`: pynwb with hdmf-zarr."""

    file_format = "nwb-zarr-time"

    def open(self) -> None:
        """Open the store and read the NWB structure."""
        self._io = NWBZarrIO(path=zarr_store(self._fs, self._path), mode="r")
        nwb = self._io.read()
        self._data = nwb.acquisition[SERIES_NAME].data

    def read(
        self, start_us: int, end_us: int, channel: int | None
    ) -> SampleArray:
        """Return the window, as `Reader.read` describes."""
        first, stop = self._samples(start_us, end_us)
        return _slice(self._data, first, stop, channel)

    def close(self) -> None:
        """Close the NWB reader."""
        self._io.close()


class TszarrReader(Reader):
    """`tszarr`: the timeseries-zarr reader.

    A one-channel read is `ContinuousChannel.read`. An all-channel read is
    `Bundle.stack` with `STACK_WORKERS` threads, transposed to
    `(samples, channels)`. The reader returns every sample that overlaps the
    window, so the result can start one sample early. `read` drops that sample
    before it returns.
    """

    file_format = "tszarr"

    def open(self) -> None:
        """Open the bundle, which reads `zarr.json` and `recording.json`."""
        self._bundle: Bundle = open_bundle(zarr_store(self._fs, self._path))
        self._keys = sorted(self._bundle.continuous)

    def read(
        self, start_us: int, end_us: int, channel: int | None
    ) -> SampleArray:
        """Return the window, as `Reader.read` describes."""
        first, stop = self._samples(start_us, end_us)
        if channel is None:
            stack = self._bundle.stack(
                self._keys, start_us, end_us, max_workers=STACK_WORKERS
            )
            skip = _skip(stack.first, first)
            return np.asarray(stack.values[:, skip : skip + stop - first].T)
        key = self._keys[channel]
        samples = self._bundle.continuous[key].read(start_us, end_us)
        skip = _skip(samples.first, first)
        return np.asarray(samples.values[skip : skip + stop - first])

    def close(self) -> None:
        """Close the bundle."""
        self._bundle.close()


def _skip(returned_first: int, first: int) -> int:
    """Return the samples to drop from the start of a TS-Zarr result.

    Raises:
        ValueError: The result starts after the first sample of the window.
    """
    if returned_first > first:
        raise ValueError(
            f"read starts at sample {returned_first}: must start at or "
            f"before sample {first}"
        )
    return first - returned_first


class ZarrDirectReader(Reader):
    """`zarr-direct`: the `raw` arrays of a TS-Zarr bundle, sliced directly.

    The row shows what the TS-Zarr reader adds. An all-channel read slices
    every channel on a pool of `STACK_WORKERS` threads.
    """

    file_format = "zarr-direct"

    def open(self) -> None:
        """Open the group, which reads the root `zarr.json`."""
        group = zarr.open_group(zarr_store(self._fs, self._path), mode="r")
        keys = sorted(group.keys(), key=int)
        self._arrays = [_raw_array(group, key) for key in keys]
        self._pool = ThreadPoolExecutor(max_workers=STACK_WORKERS)

    def read(
        self, start_us: int, end_us: int, channel: int | None
    ) -> SampleArray:
        """Return the window, as `Reader.read` describes."""
        first, stop = self._samples(start_us, end_us)
        if channel is not None:
            return np.asarray(self._arrays[channel][first:stop])
        columns = self._pool.map(
            lambda array: np.asarray(array[first:stop]), self._arrays
        )
        return np.stack(list(columns), axis=1)

    def close(self) -> None:
        """Stop the thread pool."""
        self._pool.shutdown()


class OverheadReader(Reader):
    """`overhead`: one request for a 1-byte object.

    The read ignores its window and channel and returns no samples.
    """

    file_format = "overhead"

    def open(self) -> None:
        """Fetch nothing. The object is the same for every read."""

    def read(
        self, start_us: int, end_us: int, channel: int | None
    ) -> SampleArray:
        """Fetch the object and return an empty array."""
        data = self._fs.cat_file(self._path)
        if len(data) != 1:
            raise ValueError(
                f"object {self._path!r}: holds {len(data)} bytes, expected 1"
            )
        return np.empty(0, dtype=np.float32)


READERS: dict[str, type[Reader]] = {
    "edf": EdfReader,
    "nwb-hdf5-time": NwbHdf5Reader,
    "nwb-hdf5-channel": NwbHdf5Reader,
    "nwb-zarr-time": NwbZarrReader,
    "nwb-zarr-channel": NwbZarrReader,
    "tszarr": TszarrReader,
    "zarr-direct": ZarrDirectReader,
    "overhead": OverheadReader,
}
"""The reader class of each value of `schemas.FORMAT_ROWS`."""


def relative_path(file_format: str, recording_id: str) -> str:
    """Return the path under a store root of the object a row reads."""
    if file_format == "overhead":
        return schemas.ONE_BYTE_OBJECT
    if file_format == "zarr-direct":
        return schemas.relative_path(recording_id, "tszarr")
    return schemas.relative_path(recording_id, file_format)


def make_reader(
    file_format: str,
    filesystem: CountingFileSystem,
    store: str,
    roots: StoreRoots,
    spec: RecordingSpec,
) -> Reader:
    """Return the unopened reader of a row of `schemas.FORMAT_ROWS`.

    Raises:
        ValueError: file_format is not one of `schemas.FORMAT_ROWS`.
    """
    if file_format not in READERS:
        raise ValueError(
            f"format {file_format!r}: must be one of {list(READERS)}"
        )
    path = location(store, roots, relative_path(file_format, spec.id))
    return READERS[file_format](filesystem, path, spec)
