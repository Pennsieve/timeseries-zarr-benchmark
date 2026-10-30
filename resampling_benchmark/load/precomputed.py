"""The precomputed read: reduce the stored levels to pixel columns."""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from timeseries_zarr import Bins, Bundle, ContinuousChannel

from tszbench.fs import CountingFileSystem, Counts
from tszbench.minmax import COLUMNS, column_minmax

READ_THREADS = 32
"""Threads that read channels for all users of one process."""

LEVEL_RAW = 0
"""Level number of the raw samples."""


@dataclass(frozen=True)
class PrecomputedResult:
    """Output and cost of one precomputed read.

    Attributes:
        array: Array of shape (channels, rows, 2), float32.
        level: Level read, 0 for the raw samples.
        counts: Requests and bytes of the filesystem during the read.
        read_seconds: Seconds spent reading and decoding, summed over
            channels.
        reduce_seconds: Seconds spent reducing, summed over channels.
    """

    array: npt.NDArray[np.float32]
    level: int
    counts: Counts
    read_seconds: float
    reduce_seconds: float


def choose_level(
    bundle: Bundle, start_us: int, end_us: int, columns: int = COLUMNS
) -> int:
    """Return the coarsest level that holds at least `columns` bins.

    The channels share one grid, so the first channel decides. Level 0 means no
    level holds enough bins and the read uses the raw samples.
    """
    channel = next(iter(bundle.continuous.values()))
    return int(channel.level_for(start_us, end_us, bins=columns))


def _read_channel(
    channel: ContinuousChannel,
    start_us: int,
    end_us: int,
    level: int,
    columns: int,
) -> tuple[npt.NDArray[np.float32], float, float]:
    started = time.perf_counter()
    if level == LEVEL_RAW:
        samples = channel.read(start_us, end_us)
        lows = highs = np.asarray(samples.values, dtype=np.float32)
    else:
        bins = channel.read(start_us, end_us, level=level, members=())
        assert isinstance(bins, Bins)
        lows = np.asarray(bins.min, dtype=np.float32)
        highs = np.asarray(bins.max, dtype=np.float32)
    read_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reduced = column_minmax(lows, highs, columns)
    return reduced, read_seconds, time.perf_counter() - started


class PrecomputedReader:
    """Reads the minmax of every channel of one bundle from its levels."""

    def __init__(
        self,
        bundle: Bundle,
        filesystem: CountingFileSystem,
        pool: ThreadPoolExecutor,
        columns: int = COLUMNS,
    ) -> None:
        """Bind a reader to a bundle.

        Args:
            bundle: Open bundle, read through filesystem.
            filesystem: Counting filesystem that only this reader uses, so
                that a read's counts belong to that read.
            pool: Threads that read channels. May be shared by readers.
            columns: Pixel columns.
        """
        self._bundle = bundle
        self._filesystem = filesystem
        self._pool = pool
        self._columns = columns
        self._channels = list(bundle.continuous.values())

    def choose_level(self, start_us: int, end_us: int) -> int:
        """Return the level `read` chooses for a window."""
        return choose_level(self._bundle, start_us, end_us, self._columns)

    def read(
        self, start_us: int, end_us: int, level: int | None = None
    ) -> PrecomputedResult:
        """Return the minmax of every channel over a window.

        Args:
            start_us: Start of the window in microseconds from sample 0.
            end_us: End of the window, exclusive.
            level: Level to read, or None for the coarsest level that holds
                `columns` bins in the window.
        """
        chosen = self.choose_level(start_us, end_us) if level is None else level
        before = self._filesystem.counts()
        futures = [
            self._pool.submit(
                _read_channel,
                channel,
                start_us,
                end_us,
                chosen,
                self._columns,
            )
            for channel in self._channels
        ]
        parts = [future.result() for future in futures]
        counts = self._filesystem.counts() - before
        return PrecomputedResult(
            array=np.stack([part[0] for part in parts]),
            level=chosen,
            counts=counts,
            read_seconds=sum(part[1] for part in parts),
            reduce_seconds=sum(part[2] for part in parts),
        )
