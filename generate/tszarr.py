"""Write a synthetic recording as a TS-Zarr bundle."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
from timeseries_zarr import (
    ContinuousChannelMetadata,
    RecordingMeta,
    WriteOpts,
    write_bundle,
)

from tszbench.recordings import START_TIME, RecordingSpec
from tszbench.signal import Synthesizer

BLOCK_ROWS = 16_384
"""Rows of one block the synthesizer hands to the writer. A block holds
`BLOCK_ROWS * channels * 4` bytes of float32, which is 24 MiB for R4 with 384
channels. The generation of that block holds about the same in int16 and a
few float64 columns per worker."""

ZSTD_LEVEL = 3
"""Zstd level of every array in the bundle."""

UNIT = "uV"
"""Unit of every channel."""

_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
START_US = (START_TIME - _UNIX_EPOCH) // timedelta(microseconds=1)
"""Time of sample 0 in microseconds since the Unix epoch."""


@dataclass(frozen=True)
class SynthesizedChannel:
    """The metadata of one synthesized channel."""

    id: str
    name: str
    unit: str
    rate_hz_value: int
    start_us_value: int
    num_samples_value: int

    def rate_hz(self) -> float:
        """Return the sample rate in hertz."""
        return float(self.rate_hz_value)

    def start_us(self) -> int:
        """Return the microseconds since the Unix epoch of sample 0."""
        return self.start_us_value

    def num_samples(self) -> int:
        """Return the number of samples."""
        return self.num_samples_value


class SynthesizedSource:
    """A multi-channel source that synthesizes each window on request."""

    def __init__(self, spec: RecordingSpec, workers: int = 1) -> None:
        """Prepare one channel for each channel of spec.

        Args:
            spec: The recording to synthesize.
            workers: Number of threads that generate channels at once.
        """
        self._synthesizer = Synthesizer(spec, workers)
        self._channels = [
            SynthesizedChannel(
                id=f"ch{index:03d}",
                name=f"ch{index:03d}",
                unit=UNIT,
                rate_hz_value=spec.rate_hz,
                start_us_value=START_US,
                num_samples_value=spec.samples_per_channel,
            )
            for index in range(spec.channels)
        ]

    def channels(self) -> Sequence[ContinuousChannelMetadata]:
        """Return the channels in channel order."""
        return self._channels

    def iter_windows(self) -> Iterator[Sequence[npt.NDArray[np.float32]]]:
        """Yield windows of `BLOCK_ROWS` rows, the last one shorter.

        Each window holds one column view of a synthesizer block per channel.
        """
        for block in self._synthesizer.blocks(BLOCK_ROWS):
            yield list(block.T)


def write_tszarr(
    spec: RecordingSpec, output_path: Path, workers: int = 1
) -> None:
    """Write spec as a TS-Zarr bundle at output_path.

    The bundle holds raw, minmax, mean, var, and count at every level, with
    the default inner chunk, shard size, and level threshold of `WriteOpts`.
    Time counts from `START_TIME`.

    Args:
        spec: The recording to write.
        output_path: Directory of the bundle. A bundle already there is
            replaced.
        workers: Number of threads that generate channels at once.
    """
    write_bundle(
        [SynthesizedSource(spec, workers)],
        [],
        output_path=output_path,
        opts=WriteOpts(zstd_level=ZSTD_LEVEL, write_var=True),
        recording=RecordingMeta(
            session_start_time=START_TIME, reference_time=START_TIME
        ),
    )
