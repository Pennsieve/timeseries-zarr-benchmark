"""Write a synthetic recording as an NWB file on HDF5 or on Zarr."""

import shutil
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import hdf5plugin
import numpy as np
import numpy.typing as npt
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.data_utils import GenericDataChunkIterator
from hdmf_zarr import NWBZarrIO
from hdmf_zarr.utils import ZarrDataIO
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries
from zarr.codecs import ZstdCodec

from tszbench.recordings import START_TIME, RecordingSpec
from tszbench.signal import Synthesizer

ZSTD_LEVEL = 3
"""Zstd level of the ElectricalSeries data."""

TIME_CHUNK_BYTES = 2**20
"""Size of one `-time` chunk in bytes, with every channel in the chunk."""

CHANNEL_CHUNK_ROWS = 8192
"""Rows of one `-channel` chunk, which holds a single channel."""

BUFFER_BYTES = 32 * 2**20
"""Target size of one buffer the iterator hands to the writer. A buffer holds
whole chunks of every channel, so each chunk is written once."""

SERIES_NAME = "ElectricalSeries"
"""Name of the series in the acquisition group."""

CONVERSION = 1e-6
"""Multiplier from the stored microvolts to volts."""

ChunkLayout = Literal["time", "channel"]


def chunk_shape(spec: RecordingSpec, layout: ChunkLayout) -> tuple[int, int]:
    """Return the (rows, channels) chunk shape of the data array.

    A `time` chunk holds every channel and about `TIME_CHUNK_BYTES` of
    float32. A `channel` chunk holds `CHANNEL_CHUNK_ROWS` rows of one channel.
    A recording shorter than the chunk gets a chunk of its own length.
    """
    if layout == "time":
        rows = max(1, TIME_CHUNK_BYTES // (4 * spec.channels))
        return min(rows, spec.samples_per_channel), spec.channels
    return min(CHANNEL_CHUNK_ROWS, spec.samples_per_channel), 1


def buffer_rows(spec: RecordingSpec, layout: ChunkLayout) -> int:
    """Return the rows of one buffer, a multiple of the chunk rows.

    The buffer holds at most `BUFFER_BYTES` of float32 and at least one chunk
    row of every channel. A recording shorter than that gets one buffer of its
    own length.
    """
    rows = chunk_shape(spec, layout)[0]
    rows *= max(1, BUFFER_BYTES // (4 * rows * spec.channels))
    return min(rows, spec.samples_per_channel)


class SynthesizedIterator(GenericDataChunkIterator):  # type: ignore[misc]
    """Feed synthesized blocks of one recording to a writer in sample order.

    Every buffer spans every channel and a whole number of chunk rows, so a
    writer fills each chunk with one write.
    """

    def __init__(
        self, spec: RecordingSpec, layout: ChunkLayout, workers: int = 1
    ) -> None:
        """Prepare the blocks of spec for the chunk layout.

        Args:
            spec: The recording to synthesize.
            layout: `time` or `channel`.
            workers: Number of threads that generate channels at once.
        """
        self._spec = spec
        rows = buffer_rows(spec, layout)
        self._blocks: Iterator[npt.NDArray[np.float32]] = iter(
            Synthesizer(spec, workers).blocks(rows)
        )
        self._next_row = 0
        super().__init__(
            buffer_shape=(rows, spec.channels),
            chunk_shape=chunk_shape(spec, layout),
            display_progress=False,
        )

    def _get_data(
        self, selection: tuple[slice, ...]
    ) -> npt.NDArray[np.float32]:
        """Return the next block, which must be the one the selection names."""
        rows, columns = selection
        if rows.start != self._next_row or columns != slice(
            0, self._spec.channels
        ):
            raise ValueError(
                f"selection {selection}: the iterator reads in sample order "
                f"from row {self._next_row} across every channel"
            )
        block = next(self._blocks)
        self._next_row += block.shape[0]
        return block

    def _get_maxshape(self) -> tuple[int, int]:
        """Return the shape of the whole array."""
        return self._spec.samples_per_channel, self._spec.channels

    def _get_dtype(self) -> np.dtype[np.float32]:
        """Return float32."""
        return np.dtype(np.float32)


def build_file(spec: RecordingSpec, data: Any) -> NWBFile:
    """Return an NWB file of spec whose ElectricalSeries holds data.

    The file has one device, one electrode group, and one electrode row for
    each channel. The series stores microvolts with a conversion of
    `CONVERSION` to volts, starts at time 0.0, and runs at `spec.rate_hz`.
    """
    nwbfile = NWBFile(
        session_description=f"synthetic recording {spec.id}",
        identifier=spec.id,
        session_start_time=START_TIME,
    )
    device = nwbfile.create_device(name="device")
    group = nwbfile.create_electrode_group(
        name="group",
        description="all channels",
        location="synthetic",
        device=device,
    )
    for _ in range(spec.channels):
        nwbfile.add_electrode(
            x=0.0,
            y=0.0,
            z=0.0,
            imp=0.0,
            location="synthetic",
            filtering="none",
            group=group,
        )
    region = nwbfile.create_electrode_table_region(
        region=list(range(spec.channels)), description="all electrodes"
    )
    nwbfile.add_acquisition(
        ElectricalSeries(
            name=SERIES_NAME,
            data=data,
            electrodes=region,
            starting_time=0.0,
            rate=float(spec.rate_hz),
            conversion=CONVERSION,
        )
    )
    return nwbfile


def write_nwb_hdf5(
    spec: RecordingSpec,
    output_path: Path,
    layout: ChunkLayout,
    workers: int = 1,
) -> None:
    """Write spec as NWB on HDF5, with Zstd level `ZSTD_LEVEL`.

    Args:
        spec: The recording to write.
        output_path: The `.nwb` file. A file already there is replaced. A
            failed write leaves a partial file.
        layout: `time` for chunks of every channel, `channel` for chunks of
            one channel.
        workers: Number of threads that generate channels at once.
    """
    with warnings.catch_warnings():
        # hdmf warns that a plugin filter may be missing on other machines.
        warnings.simplefilter("ignore", UserWarning)
        data = H5DataIO(
            SynthesizedIterator(spec, layout, workers),
            chunks=chunk_shape(spec, layout),
            allow_plugin_filters=True,
            **hdf5plugin.Zstd(clevel=ZSTD_LEVEL),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(output_path), mode="w") as io:
        io.write(build_file(spec, data))


def write_nwb_zarr(
    spec: RecordingSpec,
    output_path: Path,
    layout: ChunkLayout,
    workers: int = 1,
) -> None:
    """Write spec as NWB on Zarr, with Zstd level `ZSTD_LEVEL`.

    Args:
        spec: The recording to write.
        output_path: The `.nwb.zarr` directory. A directory already there is
            replaced. A failed write leaves a partial directory.
        layout: `time` for chunks of every channel, `channel` for chunks of
            one channel.
        workers: Number of threads that generate channels at once.
    """
    data = ZarrDataIO(
        SynthesizedIterator(spec, layout, workers),
        chunks=chunk_shape(spec, layout),
        compressors=ZstdCodec(level=ZSTD_LEVEL),
    )
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NWBZarrIO(str(output_path), mode="w") as io:
        io.write(build_file(spec, data))
