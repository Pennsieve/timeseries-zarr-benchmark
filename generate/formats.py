"""Write one file of a recording in any of the benchmark formats."""

import shutil
from pathlib import Path

from generate.nwb import ChunkLayout, write_nwb_hdf5, write_nwb_zarr
from generate.tszarr import write_tszarr
from tszbench import schemas
from tszbench.edf import write_edf
from tszbench.recordings import RecordingSpec
from tszbench.signal import Synthesizer

EDF_BLOCK_ROWS = 32_768
"""Rows of one block the synthesizer hands to the EDF writer."""

_NWB_FORMATS: dict[str, tuple[str, ChunkLayout]] = {
    "nwb-hdf5-time": ("hdf5", "time"),
    "nwb-hdf5-channel": ("hdf5", "channel"),
    "nwb-zarr-time": ("zarr", "time"),
    "nwb-zarr-channel": ("zarr", "channel"),
}


def formats_of(spec: RecordingSpec) -> tuple[str, ...]:
    """Return the formats written for spec, in `schemas.FORMATS` order.

    A recording without an EDF record duration has no EDF file.
    """
    return tuple(
        file_format
        for file_format in schemas.FORMATS
        if file_format != "edf" or spec.edf_record_s is not None
    )


def write_format(
    spec: RecordingSpec, file_format: str, root: Path, workers: int = 1
) -> Path:
    """Write spec in file_format and return the path of the file.

    The file is `root / schemas.relative_path(spec.id, file_format)`. A file
    already there is replaced. A failed write removes the partial file.

    Args:
        spec: The recording to write.
        file_format: One of `schemas.FORMATS`.
        root: Directory that holds one directory for each recording.
        workers: Number of threads that generate channels at once.

    Raises:
        ValueError: The format is unknown, or it is `edf` and spec has no EDF
            record duration.
    """
    if file_format not in schemas.FORMATS:
        raise ValueError(
            f"format {file_format!r}: must be one of {list(schemas.FORMATS)}"
        )
    if file_format == "edf" and spec.edf_record_s is None:
        raise ValueError(
            f"format 'edf' of recording {spec.id!r}: the recording has no EDF "
            "record duration"
        )
    path = root / schemas.relative_path(spec.id, file_format)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if file_format == "edf":
            write_edf(
                path,
                spec,
                Synthesizer(spec, workers).digital_blocks(EDF_BLOCK_ROWS),
            )
        elif file_format == "tszarr":
            write_tszarr(spec, path, workers)
        else:
            backend, layout = _NWB_FORMATS[file_format]
            writer = write_nwb_hdf5 if backend == "hdf5" else write_nwb_zarr
            writer(spec, path, layout, workers)
    except BaseException:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        raise
    return path


def write_one_byte_object(root: Path) -> Path:
    """Write the 1-byte object that the `overhead` rows read and return its path.

    The object is `root / schemas.ONE_BYTE_OBJECT`. Its byte is zero.
    """
    path = root / schemas.ONE_BYTE_OBJECT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")
    return path
