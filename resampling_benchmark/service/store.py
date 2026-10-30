"""Open a recording's TS-Zarr bundle in S3."""

from timeseries_zarr import Bundle, open_bundle

from tszbench.fs import CountingFileSystem, StoreRoots, location, zarr_store
from tszbench.schemas import relative_path


def open_recording(
    filesystem: CountingFileSystem, roots: StoreRoots, recording_id: str
) -> Bundle:
    """Open the bundle of a recording through filesystem.

    Args:
        filesystem: Counting filesystem of the S3 store.
        roots: Store roots. Only `s3` is read.
        recording_id: Recording id such as `R3`.
    """
    path = location("s3", roots, relative_path(recording_id, "tszarr"))
    return open_bundle(zarr_store(filesystem, path))
