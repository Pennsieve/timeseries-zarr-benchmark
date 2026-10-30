from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from timeseries_zarr import WriteOpts, write_bundle

from tszbench import schemas

RATE_HZ = 256.0
SAMPLES = 8192


class _Channel:
    unit = "uV"

    def __init__(self, index: int) -> None:
        self.id = f"ch{index}"
        self.name = f"Channel {index}"
        self._index = index

    def rate_hz(self) -> float:
        return RATE_HZ

    def start_us(self) -> int:
        return 0

    def num_samples(self) -> int:
        return SAMPLES

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        rng = np.random.default_rng(self._index)
        return rng.normal(0.0, 20.0, SAMPLES).astype(np.float32)[start:stop]


def write_small_bundle(path: Path) -> None:
    """Write a bundle of two channels with levels, mean, variance, and count."""
    write_bundle(
        [_Channel(0), _Channel(1)],
        [],
        output_path=path,
        opts=WriteOpts(min_bins=4, inner_len=512, write_var=True),
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A store root holding every format of recording R9 as small files."""
    recording = tmp_path / "root" / "R9"
    recording.mkdir(parents=True)
    for file_format in schemas.FORMATS:
        path = recording / schemas.FILE_NAMES[file_format]
        if file_format == "tszarr":
            write_small_bundle(path)
        elif "zarr" in file_format:
            (path / "group").mkdir(parents=True)
            (path / "zarr.json").write_bytes(b"x" * 10)
            (path / "group" / "0").write_bytes(b"y" * 100)
        else:
            path.write_bytes(b"z" * 1000)
    return tmp_path / "root"
