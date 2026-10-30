"""Per-column minimum and maximum of a window, reduced by tsdownsample."""

import numpy as np
import numpy.typing as npt
from tsdownsample import MinMaxDownsampler

COLUMNS = 2000
"""Pixel columns of every resampling benchmark response."""

_DOWNSAMPLER = MinMaxDownsampler()


def column_minmax(
    minimum: npt.NDArray[np.float32],
    maximum: npt.NDArray[np.float32],
    columns: int = COLUMNS,
) -> npt.NDArray[np.float32]:
    """Return the minmax of a run of entries as a (rows, 2) float32 array.

    An entry is a raw sample or a level bin. `minimum` and `maximum` hold the
    lowest and highest value of each entry, so a raw read passes its samples as
    both. tsdownsample splits the entries into `columns` columns. Row `i` holds
    the minimum and the maximum of the entries of column `i` that are not NaN.
    A column whose entries are all NaN is NaN in both.

    tsdownsample returns every entry when the entries number fewer than two per
    column. The result then holds one row per entry, the entry's own minimum
    and maximum.

    Args:
        minimum: Rank-1 float32 array, one value per entry.
        maximum: Rank-1 float32 array of the same length. May be `minimum`.
        columns: Number of pixel columns, at least 1.

    Raises:
        ValueError: The arrays differ in length or columns is below 1.
    """
    if columns < 1:
        raise ValueError(f"columns is {columns}: must be at least 1")
    if minimum.shape != maximum.shape or minimum.ndim != 1:
        raise ValueError(
            f"minimum {minimum.shape} and maximum {maximum.shape}: must be "
            "rank 1 and of equal length"
        )
    # tsdownsample rejects a strided view, such as one column of a minmax array.
    same = maximum is minimum
    minimum = np.ascontiguousarray(minimum)
    maximum = minimum if same else np.ascontiguousarray(maximum)
    if minimum.shape[0] < 2 * columns:
        return np.stack([minimum, maximum], axis=-1).astype(np.float32)
    low_index = _DOWNSAMPLER.downsample(minimum, n_out=2 * columns)
    high_index = (
        low_index
        if same
        else _DOWNSAMPLER.downsample(maximum, n_out=2 * columns)
    )
    # The pair of each column is its argmin and argmax in sample order. A
    # column of two entries returns both, and fmin and fmax skip a NaN one.
    low_pair = minimum[low_index].reshape(columns, 2)
    high_pair = maximum[high_index].reshape(columns, 2)
    low = np.fmin(low_pair[:, 0], low_pair[:, 1])
    high = np.fmax(high_pair[:, 0], high_pair[:, 1])
    return np.stack([low, high], axis=-1).astype(np.float32)
