"""Split a window into pixel columns and groups of columns.

Column `c` of a window of `count` samples that starts at sample `first` covers
`[first + c * count // columns, first + (c + 1) * count // columns)`. A group
holds consecutive whole columns and about one raw shard of samples.
"""

import math
from dataclasses import dataclass

from tszbench.minmax import COLUMNS

GROUP_SAMPLES = 4_194_304
"""Samples of one raw shard. A group holds as many whole columns as fit."""

MICROSECONDS_PER_SECOND = 1_000_000


@dataclass(frozen=True)
class Group:
    """Consecutive columns of one window, read and reduced together.

    Attributes:
        first_column: Index of the first column of the group.
        columns: Number of columns the reduction splits the group into.
        first: Index of the first sample of the group in the channel.
        stop: Index after the last sample of the group.
    """

    first_column: int
    columns: int
    first: int
    stop: int

    @property
    def samples(self) -> int:
        """Number of samples in the group."""
        return self.stop - self.first

    @property
    def rows(self) -> int:
        """Rows the reduction of the group returns.

        A group with fewer than two samples per column returns one row per
        sample.
        """
        return (
            self.columns if self.samples >= 2 * self.columns else self.samples
        )


def column_bounds(
    first: int, count: int, columns: int = COLUMNS
) -> list[tuple[int, int]]:
    """Return the half-open sample range of every column of a window.

    Args:
        first: Index of the first sample of the window.
        count: Number of samples in the window, at least `2 * columns`.
        columns: Number of columns.

    Raises:
        ValueError: count is below `2 * columns`.
    """
    if count < 2 * columns:
        raise ValueError(
            f"window of {count} samples: must hold at least {2 * columns} "
            f"samples for {columns} columns"
        )
    return [
        (first + c * count // columns, first + (c + 1) * count // columns)
        for c in range(columns)
    ]


def plan_groups(
    first: int,
    count: int,
    columns: int = COLUMNS,
    group_samples: int = GROUP_SAMPLES,
) -> list[Group]:
    """Return the groups that cover a window once, in sample order.

    A window of fewer than `2 * columns` samples is one group that holds every
    sample. Otherwise a group holds as many whole columns as fit in
    `group_samples` samples, and at least one. The last group holds the
    remaining columns.

    Args:
        first: Index of the first sample of the window.
        count: Number of samples in the window, at least 1.
        columns: Number of columns.
        group_samples: Samples a group aims to hold.

    Raises:
        ValueError: count is below 1.
    """
    if count < 1:
        raise ValueError(f"window of {count} samples: must hold at least 1")
    if count < 2 * columns:
        return [Group(0, columns, first, first + count)]
    bounds = column_bounds(first, count, columns)
    per_group = max(1, group_samples // math.ceil(count / columns))
    groups = []
    for first_column in range(0, columns, per_group):
        last_column = min(first_column + per_group, columns)
        groups.append(
            Group(
                first_column=first_column,
                columns=last_column - first_column,
                first=bounds[first_column][0],
                stop=bounds[last_column - 1][1],
            )
        )
    return groups


def sample_bounds_us(first: int, stop: int, rate_hz: int) -> tuple[int, int]:
    """Return whole-microsecond bounds whose read holds samples `[first, stop)`.

    The reader returns every sample that overlaps its window. The start is the
    first whole microsecond at or after the start of sample `first`. The end is
    the last whole microsecond at or before the start of sample `stop`. The
    bounds hold exactly the samples when the sample period is at least 1
    microsecond.

    Raises:
        ValueError: The range is empty or the sample period is below 1
            microsecond.
    """
    if rate_hz > MICROSECONDS_PER_SECOND or stop <= first:
        raise ValueError(
            f"samples [{first}, {stop}) at {rate_hz} Hz: need a non-empty "
            f"range and a rate of at most {MICROSECONDS_PER_SECOND} Hz"
        )
    start_us = -((-first * MICROSECONDS_PER_SECOND) // rate_hz)
    end_us = (stop * MICROSECONDS_PER_SECOND) // rate_hz
    return start_us, end_us
