import math
from fractions import Fraction

import pytest

from resampling_benchmark.service.groups import (
    GROUP_SAMPLES,
    Group,
    column_bounds,
    plan_groups,
    sample_bounds_us,
)
from tszbench.minmax import COLUMNS
from tszbench.recordings import RECORDINGS, sample_range

FIRST = 12_345


def covered(groups: list[Group]) -> list[tuple[int, int]]:
    return [(group.first, group.stop) for group in groups]


@pytest.mark.parametrize(
    "count", [2 * COLUMNS, 2 * COLUMNS + 1, 7 * COLUMNS + 1999, 123_457]
)
def test_columns_cover_every_sample_once(count: int) -> None:
    bounds = column_bounds(FIRST, count)
    assert len(bounds) == COLUMNS
    assert bounds[0][0] == FIRST
    assert bounds[-1][1] == FIRST + count
    for (_, stop), (start, _) in zip(bounds, bounds[1:], strict=False):
        assert stop == start
    widths = {stop - start for start, stop in bounds}
    assert widths <= {count // COLUMNS, count // COLUMNS + 1}


def test_columns_reject_a_window_below_two_samples_per_column() -> None:
    with pytest.raises(ValueError, match="at least 4000"):
        column_bounds(0, 2 * COLUMNS - 1)


@pytest.mark.parametrize(
    "count",
    [2 * COLUMNS, 2 * COLUMNS + 1, 10 * COLUMNS + 777, 3_000_001, 20_000_000],
)
@pytest.mark.parametrize("group_samples", [GROUP_SAMPLES, 9_000, 1])
def test_groups_cover_every_sample_once(count: int, group_samples: int) -> None:
    groups = plan_groups(FIRST, count, group_samples=group_samples)
    assert groups[0].first == FIRST
    assert groups[-1].stop == FIRST + count
    for left, right in zip(groups, groups[1:], strict=False):
        assert left.stop == right.first
        assert left.first_column + left.columns == right.first_column
    assert sum(group.columns for group in groups) == COLUMNS
    assert sum(group.samples for group in groups) == count
    assert groups[-1].first_column + groups[-1].columns == COLUMNS


def test_group_edges_are_column_edges() -> None:
    count = 7 * COLUMNS + 1999
    bounds = column_bounds(FIRST, count)
    for group in plan_groups(FIRST, count, group_samples=9_000):
        assert group.first == bounds[group.first_column][0]
        last = group.first_column + group.columns - 1
        assert group.stop == bounds[last][1]


def test_a_window_below_two_samples_per_column_is_one_group() -> None:
    groups = plan_groups(FIRST, 2 * COLUMNS - 1)
    assert groups == [Group(0, COLUMNS, FIRST, FIRST + 2 * COLUMNS - 1)]
    assert groups[0].rows == 2 * COLUMNS - 1


def test_a_window_of_one_sample_is_one_group_of_one_row() -> None:
    (group,) = plan_groups(5, 1)
    assert (group.samples, group.rows) == (1, 1)


def test_a_group_of_whole_columns_returns_one_row_per_column() -> None:
    for group in plan_groups(0, 20_000_000):
        assert group.rows == group.columns


def test_a_column_wider_than_a_shard_makes_one_group_per_column() -> None:
    groups = plan_groups(0, COLUMNS * (GROUP_SAMPLES + 1))
    assert len(groups) == COLUMNS
    assert {group.columns for group in groups} == {1}


def test_groups_of_the_24_hour_window_of_r3_hold_48_columns() -> None:
    spec = RECORDINGS["R3"]
    groups = plan_groups(0, spec.samples_per_channel)
    per_column = math.ceil(spec.samples_per_channel / COLUMNS)
    assert GROUP_SAMPLES // per_column == 48
    assert len(groups) == math.ceil(COLUMNS / 48)
    assert groups[0].columns == 48
    assert groups[0].samples <= GROUP_SAMPLES


def test_groups_without_a_remainder_hold_equal_columns() -> None:
    groups = plan_groups(0, 30 * COLUMNS, group_samples=7 * 30)
    assert {group.columns for group in groups[:-1]} == {7}
    assert groups[-1].columns == COLUMNS - 7 * (len(groups) - 1)
    assert {group.samples // group.columns for group in groups} == {30}


@pytest.mark.parametrize("rate_hz", [256, 512, 2000, 30_000])
@pytest.mark.parametrize("first", [0, 1, 4093, 65_537, 1_000_003])
@pytest.mark.parametrize("count", [2, 3, 4000, 4097])
def test_sample_bounds_make_the_reader_return_exactly_the_samples(
    rate_hz: int, first: int, count: int
) -> None:
    stop = first + count
    start_us, end_us = sample_bounds_us(first, stop, rate_hz)
    period = Fraction(1_000_000, rate_hz)
    assert math.floor(Fraction(start_us) / period) == first
    assert math.ceil(Fraction(end_us) / period) == stop
    if first * 1_000_000 % rate_hz == 0:
        assert sample_range(start_us, end_us, rate_hz) == (first, stop)


def test_sample_bounds_reject_an_empty_range() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        sample_bounds_us(5, 5, 2000)
