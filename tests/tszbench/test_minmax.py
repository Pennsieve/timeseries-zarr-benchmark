import warnings

import numpy as np
import pytest

from tszbench.minmax import column_minmax


def columns_reference(values, columns):
    per_column = values.reshape(columns, -1)
    return np.stack([per_column.min(axis=1), per_column.max(axis=1)], axis=-1)


@pytest.mark.parametrize("per_column", [2, 3, 10, 600, 86_400])
def test_whole_columns_match_the_reference(per_column):
    columns = 48
    rng = np.random.default_rng(per_column)
    values = rng.standard_normal(columns * per_column).astype(np.float32)
    np.testing.assert_array_equal(
        column_minmax(values, values, columns),
        columns_reference(values, columns),
    )


def test_bins_reduce_their_minimum_and_maximum_separately():
    columns, per_column = 20, 4
    rng = np.random.default_rng(0)
    low = rng.standard_normal(columns * per_column).astype(np.float32)
    high = low + rng.uniform(0, 1, low.shape).astype(np.float32)
    result = column_minmax(low, high, columns)
    np.testing.assert_array_equal(
        result[:, 0], low.reshape(columns, per_column).min(axis=1)
    )
    np.testing.assert_array_equal(
        result[:, 1], high.reshape(columns, per_column).max(axis=1)
    )


def test_a_column_with_a_nan_entry_holds_the_minmax_of_the_others():
    columns, per_column = 10, 5
    values = np.arange(columns * per_column, dtype=np.float32)
    values[10] = np.nan
    values[14] = np.nan
    result = column_minmax(values, values, columns)
    np.testing.assert_array_equal(result[2], [11, 13])
    assert not np.isnan(result).any()


def test_a_column_of_nan_entries_is_nan_and_its_neighbors_are_not():
    columns, per_column = 10, 5
    values = np.arange(columns * per_column, dtype=np.float32)
    values[10:15] = np.nan
    result = column_minmax(values, values, columns)
    assert np.isnan(result[2]).all()
    assert not np.isnan(np.delete(result, 2, axis=0)).any()


def test_a_column_of_two_entries_skips_its_nan_entry():
    values = np.arange(40, dtype=np.float32)
    values[6] = np.nan
    result = column_minmax(values, values, 20)
    np.testing.assert_array_equal(result[3], [7, 7])


@pytest.mark.parametrize("per_column", [2, 3, 10, 600])
def test_columns_with_nan_runs_match_nanmin_and_nanmax(per_column):
    columns = 48
    rng = np.random.default_rng(per_column)
    values = rng.standard_normal(columns * per_column).astype(np.float32)
    for start in rng.integers(0, values.size, 6):
        values[start : start + int(rng.integers(1, 3 * per_column))] = np.nan
    per = values.reshape(columns, per_column)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.stack(
            [np.nanmin(per, axis=1), np.nanmax(per, axis=1)], axis=-1
        )
    np.testing.assert_array_equal(
        column_minmax(values, values, columns), expected
    )


def test_fewer_than_two_entries_per_column_returns_every_entry():
    values = np.arange(30, dtype=np.float32)
    result = column_minmax(values, values + 1, 20)
    assert result.shape == (30, 2)
    np.testing.assert_array_equal(result[:, 1], values + 1)


def test_two_entries_per_column_pairs_neighbors():
    values = np.arange(40, dtype=np.float32)[::-1].copy()
    result = column_minmax(values, values, 20)
    np.testing.assert_array_equal(result[:, 0], values[1::2])
    np.testing.assert_array_equal(result[:, 1], values[0::2])


def test_rejects_arrays_of_unequal_length():
    with pytest.raises(ValueError, match="equal length"):
        column_minmax(np.zeros(10, np.float32), np.zeros(9, np.float32), 2)


def test_a_strided_column_gives_the_minmax_of_its_copy():
    pairs = np.random.default_rng(5).normal(size=(8_000, 2)).astype(np.float32)
    expected = column_minmax(pairs[:, 0].copy(), pairs[:, 1].copy(), 1_000)
    assert np.array_equal(
        column_minmax(pairs[:, 0], pairs[:, 1], 1_000), expected
    )
