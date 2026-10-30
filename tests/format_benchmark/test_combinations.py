from pathlib import Path

import pytest

from format_benchmark.combinations import (
    Combination,
    complete_parts,
    expected_rows,
    is_complete,
    parse_part,
    plan,
)
from format_benchmark.positions import WINDOWS_S
from tszbench import schemas
from tszbench.csvmeta import CsvWriter

COMBINATION = Combination("R1", "http", "tszarr", "all", 60)


BUDGET_S = 100.0
MIN_READS = 3


def write_part(
    path: Path,
    columns: tuple[str, ...],
    rows: int,
    options: dict[str, str] | None = None,
) -> None:
    header = {"combination": "x"} | (options or {})
    with CsvWriter(path, columns, header) as writer:
        for index in range(rows):
            writer.write(dict.fromkeys(columns, index))


def write_combination(
    combination: Combination,
    results: Path,
    made: int,
    planned: int,
    read_budget_s: float = BUDGET_S,
    min_reads: int = MIN_READS,
    open_rows: int = 1,
) -> None:
    """Write a part that made reads of planned reads under the options."""
    options = {
        "read_budget_s": str(read_budget_s),
        "min_reads": str(min_reads),
    }
    write_part(
        combination.reads_path(results), schemas.FORMAT_READS, made, options
    )
    with CsvWriter(
        combination.open_path(results), schemas.FORMAT_OPENS, options
    ) as writer:
        for _ in range(open_rows):
            writer.write(
                dict.fromkeys(schemas.FORMAT_OPENS, 0)
                | {"reads": made, "planned_reads": planned}
            )


def test_stem_names_store_format_kind_and_window() -> None:
    assert COMBINATION.stem == "http-tszarr-all-60s"


def test_paths_sit_under_the_recording_directory(tmp_path: Path) -> None:
    reads = COMBINATION.reads_path(tmp_path)
    opened = COMBINATION.open_path(tmp_path)
    assert reads.parent == opened.parent == tmp_path / "format-parts" / "R1"
    assert reads.name == "http-tszarr-all-60s.csv"
    assert opened.name == "http-tszarr-all-60s-open.csv"


def test_plan_orders_by_recording_store_format_kind_and_window() -> None:
    measured, skipped = plan(
        ["R1", "R2"],
        ["local", "http"],
        ["edf", "tszarr"],
        ["one", "all"],
        [1, 10],
    )
    assert skipped == []
    assert len(measured) == 2 * 2 * 2 * 2 * 2
    keys = [
        (c.recording, c.store, c.file_format, c.read_kind, c.window_s)
        for c in measured
    ]
    assert keys == sorted(
        keys,
        key=lambda key: (
            ["R1", "R2"].index(key[0]),
            ["local", "http"].index(key[1]),
            ["edf", "tszarr"].index(key[2]),
            ["one", "all"].index(key[3]),
            key[4],
        ),
    )


def test_plan_is_the_same_on_every_call() -> None:
    arguments = (["R2", "R4"], schemas.STORES, schemas.FORMAT_ROWS)
    first = plan(*arguments, schemas.READ_KINDS, WINDOWS_S)
    assert first == plan(*arguments, schemas.READ_KINDS, WINDOWS_S)


def test_plan_skips_edf_of_r4_and_big_all_channel_reads() -> None:
    measured, skipped = plan(
        ["R4"], ["local"], ["edf", "tszarr"], ["one", "all"], WINDOWS_S
    )
    reasons = {
        (c.file_format, c.read_kind, c.window_s): reason
        for c, reason in skipped
    }
    assert {key for key in reasons if key[0] == "edf"} == {
        ("edf", kind, window) for kind in ("one", "all") for window in WINDOWS_S
    }
    assert {key for key in reasons if key[0] == "tszarr"} == {
        ("tszarr", "all", 600)
    }
    assert len(measured) == 4 + 3


def test_plan_measures_the_overhead_once_for_each_recording_and_store() -> None:
    measured, skipped = plan(
        ["R1", "R2"], ["local", "s3"], ["overhead"], ["one", "all"], [10, 60]
    )
    assert [
        (c.recording, c.store, c.read_kind, c.window_s) for c in measured
    ] == [
        ("R1", "local", "one", 10),
        ("R1", "s3", "one", 10),
        ("R2", "local", "one", 10),
        ("R2", "s3", "one", 10),
    ]
    assert all(reason for _, reason in skipped)
    assert len(skipped) == 4 * 3


def test_expected_rows_follow_the_repeats_of_the_window() -> None:
    assert (
        expected_rows(Combination("R1", "local", "edf", "one", 600), None) == 20
    )
    assert expected_rows(Combination("R1", "local", "edf", "one", 600), 5) == 5


def complete(
    results: Path,
    max_repeats: int | None = 3,
    read_budget_s: float = BUDGET_S,
    min_reads: int = MIN_READS,
    combination: Combination = COMBINATION,
) -> bool:
    return is_complete(
        combination, results, max_repeats, read_budget_s, min_reads
    )


def test_a_combination_without_files_is_incomplete(tmp_path: Path) -> None:
    assert not complete(tmp_path)


def test_a_part_with_every_planned_read_is_complete(tmp_path: Path) -> None:
    write_combination(COMBINATION, tmp_path, 3, 3)
    assert complete(tmp_path)


def test_a_part_with_every_planned_read_is_complete_under_other_options(
    tmp_path: Path,
) -> None:
    write_combination(COMBINATION, tmp_path, 3, 3)
    assert complete(tmp_path, read_budget_s=5, min_reads=1)


def test_a_part_that_stopped_early_is_complete_under_the_same_options(
    tmp_path: Path,
) -> None:
    write_combination(COMBINATION, tmp_path, 3, 100)
    assert complete(tmp_path, max_repeats=None)


def test_a_part_that_stopped_early_is_incomplete_under_a_larger_budget(
    tmp_path: Path,
) -> None:
    write_combination(COMBINATION, tmp_path, 3, 100)
    assert not complete(tmp_path, max_repeats=None, read_budget_s=BUDGET_S * 2)


def test_a_part_that_stopped_early_is_incomplete_under_a_larger_minimum(
    tmp_path: Path,
) -> None:
    write_combination(COMBINATION, tmp_path, 3, 100)
    assert not complete(tmp_path, max_repeats=None, min_reads=MIN_READS + 1)


def test_a_part_that_made_fewer_reads_than_the_minimum_is_incomplete(
    tmp_path: Path,
) -> None:
    write_combination(COMBINATION, tmp_path, 2, 100)
    assert not complete(tmp_path, max_repeats=None)


def test_a_part_with_a_wrong_planned_count_is_incomplete(
    tmp_path: Path,
) -> None:
    write_combination(COMBINATION, tmp_path, 3, 3)
    assert not complete(tmp_path, max_repeats=5)
    assert not complete(tmp_path, max_repeats=2)


def test_a_part_without_the_options_in_its_header_is_incomplete_if_early(
    tmp_path: Path,
) -> None:
    write_combination(COMBINATION, tmp_path, 3, 100)
    write_part(COMBINATION.reads_path(tmp_path), schemas.FORMAT_READS, 3)
    assert not complete(tmp_path, max_repeats=None)


@pytest.mark.parametrize("rows", [2, 4])
def test_a_part_whose_reads_file_disagrees_with_the_open_row_is_incomplete(
    tmp_path: Path, rows: int
) -> None:
    write_combination(COMBINATION, tmp_path, 3, 3)
    write_part(COMBINATION.reads_path(tmp_path), schemas.FORMAT_READS, rows)
    assert not complete(tmp_path)


def test_a_part_with_two_open_rows_is_incomplete(tmp_path: Path) -> None:
    write_combination(COMBINATION, tmp_path, 3, 3, open_rows=2)
    assert not complete(tmp_path)


def test_a_part_without_the_open_file_is_incomplete(tmp_path: Path) -> None:
    write_combination(COMBINATION, tmp_path, 3, 3)
    COMBINATION.open_path(tmp_path).unlink()
    assert not complete(tmp_path)


def test_parse_part_reads_back_the_combination_of_a_reads_path(
    tmp_path: Path,
) -> None:
    for combination in (
        COMBINATION,
        Combination("R2-1h", "s3", "zarr-direct", "one", 600),
    ):
        assert parse_part(combination.reads_path(tmp_path)) == combination


def test_parse_part_ignores_open_csvs_and_stray_files(tmp_path: Path) -> None:
    parts = tmp_path / "format-parts" / "R1"
    assert parse_part(COMBINATION.open_path(tmp_path)) is None
    assert parse_part(parts / "notes.csv") is None
    assert parse_part(parts / "http-tszarr-all-61s.csv") is None
    assert (
        parse_part(tmp_path / "format-parts" / "R9" / "http-edf-one-1s.csv")
        is None
    )


def test_complete_parts_lists_whole_combinations_in_plan_order(
    tmp_path: Path,
) -> None:
    later = Combination("R2", "local", "edf", "one", 1)
    earlier = Combination("R1", "s3", "tszarr", "one", 10)
    broken = Combination("R1", "local", "edf", "one", 1)
    for combination in (later, earlier, broken):
        rows = expected_rows(combination, 2)
        write_combination(combination, tmp_path, rows, rows)
    write_combination(broken, tmp_path, 2, 2, open_rows=2)
    assert complete_parts(tmp_path, 2, BUDGET_S, MIN_READS) == [earlier, later]


def test_complete_parts_leaves_out_a_part_that_stopped_under_other_options(
    tmp_path: Path,
) -> None:
    early = Combination("R1", "local", "edf", "one", 1)
    whole = Combination("R1", "local", "edf", "one", 10)
    write_combination(early, tmp_path, MIN_READS, 100)
    write_combination(whole, tmp_path, 100, 100)
    assert complete_parts(tmp_path, None, BUDGET_S, MIN_READS) == [early, whole]
    assert complete_parts(tmp_path, None, BUDGET_S * 2, MIN_READS) == [whole]
