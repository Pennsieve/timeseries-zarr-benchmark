from pathlib import Path

import numpy as np
import pytest

import format_benchmark.worker as worker
from format_benchmark.combinations import Combination
from format_benchmark.positions import positions, selection
from tests.format_benchmark.conftest import TINY_ID, TINY_SPEC
from tszbench import schemas
from tszbench.csvmeta import read_header, read_rows
from tszbench.fs import StoreRoots

WINDOW_S = 10
REPEATS = 3
PLANNED = 10
"""Planned reads of the budget tests."""

TINY_KINDS = tuple(
    kind
    for kind in schemas.READ_KINDS
    if (schemas.subset_size(kind) or 0) < TINY_SPEC.channels
)
"""Read kinds with a block narrower than `TINY_SPEC`."""


class FakeClock:
    """A clock that moves only when a fake read advances it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeReader:
    """A reader whose every read takes `seconds` on the fake clock."""

    def __init__(self, clock: FakeClock, seconds: float) -> None:
        self.clock = clock
        self.seconds = seconds

    def open(self) -> None:
        pass

    def read(
        self, start_us: int, end_us: int, channel: int | range | None
    ) -> "np.ndarray[tuple[int], np.dtype[np.float64]]":
        self.clock.now += self.seconds
        return np.zeros(1)

    def close(self) -> None:
        pass


def run_timed(
    tmp_path: Path,
    tiny_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_seconds: float,
    read_budget_s: float,
    min_reads: int,
    max_repeats: int = PLANNED,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Run a combination whose reads take read_seconds each.

    Returns the read rows and the open row.
    """
    clock = FakeClock()
    reader = FakeReader(clock, read_seconds)
    monkeypatch.setattr(worker, "perf_counter", clock)
    monkeypatch.setattr(worker, "make_reader", lambda *arguments: reader)
    roots = StoreRoots(local=tiny_root, http="", s3="")
    combination = Combination(TINY_ID, "local", "edf", "one", WINDOW_S)
    reads_path = combination.reads_path(tmp_path)
    reads_path.parent.mkdir(parents=True, exist_ok=True)
    worker.run(
        combination,
        roots,
        reads_path,
        combination.open_path(tmp_path),
        max_repeats,
        read_budget_s,
        min_reads,
    )
    (opened,) = read_rows(combination.open_path(tmp_path))
    return read_rows(reads_path), opened


def run_combination(
    tmp_path: Path,
    roots: StoreRoots,
    store: str,
    file_format: str,
    read_kind: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    combination = Combination(TINY_ID, store, file_format, read_kind, WINDOW_S)
    reads_path = combination.reads_path(tmp_path)
    open_path = combination.open_path(tmp_path)
    reads_path.parent.mkdir(parents=True, exist_ok=True)
    worker.run(combination, roots, reads_path, open_path, REPEATS)
    return read_rows(reads_path), read_rows(open_path)


def _first_channel(channel: int | range | None) -> int:
    if channel is None:
        return worker.ALL_CHANNELS
    return channel.start if isinstance(channel, range) else channel


@pytest.mark.usefixtures("tiny_recording")
def test_the_worker_writes_one_row_per_read_and_one_open_row(
    store: str, roots: StoreRoots, tmp_path: Path
) -> None:
    planned = positions(TINY_SPEC, WINDOW_S, REPEATS)
    for file_format in schemas.FORMAT_ROWS:
        for read_kind in TINY_KINDS:
            if file_format == "overhead" and read_kind in schemas.SUBSET_KINDS:
                continue
            reads, opened = run_combination(
                tmp_path, roots, store, file_format, read_kind
            )
            label = f"{file_format} {read_kind}"
            assert [set(row) for row in reads] == [
                set(schemas.FORMAT_READS)
            ] * REPEATS
            assert len(opened) == 1, label
            assert set(opened[0]) == set(schemas.FORMAT_OPENS), label
            assert [int(row["repeat"]) for row in reads] == [0, 1, 2], label
            assert [int(row["start_us"]) for row in reads] == [
                read.start_us for read in planned
            ], label
            expected_channels = [
                _first_channel(selection(TINY_SPEC, read_kind, read))
                for read in planned
            ]
            assert [int(row["channel"]) for row in reads] == expected_channels
            for row in reads:
                assert row["recording"] == TINY_ID
                assert row["store"] == store
                assert row["format"] == file_format
                assert row["read_kind"] == read_kind
                assert row["window_s"] == str(WINDOW_S)
                assert float(row["seconds"]) > 0, label
            if store != "local":
                if file_format != "overhead":
                    assert int(opened[0]["requests"]) > 0, label
                    assert int(opened[0]["bytes"]) > 0, label
                for row in reads:
                    assert int(row["requests"]) > 0, label
                    assert int(row["bytes"]) > 0, label


@pytest.mark.usefixtures("tiny_recording")
@pytest.mark.parametrize("read_kind", ["one", "channels4", "all"])
def test_samples_count_every_sample_over_every_channel(
    tmp_path: Path, tiny_root: Path, read_kind: str
) -> None:
    roots = StoreRoots(local=tiny_root, http="", s3="")
    rows_per_window = WINDOW_S * TINY_SPEC.rate_hz
    channels = {
        "one": 1,
        "channels4": 4,
        "all": TINY_SPEC.channels,
    }[read_kind]
    for file_format in schemas.FORMAT_ROWS:
        reads, _ = run_combination(
            tmp_path, roots, "local", file_format, read_kind
        )
        samples = [int(row["samples"]) for row in reads]
        if file_format == "overhead":
            assert samples == [0] * REPEATS
        elif file_format == "tszarr":
            assert all(
                rows_per_window * channels
                <= count
                <= (rows_per_window + 1) * channels
                for count in samples
            )
        else:
            assert samples == [rows_per_window * channels] * REPEATS, (
                file_format
            )


@pytest.mark.usefixtures("tiny_recording")
def test_a_subset_read_records_the_first_channel_of_its_block(
    tmp_path: Path, tiny_root: Path
) -> None:
    roots = StoreRoots(local=tiny_root, http="", s3="")
    planned = positions(TINY_SPEC, WINDOW_S, REPEATS)
    blocks = [selection(TINY_SPEC, "channels4", read) for read in planned]
    assert all(isinstance(block, range) for block in blocks)
    for file_format in ("edf", "nwb-zarr-channel", "tszarr", "zarr-direct"):
        reads, _ = run_combination(
            tmp_path, roots, "local", file_format, "channels4"
        )
        assert [int(row["channel"]) for row in reads] == [
            block.start for block in blocks if isinstance(block, range)
        ]
        assert {row["read_kind"] for row in reads} == {"channels4"}


@pytest.mark.usefixtures("tiny_recording")
def test_every_format_reads_the_same_windows(
    tmp_path: Path, tiny_root: Path
) -> None:
    roots = StoreRoots(local=tiny_root, http="", s3="")
    starts = {
        file_format: [
            row["start_us"]
            for row in run_combination(
                tmp_path, roots, "local", file_format, "one"
            )[0]
        ]
        for file_format in schemas.FORMAT_ROWS
    }
    assert len({tuple(value) for value in starts.values()}) == 1


@pytest.mark.usefixtures("tiny_recording")
def test_a_read_of_the_overhead_object_is_one_request_of_one_byte(
    store: str, roots: StoreRoots, tmp_path: Path
) -> None:
    reads, opened = run_combination(tmp_path, roots, store, "overhead", "one")
    assert {(row["requests"], row["bytes"]) for row in reads} == {("1", "1")}
    assert (opened[0]["requests"], opened[0]["bytes"]) == ("0", "0")


@pytest.mark.usefixtures("tiny_recording")
def test_a_finished_run_leaves_no_partial_file(
    tmp_path: Path, tiny_root: Path
) -> None:
    roots = StoreRoots(local=tiny_root, http="", s3="")
    run_combination(tmp_path, roots, "local", "edf", "one")
    directory = tmp_path / "format-parts" / TINY_ID
    assert sorted(path.name for path in directory.iterdir()) == [
        "local-edf-one-10s-open.csv",
        "local-edf-one-10s.csv",
    ]


@pytest.mark.usefixtures("tiny_recording")
def test_a_failed_read_leaves_no_csv(
    tmp_path: Path, tiny_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = StoreRoots(local=tiny_root, http="", s3="")
    calls: list[int] = []
    original = worker.make_reader

    def failing_reader(*arguments: object) -> object:
        reader = original(*arguments)  # type: ignore[arg-type]
        read = reader.read

        def failing_read(*args: object) -> np.ndarray:  # type: ignore[type-arg]
            calls.append(1)
            if len(calls) == 2:
                raise OSError("store unreachable")
            return read(*args)  # type: ignore[arg-type]

        reader.read = failing_read  # type: ignore[method-assign]
        return reader

    monkeypatch.setattr(worker, "make_reader", failing_reader)
    with pytest.raises(OSError, match="store unreachable"):
        run_combination(tmp_path, roots, "local", "edf", "one")
    directory = tmp_path / "format-parts" / TINY_ID
    assert list(directory.iterdir()) == []


@pytest.mark.usefixtures("tiny_recording")
def test_the_command_line_runs_one_combination(
    tmp_path: Path, tiny_root: Path
) -> None:
    reads = tmp_path / "reads.csv"
    opened = tmp_path / "open.csv"
    status = worker.main(
        [
            "--recording", TINY_ID,
            "--store", "local",
            "--format", "zarr-direct",
            "--read-kind", "all",
            "--window-s", "1",
            "--output", str(reads),
            "--open-output", str(opened),
            "--max-repeats", "2",
            "--local-root", str(tiny_root),
        ]
    )  # fmt: skip
    assert status == 0
    assert len(read_rows(reads)) == 2
    assert len(read_rows(opened)) == 1


@pytest.mark.usefixtures("tiny_recording")
def test_the_command_line_rejects_a_window_longer_than_the_recording(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        worker.main(
            [
                "--recording", TINY_ID,
                "--store", "local",
                "--format", "edf",
                "--read-kind", "one",
                "--window-s", "60",
                "--output", str(tmp_path / "a.csv"),
                "--open-output", str(tmp_path / "b.csv"),
            ]
        )  # fmt: skip
    assert exit_info.value.code == 2
    assert "longer than recording" in capsys.readouterr().err


@pytest.mark.usefixtures("tiny_recording")
def test_the_worker_stops_after_the_read_that_reaches_the_budget(
    tmp_path: Path, tiny_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads, opened = run_timed(tmp_path, tiny_root, monkeypatch, 1.0, 4.5, 1)
    assert [int(row["repeat"]) for row in reads] == [0, 1, 2, 3, 4]
    assert sum(float(row["seconds"]) for row in reads) == 5.0
    assert (opened["reads"], opened["planned_reads"]) == ("5", str(PLANNED))


@pytest.mark.usefixtures("tiny_recording")
def test_the_worker_stops_when_the_seconds_equal_the_budget(
    tmp_path: Path, tiny_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads, _ = run_timed(tmp_path, tiny_root, monkeypatch, 1.0, 4.0, 1)
    assert len(reads) == 4


@pytest.mark.usefixtures("tiny_recording")
def test_the_worker_makes_every_planned_read_below_the_budget(
    tmp_path: Path, tiny_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads, opened = run_timed(tmp_path, tiny_root, monkeypatch, 1.0, 100.0, 1)
    assert len(reads) == PLANNED
    assert (opened["reads"], opened["planned_reads"]) == (str(PLANNED),) * 2


@pytest.mark.usefixtures("tiny_recording")
def test_the_worker_makes_the_minimum_when_one_read_passes_the_budget(
    tmp_path: Path, tiny_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads, opened = run_timed(tmp_path, tiny_root, monkeypatch, 10.0, 1.0, 3)
    assert len(reads) == 3
    assert opened["reads"] == "3"


@pytest.mark.usefixtures("tiny_recording")
def test_the_worker_makes_the_planned_count_when_it_is_below_the_minimum(
    tmp_path: Path, tiny_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads, opened = run_timed(
        tmp_path, tiny_root, monkeypatch, 10.0, 1.0, 5, max_repeats=2
    )
    assert len(reads) == 2
    assert (opened["reads"], opened["planned_reads"]) == ("2", "2")


@pytest.mark.usefixtures("tiny_recording")
def test_a_combination_that_stops_early_holds_a_prefix_of_the_planned_reads(
    tmp_path: Path, tiny_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads, _ = run_timed(tmp_path, tiny_root, monkeypatch, 1.0, 3.0, 1)
    planned = positions(TINY_SPEC, WINDOW_S, PLANNED)
    assert [int(row["start_us"]) for row in reads] == [
        read.start_us for read in planned[: len(reads)]
    ]


@pytest.mark.usefixtures("tiny_recording")
def test_both_part_headers_record_the_budget_and_the_minimum(
    tmp_path: Path, tiny_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_timed(tmp_path, tiny_root, monkeypatch, 1.0, 2.5, 2)
    combination = Combination(TINY_ID, "local", "edf", "one", WINDOW_S)
    for path in (
        combination.reads_path(tmp_path),
        combination.open_path(tmp_path),
    ):
        header = read_header(path)
        assert float(header["read_budget_s"]) == 2.5
        assert header["min_reads"] == "2"


@pytest.mark.usefixtures("tiny_recording")
def test_the_open_row_of_a_real_run_counts_its_reads(
    tmp_path: Path, tiny_root: Path
) -> None:
    roots = StoreRoots(local=tiny_root, http="", s3="")
    reads, opened = run_combination(tmp_path, roots, "local", "edf", "one")
    assert len(reads) == REPEATS
    assert (opened[0]["reads"], opened[0]["planned_reads"]) == (
        str(REPEATS),
        str(REPEATS),
    )


@pytest.mark.parametrize(
    ("budget", "minimum"), [(0, 1), (-1, 1), (1, 0), (1, -2)]
)
def test_run_rejects_a_budget_or_a_minimum_out_of_range(
    tmp_path: Path, tiny_root: Path, budget: float, minimum: int
) -> None:
    combination = Combination("R1", "local", "edf", "one", 1)
    with pytest.raises(ValueError, match="must be"):
        worker.run(
            combination,
            StoreRoots(local=tiny_root, http="", s3=""),
            tmp_path / "a.csv",
            tmp_path / "b.csv",
            1,
            budget,
            minimum,
        )


def worker_arguments(tmp_path: Path, tiny_root: Path) -> list[str]:
    return [
        "--recording", TINY_ID,
        "--store", "local",
        "--format", "zarr-direct",
        "--read-kind", "all",
        "--window-s", "1",
        "--output", str(tmp_path / "reads.csv"),
        "--open-output", str(tmp_path / "open.csv"),
        "--max-repeats", "4",
        "--local-root", str(tiny_root),
    ]  # fmt: skip


@pytest.mark.usefixtures("tiny_recording")
def test_the_command_line_takes_the_budget_and_the_minimum(
    tmp_path: Path, tiny_root: Path
) -> None:
    status = worker.main(
        [
            *worker_arguments(tmp_path, tiny_root),
            "--read-budget-s", "0.000001",
            "--min-reads", "2",
        ]
    )  # fmt: skip
    assert status == 0
    assert len(read_rows(tmp_path / "reads.csv")) == 2
    (opened,) = read_rows(tmp_path / "open.csv")
    assert (opened["reads"], opened["planned_reads"]) == ("2", "4")
    header = read_header(tmp_path / "reads.csv")
    assert (float(header["read_budget_s"]), header["min_reads"]) == (1e-6, "2")


@pytest.mark.usefixtures("tiny_recording")
@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--read-budget-s", "0"),
        ("--read-budget-s", "-3"),
        ("--min-reads", "0"),
    ],
)
def test_the_command_line_rejects_a_budget_or_a_minimum_out_of_range(
    tmp_path: Path,
    tiny_root: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        worker.main([*worker_arguments(tmp_path, tiny_root), option, value])
    assert exit_info.value.code == 2
    assert option in capsys.readouterr().err
    assert not (tmp_path / "reads.csv").exists()
