import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

import format_benchmark.__main__ as format_benchmark_main
import format_benchmark.worker as worker
from format_benchmark.combinations import Combination
from format_benchmark.positions import MIN_READS, READ_BUDGET_S, positions
from generate.formats import write_format, write_one_byte_object
from tests.format_benchmark.conftest import TINY_ID, TINY_SPEC
from tszbench import schemas
from tszbench.csvmeta import read_header, read_rows
from tszbench.fs import StoreRoots
from tszbench.recordings import FORMAT_RECORDINGS, RECORDINGS, RecordingSpec

REPEATS = 2
HOUR_SPEC = RECORDINGS["R1"].cut(3600, "1h")

WorkerRunner = Callable[
    [Combination, StoreRoots, Path, int | None, float, int], int
]


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[Combination]:
    """Run each worker in this process, and list the combinations it ran."""
    ran: list[Combination] = []

    def run_in_process(
        combination: Combination,
        roots: StoreRoots,
        results: Path,
        max_repeats: int | None,
        read_budget_s: float,
        min_reads: int,
    ) -> int:
        ran.append(combination)
        reads = combination.reads_path(results)
        reads.parent.mkdir(parents=True, exist_ok=True)
        worker.run(
            combination,
            roots,
            reads,
            combination.open_path(results),
            max_repeats,
            read_budget_s,
            min_reads,
        )
        return 0

    monkeypatch.setattr(format_benchmark_main, "run_worker", run_in_process)
    return ran


def command(tiny_root: Path, results: Path, *extra: str) -> list[str]:
    return [
        "--recordings", TINY_ID,
        "--stores", "local",
        "--formats", "edf,tszarr,overhead",
        "--windows", "1,10",
        "--max-repeats", str(REPEATS),
        "--results", str(results),
        "--local-root", str(tiny_root),
        *extra,
    ]  # fmt: skip


@pytest.mark.usefixtures("tiny_recording")
def test_main_runs_every_combination_and_merges_the_csvs(
    tmp_path: Path, tiny_root: Path, started: list[Combination]
) -> None:
    assert format_benchmark_main.main(command(tiny_root, tmp_path)) == 0
    assert [(c.file_format, c.read_kind, c.window_s) for c in started] == [
        ("edf", "one", 1),
        ("edf", "one", 10),
        ("edf", "channels4", 10),
        ("edf", "all", 1),
        ("edf", "all", 10),
        ("tszarr", "one", 1),
        ("tszarr", "one", 10),
        ("tszarr", "channels4", 10),
        ("tszarr", "all", 1),
        ("tszarr", "all", 10),
        ("overhead", "one", 1),
    ]
    reads = read_rows(tmp_path / "format-reads.csv")
    opens = read_rows(tmp_path / "format-opens.csv")
    assert len(reads) == 11 * REPEATS
    assert len(opens) == 11
    assert set(reads[0]) == set(schemas.FORMAT_READS)
    assert set(opens[0]) == set(schemas.FORMAT_OPENS)
    assert [row["repeat"] for row in reads[:REPEATS]] == ["0", "1"]


@pytest.mark.usefixtures("tiny_recording")
def test_main_runs_a_subset_read_kind_at_ten_seconds_only(
    tmp_path: Path, tiny_root: Path, started: list[Combination]
) -> None:
    arguments = command(
        tiny_root, tmp_path, "--read-kinds", "channels4,channels16"
    )
    assert format_benchmark_main.main(arguments) == 0
    assert [(c.file_format, c.read_kind, c.window_s) for c in started] == [
        ("edf", "channels4", 10),
        ("tszarr", "channels4", 10),
    ]
    reads = read_rows(tmp_path / "format-reads.csv")
    assert {row["read_kind"] for row in reads} == {"channels4"}


def test_main_rejects_an_unknown_read_kind(
    tmp_path: Path, tiny_root: Path
) -> None:
    with pytest.raises(SystemExit):
        format_benchmark_main.main(
            command(tiny_root, tmp_path, "--read-kinds", "channels8")
        )


@pytest.mark.usefixtures("tiny_recording")
def test_main_records_the_roots_and_the_run_in_both_headers(
    tmp_path: Path, tiny_root: Path, started: list[Combination]
) -> None:
    format_benchmark_main.main(
        command(
            tiny_root,
            tmp_path,
            "--drop-caches", "true",
            "--drop-caches", "exit 0",
        )
    )  # fmt: skip
    for name in ("format-reads.csv", "format-opens.csv"):
        header = read_header(tmp_path / name)
        assert header["recordings"] == TINY_ID
        assert header["local_root"] == str(tiny_root)
        assert header["http_root"] == "none"
        assert header["http_server"] == "none"
        assert header["nginx_build"] == "none"
        assert header["s3_root"] == "none"
        assert header["max_repeats"] == str(REPEATS)
        assert header["drop_caches"] == "true; exit 0"
        assert header[f"seed.{TINY_ID}"] == str(TINY_SPEC.seed)
        assert header["version.timeseries-zarr"] != "absent"
        assert "date" in header
        assert "commit" in header


EARLY = ("--read-budget-s", "0.000000001", "--min-reads", "1")
"""Options under which every combination stops after its first read."""


@pytest.mark.usefixtures("tiny_recording")
def test_main_records_the_budget_and_the_minimum_in_both_headers(
    tmp_path: Path, tiny_root: Path, started: list[Combination]
) -> None:
    format_benchmark_main.main(
        command(
            tiny_root, tmp_path, "--read-budget-s", "2.5", "--min-reads", "2"
        )
    )
    for name in ("format-reads.csv", "format-opens.csv"):
        header = read_header(tmp_path / name)
        assert float(header["read_budget_s"]) == 2.5
        assert header["min_reads"] == "2"


@pytest.mark.usefixtures("tiny_recording")
def test_main_records_the_default_budget_and_minimum(
    tmp_path: Path, tiny_root: Path, started: list[Combination]
) -> None:
    format_benchmark_main.main(command(tiny_root, tmp_path))
    header = read_header(tmp_path / "format-reads.csv")
    assert float(header["read_budget_s"]) == READ_BUDGET_S
    assert header["min_reads"] == str(MIN_READS)


@pytest.mark.usefixtures("tiny_recording")
def test_main_skips_a_combination_that_stopped_early_under_the_same_options(
    tmp_path: Path, tiny_root: Path, started: list[Combination]
) -> None:
    format_benchmark_main.main(command(tiny_root, tmp_path, *EARLY))
    opens = read_rows(tmp_path / "format-opens.csv")
    assert {row["reads"] for row in opens} == {"1"}
    assert {row["planned_reads"] for row in opens} == {str(REPEATS)}
    started.clear()
    format_benchmark_main.main(command(tiny_root, tmp_path, *EARLY))
    assert started == []
    assert len(read_rows(tmp_path / "format-reads.csv")) == len(opens)


@pytest.mark.usefixtures("tiny_recording")
def test_main_reruns_the_combinations_that_stopped_early_under_a_larger_budget(
    tmp_path: Path, tiny_root: Path, started: list[Combination]
) -> None:
    format_benchmark_main.main(command(tiny_root, tmp_path, *EARLY))
    count = len(started)
    started.clear()
    format_benchmark_main.main(command(tiny_root, tmp_path, "--min-reads", "1"))
    assert len(started) == count
    assert len(read_rows(tmp_path / "format-reads.csv")) == count * REPEATS


@pytest.mark.usefixtures("tiny_recording")
def test_main_keeps_a_combination_that_made_every_read_under_new_options(
    tmp_path: Path, tiny_root: Path, started: list[Combination]
) -> None:
    format_benchmark_main.main(command(tiny_root, tmp_path))
    started.clear()
    format_benchmark_main.main(command(tiny_root, tmp_path, *EARLY))
    assert started == []


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--read-budget-s", "0"),
        ("--read-budget-s", "-1"),
        ("--min-reads", "0"),
    ],
)
def test_main_rejects_a_budget_or_a_minimum_out_of_range(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        format_benchmark_main.main(
            ["--local-root", str(tmp_path), option, value]
        )
    assert exit_info.value.code == 2
    assert option in capsys.readouterr().err


@pytest.mark.usefixtures("tiny_recording")
def test_main_skips_the_combinations_whose_csvs_are_whole(
    tmp_path: Path, tiny_root: Path, started: list[Combination]
) -> None:
    format_benchmark_main.main(command(tiny_root, tmp_path))
    first = len(started)
    started.clear()
    format_benchmark_main.main(command(tiny_root, tmp_path))
    assert started == []
    assert len(read_rows(tmp_path / "format-reads.csv")) == first * REPEATS


@pytest.mark.usefixtures("tiny_recording")
def test_main_reruns_only_the_combination_that_lost_a_file(
    tmp_path: Path, tiny_root: Path, started: list[Combination]
) -> None:
    format_benchmark_main.main(command(tiny_root, tmp_path))
    lost = Combination(TINY_ID, "local", "tszarr", "all", 10)
    lost.open_path(tmp_path).unlink()
    started.clear()
    format_benchmark_main.main(command(tiny_root, tmp_path))
    assert started == [lost]


@pytest.mark.usefixtures("tiny_recording")
def test_main_reruns_a_combination_run_with_fewer_repeats(
    tmp_path: Path, tiny_root: Path, started: list[Combination]
) -> None:
    arguments = command(tiny_root, tmp_path)
    arguments[arguments.index("--max-repeats") + 1] = "1"
    format_benchmark_main.main(arguments)
    started.clear()
    format_benchmark_main.main(command(tiny_root, tmp_path))
    assert len(started) == 11


@pytest.mark.usefixtures("tiny_recording")
def test_main_keeps_the_rows_of_combinations_outside_the_command(
    tmp_path: Path, tiny_root: Path, started: list[Combination]
) -> None:
    format_benchmark_main.main(command(tiny_root, tmp_path))
    before = read_rows(tmp_path / "format-reads.csv")
    started.clear()
    arguments = command(tiny_root, tmp_path, "--read-kinds", "one")
    assert format_benchmark_main.main(arguments) == 0
    assert started == []
    assert read_rows(tmp_path / "format-reads.csv") == before


def test_main_keeps_the_rows_of_a_recording_that_an_earlier_command_ran(
    tmp_path: Path,
    tiny_root: Path,
    started: list[Combination],
    monkeypatch: pytest.MonkeyPatch,
    tiny_recording: RecordingSpec,
) -> None:
    import format_benchmark.combinations as combinations

    second = replace(tiny_recording, id="R2")

    def resolve_both(name: str) -> RecordingSpec:
        return {"R2": second, TINY_ID: tiny_recording}[name]

    monkeypatch.setattr(combinations, "resolve", resolve_both)
    root = tmp_path / "root"
    root.mkdir()
    for name in ("R2", TINY_ID):
        (root / name).symlink_to(tiny_root / TINY_ID)
    (root / "overhead").symlink_to(tiny_root / "overhead")
    for name in ("R2", TINY_ID):
        arguments = command(root, tmp_path)
        arguments[arguments.index("--recordings") + 1] = name
        assert format_benchmark_main.main(arguments) == 0
    reads = read_rows(tmp_path / "format-reads.csv")
    assert [row["recording"] for row in reads] == ["R2"] * 22 + [TINY_ID] * 22
    assert len(read_rows(tmp_path / "format-opens.csv")) == 22
    header = read_header(tmp_path / "format-reads.csv")
    assert header["recordings"] == f"R2,{TINY_ID}"
    assert header["seed.R2"] == str(second.seed)
    assert header[f"seed.{TINY_ID}"] == str(TINY_SPEC.seed)


@pytest.mark.usefixtures("tiny_recording")
def test_main_runs_each_drop_caches_command_before_each_process(
    tmp_path: Path, tiny_root: Path, started: list[Combination]
) -> None:
    log = tmp_path / "drops.log"
    arguments = command(
        tiny_root,
        tmp_path,
        "--drop-caches", f"echo client >> {log}",
        "--drop-caches", f"echo server >> {log}",
    )  # fmt: skip
    format_benchmark_main.main(arguments)
    assert log.read_text().split() == ["client", "server"] * len(started)
    log.unlink()
    format_benchmark_main.main(arguments)
    assert not log.exists()


@pytest.mark.usefixtures("tiny_recording")
def test_main_stops_when_a_drop_caches_command_fails(
    tmp_path: Path,
    tiny_root: Path,
    started: list[Combination],
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = format_benchmark_main.main(
        command(tiny_root, tmp_path, "--drop-caches", "exit 3")
    )
    assert status == 1
    assert started == []
    assert "status 3" in capsys.readouterr().err


@pytest.mark.usefixtures("tiny_recording")
def test_main_reports_a_failed_worker_and_merges_the_rest(
    tmp_path: Path,
    tiny_root: Path,
    started: list[Combination],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_in_process = format_benchmark_main.run_worker

    def failing(
        combination: Combination,
        roots: StoreRoots,
        results: Path,
        max_repeats: int | None,
        read_budget_s: float,
        min_reads: int,
    ) -> int:
        if combination.file_format == "tszarr":
            return 1
        return run_in_process(
            combination, roots, results, max_repeats, read_budget_s, min_reads
        )

    monkeypatch.setattr(format_benchmark_main, "run_worker", failing)
    assert format_benchmark_main.main(command(tiny_root, tmp_path)) == 1
    formats = {
        row["format"] for row in read_rows(tmp_path / "format-reads.csv")
    }
    assert formats == {"edf", "overhead"}


@pytest.mark.usefixtures("tiny_recording")
def test_main_logs_the_skipped_combinations(
    tmp_path: Path,
    tiny_root: Path,
    started: list[Combination],
    capsys: pytest.CaptureFixture[str],
) -> None:
    format_benchmark_main.main(
        command(tiny_root, tmp_path, "--windows", "1,60")
    )
    error = capsys.readouterr().err
    assert "window of 60 s is longer than recording tiny" in error


def test_main_rejects_an_unknown_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        format_benchmark_main.main(
            ["--formats", "parquet", "--local-root", str(tmp_path)]
        )
    assert exit_info.value.code == 2
    assert "must be one of" in capsys.readouterr().err


def test_recordings_default_to_the_format_recordings() -> None:
    arguments = format_benchmark_main.build_parser().parse_args([])
    assert arguments.recordings.split(",") == list(FORMAT_RECORDINGS)


def test_main_rejects_an_unknown_recording(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        format_benchmark_main.main(
            ["--recordings", "R9", "--local-root", str(tmp_path)]
        )
    assert exit_info.value.code == 2
    assert "recording" in capsys.readouterr().err


def test_main_rejects_a_store_without_a_root(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        format_benchmark_main.main(["--stores", "http"])
    assert exit_info.value.code == 2
    assert "--http-root" in capsys.readouterr().err


def test_main_rejects_a_window_outside_the_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        format_benchmark_main.main(
            ["--windows", "5", "--local-root", str(tmp_path)]
        )
    assert exit_info.value.code == 2
    assert "window" in capsys.readouterr().err


@pytest.mark.network
def test_server_header_names_the_file_server(
    tiny_root: Path, serve_http: Callable[[Path], str]
) -> None:
    assert "BaseHTTP" in format_benchmark_main.server_header(
        serve_http(tiny_root)
    )


def test_server_header_is_unknown_when_nothing_answers() -> None:
    assert (
        format_benchmark_main.server_header("http://127.0.0.1:9/") == "unknown"
    )


@pytest.mark.network
def test_nginx_build_is_the_first_line_of_the_version_file(
    tmp_path: Path, serve_http: Callable[[Path], str]
) -> None:
    (tmp_path / "nginx-version.txt").write_text(
        "nginx version: nginx/1.24.0 (Ubuntu)\nbuilt with OpenSSL 3.0\n"
    )
    assert (
        format_benchmark_main.nginx_build(serve_http(tmp_path))
        == "nginx version: nginx/1.24.0 (Ubuntu)"
    )


@pytest.mark.network
def test_nginx_build_is_unknown_without_the_version_file(
    tmp_path: Path, serve_http: Callable[[Path], str]
) -> None:
    assert format_benchmark_main.nginx_build(serve_http(tmp_path)) == "unknown"


def test_nginx_build_is_unknown_when_nothing_answers() -> None:
    assert format_benchmark_main.nginx_build("http://127.0.0.1:9/") == "unknown"


def test_merge_keeps_the_order_of_the_combinations(tmp_path: Path) -> None:
    combinations = [
        Combination("R1", "local", "edf", "one", 1),
        Combination("R1", "local", "edf", "one", 10),
    ]
    for index, combination in enumerate(combinations):
        combination.reads_path(tmp_path).parent.mkdir(
            parents=True, exist_ok=True
        )
        for path, columns in (
            (combination.reads_path(tmp_path), schemas.FORMAT_READS),
            (combination.open_path(tmp_path), schemas.FORMAT_OPENS),
        ):
            path.write_text(
                "# combination: x\n"
                + ",".join(columns)
                + "\n"
                + ",".join(str(index) for _ in columns)
                + "\n"
            )
    reads, opens = format_benchmark_main.merge(
        tmp_path, list(reversed(combinations)), {"date": "d"}
    )
    assert (reads, opens) == (2, 2)
    assert [
        row["recording"] for row in read_rows(tmp_path / "format-reads.csv")
    ] == [
        "1",
        "0",
    ]
    assert read_header(tmp_path / "format-reads.csv") == {"date": "d"}


def test_merge_leaves_out_a_combination_without_files(tmp_path: Path) -> None:
    missing = Combination("R1", "local", "edf", "one", 1)
    assert format_benchmark_main.merge(tmp_path, [missing], {}) == (0, 0)


def test_worker_command_passes_every_option(tmp_path: Path) -> None:
    roots = StoreRoots(
        local=tmp_path, http="http://h/", s3="b/p", s3_endpoint_url="http://e"
    )
    combination = Combination("R2", "s3", "tszarr", "one", 60)
    line = format_benchmark_main.worker_command(
        combination, roots, tmp_path, 5, 12.5, 4
    )
    assert line[:3] == [sys.executable, "-m", "format_benchmark.worker"]
    for option, value in (
        ("--recording", "R2"),
        ("--store", "s3"),
        ("--format", "tszarr"),
        ("--read-kind", "one"),
        ("--window-s", "60"),
        ("--http-root", "http://h/"),
        ("--s3-root", "b/p"),
        ("--s3-endpoint-url", "http://e"),
        ("--max-repeats", "5"),
        ("--read-budget-s", "12.5"),
        ("--min-reads", "4"),
    ):
        assert line[line.index(option) + 1] == value


@pytest.fixture(scope="session")
def hour_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a store root that holds R1-1h as EDF and TS-Zarr."""
    root = tmp_path_factory.mktemp("format_benchmark-hour")
    for file_format in ("edf", "tszarr"):
        write_format(HOUR_SPEC, file_format, root)
    write_one_byte_object(root)
    return root


def run_command_line(
    results: Path, store_options: list[str], *extra: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, "-m", "format_benchmark",
            "--recordings", "R1-1h",
            "--formats", "edf,tszarr",
            "--read-kinds", "one",
            "--windows", "1",
            "--max-repeats", "3",
            "--results", str(results),
            *store_options,
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )  # fmt: skip


def test_the_command_line_runs_each_combination_in_a_fresh_process(
    tmp_path: Path, hour_root: Path
) -> None:
    marker = tmp_path / "marker"
    done = run_command_line(
        tmp_path / "results",
        ["--stores", "local", "--local-root", str(hour_root)],
        "--drop-caches",
        f"echo dropped >> {marker}",
    )
    assert done.returncode == 0, done.stderr
    assert marker.read_text().split() == ["dropped", "dropped"]
    reads = read_rows(tmp_path / "results" / "format-reads.csv")
    assert [(row["format"], row["repeat"]) for row in reads] == [
        ("edf", "0"),
        ("edf", "1"),
        ("edf", "2"),
        ("tszarr", "0"),
        ("tszarr", "1"),
        ("tszarr", "2"),
    ]
    planned = positions(HOUR_SPEC, 1, 3)
    assert [int(row["start_us"]) for row in reads[:3]] == [
        read.start_us for read in planned
    ]
    assert {int(row["samples"]) for row in reads[:3]} == {HOUR_SPEC.rate_hz}


@pytest.mark.network
def test_the_command_line_reads_the_http_and_s3_stores(
    tmp_path: Path,
    hour_root: Path,
    store_roots: Callable[[Path], StoreRoots],
) -> None:
    roots = store_roots(hour_root)
    done = run_command_line(
        tmp_path / "results",
        [
            "--stores", "http,s3",
            "--http-root", roots.http,
            "--s3-root", roots.s3,
            "--s3-endpoint-url", roots.s3_endpoint_url or "",
        ],
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    reads = read_rows(tmp_path / "results" / "format-reads.csv")
    assert {(row["store"], row["format"]) for row in reads} == {
        (store, file_format)
        for store in ("http", "s3")
        for file_format in ("edf", "tszarr")
    }
    assert all(int(row["requests"]) > 0 for row in reads)
    header = read_header(tmp_path / "results" / "format-reads.csv")
    assert "BaseHTTP" in header["http_server"]
    assert header["nginx_build"] == "unknown"
    assert header["http_root"] == roots.http
    assert header["s3_endpoint_url"] == roots.s3_endpoint_url
