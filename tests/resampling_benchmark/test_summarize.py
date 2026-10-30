import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from resampling_benchmark.load.generator import REQUEST_TIMEOUT_S
from resampling_benchmark.load.orchestrate import LINK_BYTES_PER_S
from resampling_benchmark.summarize import (
    FAILED_REQUEST_SECONDS_KEY,
    deployment_metadata,
    find_runs,
    summarize,
    summarize_run,
)
from tszbench.csvmeta import CsvWriter, read_header
from tszbench.recordings import RECORDINGS, RecordingSpec
from tszbench.schemas import (
    LINK_LIMITED_FILE,
    MONITOR,
    RESAMPLING_REQUESTS,
    RESAMPLING_REQUESTS_FILE,
    RESAMPLING_RUNS,
    RESAMPLING_RUNS_FILE,
    SERVER_METADATA_FILE,
)

SPEC = replace(RECORDINGS["R3"], id="S1", channels=3, rate_hz=100)


def lookup(name: str) -> RecordingSpec:
    assert name == "S1"
    return SPEC


def request_row(
    sequence: int,
    *,
    user: int = 0,
    generator: int = 0,
    seconds: float | str = 1.0,
    dispatch: float | str = 0.0,
    status: str = "ok",
    warmup: bool = False,
    size: int = 1000,
    users: int = 2,
) -> dict[str, object]:
    return {
        "run": "server-S1-10-2",
        "method": "server",
        "recording": "S1",
        "window_s": "10",
        "users": users,
        "generator": generator,
        "user": user,
        "sequence": sequence,
        "start_us": 0,
        "dispatch_s": dispatch,
        "seconds": seconds,
        "bytes": size,
        "s3_requests": 3,
        "s3_bytes": 30,
        "status": status,
        "warmup": "true" if warmup else "false",
        "level": 0,
        "read_seconds": 0.5,
        "reduce_seconds": 0.25,
    }


def write_requests(
    path: Path,
    rows: list[dict[str, object]],
    repeated: bool = False,
    **extra: str,
) -> None:
    header = {
        "run": "server-S1-10-2",
        "repeated": str(repeated).lower(),
        **extra,
    }
    with CsvWriter(path, RESAMPLING_REQUESTS, header) as writer:
        writer.write_all(rows)


def write_monitor(
    path: Path, role: str, values: list[tuple[float, float]]
) -> None:
    with CsvWriter(path, MONITOR, {}) as writer:
        for index, (cpu, rx) in enumerate(values):
            writer.write(
                {
                    "run": "server-S1-10-2",
                    "host": "h",
                    "role": role,
                    "time_s": float(index),
                    "cpu_percent": cpu,
                    "rx_bytes_per_s": rx,
                    "tx_bytes_per_s": 0.0,
                }
            )


@pytest.fixture
def run_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "raw" / "server-S1-10-2"
    directory.mkdir(parents=True)
    seconds = [1.0, 2.0, 3.0, 4.0, 5.0]
    first = [request_row(0, warmup=True, seconds=99.0)]
    first += [
        request_row(index + 1, seconds=value, dispatch=float(index))
        for index, value in enumerate(seconds)
    ]
    first.append(request_row(6, seconds=180.5, dispatch=5.0, status="timeout"))
    second = [
        request_row(0, user=1, generator=1, warmup=True),
        request_row(
            1, user=0, generator=1, seconds=1.0, dispatch=0.5, size=3000
        ),
        request_row(
            2,
            user=0,
            generator=1,
            seconds="",
            dispatch=1.5,
            status="error:ConnectError",
        ),
    ]
    write_requests(directory / "requests-0.csv", first)
    write_requests(directory / "requests-1.csv", second)
    write_monitor(
        directory / "monitor-server.csv", "server", [(10.0, 0), (80.0, 5)]
    )
    write_monitor(
        directory / "monitor-generator-0.csv",
        "generator",
        [(5.0, 0.25 * LINK_BYTES_PER_S), (6.0, 0.5 * LINK_BYTES_PER_S)],
    )
    write_monitor(
        directory / "monitor-generator-1.csv",
        "generator",
        [(5.0, 0.125 * LINK_BYTES_PER_S)],
    )
    return directory.parent


def test_find_runs_lists_the_files_of_each_run(run_directory: Path) -> None:
    (run,) = find_runs(run_directory)
    assert run.run == "server-S1-10-2"
    assert [path.name for path in run.requests] == [
        "requests-0.csv",
        "requests-1.csv",
    ]
    assert len(run.generator_monitors) == 2
    assert len(run.server_monitors) == 1


def test_the_run_row_leaves_out_warmup_and_counts_failures(
    run_directory: Path,
) -> None:
    row = summarize_run(find_runs(run_directory)[0], lookup)
    assert set(row) == set(RESAMPLING_RUNS)
    assert row["requests"] == 8
    assert row["failures"] == 2
    assert row["failure_rate"] == pytest.approx(0.25)
    assert row["users"] == 2
    assert row["generators"] == 2
    assert row["users_per_generator"] == 2
    assert row["repeated"] == "false"


def test_the_run_row_charges_failed_requests_at_least_the_timeout(
    run_directory: Path,
) -> None:
    row = summarize_run(find_runs(run_directory)[0], lookup)
    seconds = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 180.5, REQUEST_TIMEOUT_S]
    assert row["median_s"] == pytest.approx(np.percentile(seconds, 50))
    assert row["p5_s"] == pytest.approx(np.percentile(seconds, 5))
    assert row["p95_s"] == pytest.approx(np.percentile(seconds, 95))


def test_median_bytes_cover_successful_requests_only(
    run_directory: Path,
) -> None:
    row = summarize_run(find_runs(run_directory)[0], lookup)
    assert row["median_bytes"] == pytest.approx(1000.0)


def test_thirty_percent_timeouts_move_the_median_and_p95(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "raw" / "server-S1-10-2"
    healthy = [request_row(index, seconds=1.0) for index in range(7)]
    timeouts = [
        request_row(index, seconds=REQUEST_TIMEOUT_S + 1.0, status="timeout")
        for index in range(7, 10)
    ]
    write_requests(directory / "requests-0.csv", healthy + timeouts)
    row = summarize_run(find_runs(directory.parent)[0], lookup)
    seconds = [1.0] * 7 + [REQUEST_TIMEOUT_S + 1.0] * 3
    assert row["failure_rate"] == pytest.approx(0.3)
    assert row["median_s"] == pytest.approx(1.0)
    assert row["p95_s"] == pytest.approx(np.percentile(seconds, 95))
    assert float(str(row["p95_s"])) > REQUEST_TIMEOUT_S


def test_an_error_after_half_a_second_counts_as_the_timeout(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "raw" / "server-S1-10-2"
    write_requests(
        directory / "requests-0.csv",
        [
            request_row(0, seconds=1.0),
            request_row(1, seconds=0.5, status="error:503 SlowDown"),
        ],
    )
    row = summarize_run(find_runs(directory.parent)[0], lookup)
    assert row["p95_s"] == pytest.approx(
        np.percentile([1.0, REQUEST_TIMEOUT_S], 95)
    )
    assert row["failure_rate"] == pytest.approx(0.5)


def test_a_run_without_failures_keeps_the_seconds_of_its_requests(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "raw" / "server-S1-10-2"
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    write_requests(
        directory / "requests-0.csv",
        [
            request_row(index, seconds=value)
            for index, value in enumerate(values)
        ],
    )
    row = summarize_run(find_runs(directory.parent)[0], lookup)
    assert row["failure_rate"] == 0.0
    assert row["median_s"] == pytest.approx(3.0)
    assert row["p5_s"] == pytest.approx(np.percentile(values, 5))
    assert row["p95_s"] == pytest.approx(np.percentile(values, 95))


def test_requests_per_second_divides_successes_by_the_measured_span(
    run_directory: Path,
) -> None:
    row = summarize_run(find_runs(run_directory)[0], lookup)
    last_completion = 5.0 + 180.5
    assert row["requests_per_second"] == pytest.approx(6 / last_completion)


def test_raw_bytes_hold_every_channel_as_float32(run_directory: Path) -> None:
    row = summarize_run(find_runs(run_directory)[0], lookup)
    assert row["raw_bytes"] == 3 * 10 * 100 * 4


def test_monitor_peaks_come_from_the_monitor_files(
    run_directory: Path,
) -> None:
    row = summarize_run(find_runs(run_directory)[0], lookup)
    assert row["server_cpu_max"] == pytest.approx(80.0)
    assert row["generator_link_max"] == pytest.approx(0.5)


def test_monitor_columns_are_empty_without_monitor_files(
    run_directory: Path,
) -> None:
    for path in run_directory.rglob("monitor-*.csv"):
        path.unlink()
    row = summarize_run(find_runs(run_directory)[0], lookup)
    assert row["server_cpu_max"] == ""
    assert row["generator_link_max"] == ""


def test_a_repeated_run_is_marked_from_the_request_header(
    run_directory: Path,
) -> None:
    path = run_directory / "server-S1-10-2" / "requests-0.csv"
    write_requests(path, [request_row(1)], repeated=True)
    row = summarize_run(find_runs(run_directory)[0], lookup)
    assert row["repeated"] == "true"


def test_a_run_without_a_success_has_the_timeout_as_its_median(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "raw" / "server-S1-10-2"
    write_requests(
        directory / "requests-0.csv",
        [
            request_row(0, warmup=True),
            request_row(1, status="timeout", seconds=0.5),
            request_row(2, status="error:ConnectError", seconds=""),
        ],
    )
    row = summarize_run(find_runs(directory.parent)[0], lookup)
    assert row["failure_rate"] == 1.0
    assert row["median_s"] == REQUEST_TIMEOUT_S
    assert row["p5_s"] == REQUEST_TIMEOUT_S
    assert row["p95_s"] == REQUEST_TIMEOUT_S
    assert row["median_bytes"] == ""
    assert row["requests_per_second"] == ""


def test_summarize_writes_both_csvs_with_every_request(
    run_directory: Path, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    requests, runs = summarize(run_directory, results, lookup)
    assert (requests, runs) == (10, 1)
    for name, columns in (
        (RESAMPLING_REQUESTS_FILE, RESAMPLING_REQUESTS),
        (RESAMPLING_RUNS_FILE, RESAMPLING_RUNS),
    ):
        lines = [
            line
            for line in (results / name).read_text().splitlines()
            if not line.startswith("#")
        ]
        assert lines[0].split(",") == list(columns)
        assert "commit" in read_header(results / name)


def test_both_csvs_record_the_failed_request_seconds_in_the_header(
    run_directory: Path, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    summarize(run_directory, results, lookup)
    assert FAILED_REQUEST_SECONDS_KEY == "failed_request_seconds"
    for name in (RESAMPLING_REQUESTS_FILE, RESAMPLING_RUNS_FILE):
        header = read_header(results / name)
        assert float(header[FAILED_REQUEST_SECONDS_KEY]) == REQUEST_TIMEOUT_S


def test_a_run_is_link_limited_only_when_it_has_the_marker_file(
    run_directory: Path,
) -> None:
    assert (
        summarize_run(find_runs(run_directory)[0], lookup)["link_limited"]
        == "false"
    )
    (run_directory / "server-S1-10-2" / LINK_LIMITED_FILE).touch()
    assert (
        summarize_run(find_runs(run_directory)[0], lookup)["link_limited"]
        == "true"
    )


def test_a_high_link_peak_alone_does_not_mark_a_run_link_limited(
    run_directory: Path,
) -> None:
    row = summarize_run(find_runs(run_directory)[0], lookup)
    assert row["generator_link_max"] == pytest.approx(0.5)
    assert row["link_limited"] == "false"


def test_the_header_copies_the_server_metadata_and_the_generator_fields(
    run_directory: Path, tmp_path: Path
) -> None:
    directory = run_directory / "server-S1-10-2"
    (directory / SERVER_METADATA_FILE).write_text(
        json.dumps({"instance": "c6i.8xlarge", "ami": "ami-9"})
    )
    write_requests(
        directory / "requests-0.csv",
        [request_row(1)],
        instance="c6i.8xlarge",
        ami="ami-1",
        date="2026-10-24T10:00:00+00:00",
    )
    write_requests(
        directory / "requests-1.csv",
        [request_row(1, generator=1)],
        instance="c6i.8xlarge",
        ami="ami-1",
        date="2026-10-23T09:00:00+00:00",
    )
    results = tmp_path / "results"
    summarize(run_directory, results, lookup)
    for name in (RESAMPLING_REQUESTS_FILE, RESAMPLING_RUNS_FILE):
        header = read_header(results / name)
        assert header["server.instance"] == "c6i.8xlarge"
        assert header["server.ami"] == "ami-9"
        assert header["generator.instance"] == "c6i.8xlarge"
        assert header["generator.ami"] == "ami-1"
        assert header["first_run_date"] == "2026-10-23T09:00:00+00:00"
        assert header["last_run_date"] == "2026-10-24T10:00:00+00:00"


def test_the_deployment_metadata_lists_distinct_generator_instances(
    run_directory: Path,
) -> None:
    directory = run_directory / "server-S1-10-2"
    write_requests(directory / "requests-0.csv", [request_row(1)], instance="b")
    write_requests(directory / "requests-1.csv", [request_row(1)], instance="a")
    fields = deployment_metadata(find_runs(run_directory))
    assert fields["generator.instance"] == "a,b"


def test_the_deployment_metadata_leaves_out_what_the_run_lacks(
    run_directory: Path,
) -> None:
    assert deployment_metadata(find_runs(run_directory)) == {}
