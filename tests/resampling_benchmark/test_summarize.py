import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resampling_benchmark.load.generator import (
    LINK_BYTES_PER_S,
    REQUEST_TIMEOUT_S,
)
from resampling_benchmark.summarize import (
    FAILED_REQUEST_SECONDS_KEY,
    deployment_metadata,
    find_runs,
    measured_span,
    median_charged_seconds,
    summarize,
    summarize_run,
    sustained_mean,
)
from tszbench.csvmeta import CsvWriter, read_header
from tszbench.recordings import RECORDINGS, RecordingSpec
from tszbench.schemas import (
    CLIENT_LIMITED_FILE,
    MONITOR,
    NOT_RUN_FILE,
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
    path: Path,
    role: str,
    values: list[tuple[float, float]],
    first_time: float = 0.0,
    memory: float = 10.0,
    **header: str,
) -> None:
    with CsvWriter(path, MONITOR, header) as writer:
        for index, (cpu, rx) in enumerate(values):
            writer.write(
                {
                    "run": "server-S1-10-2",
                    "host": "h",
                    "role": role,
                    "time_s": first_time + index,
                    "cpu_percent": cpu,
                    "rx_bytes_per_s": rx,
                    "tx_bytes_per_s": 0.0,
                    "memory_percent": memory,
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


def test_a_connection_error_that_returns_at_once_counts_as_the_timeout(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "raw" / "server-S1-10-2"
    write_requests(
        directory / "requests-0.csv",
        [
            request_row(0, seconds=1.0),
            request_row(1, seconds=0.001, status="error:RemoteProtocolError"),
            request_row(2, seconds=0.002, status="error:ReadError"),
        ],
    )
    row = summarize_run(find_runs(directory.parent)[0], lookup)
    seconds = [1.0, REQUEST_TIMEOUT_S, REQUEST_TIMEOUT_S]
    assert row["failures"] == 2
    assert row["median_s"] == pytest.approx(REQUEST_TIMEOUT_S)
    assert row["p95_s"] == pytest.approx(np.percentile(seconds, 95))


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
    assert row["server_memory_max"] == pytest.approx(10.0)


def test_the_sustained_columns_are_empty_without_a_start_time(
    run_directory: Path,
) -> None:
    row = summarize_run(find_runs(run_directory)[0], lookup)
    for name in (
        "server_cpu_sustained",
        "generator_cpu_sustained",
        "generator_link_sustained",
        "generator_cpu_s_per_request",
    ):
        assert row[name] == ""


def test_monitor_columns_are_empty_without_monitor_files(
    run_directory: Path,
) -> None:
    for path in run_directory.rglob("monitor-*.csv"):
        path.unlink()
    row = summarize_run(find_runs(run_directory)[0], lookup)
    assert row["server_cpu_max"] == ""
    assert row["server_memory_max"] == ""
    assert row["generator_link_sustained"] == ""


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


def test_a_run_is_client_limited_only_when_it_has_the_marker_file(
    run_directory: Path,
) -> None:
    row = summarize_run(find_runs(run_directory)[0], lookup)
    assert (row["client_limited"], row["client_limit"]) == ("false", "")
    marker = run_directory / "server-S1-10-2" / CLIENT_LIMITED_FILE
    marker.write_text("cpu\n")
    row = summarize_run(find_runs(run_directory)[0], lookup)
    assert (row["client_limited"], row["client_limit"]) == ("true", "cpu")
    marker.write_text("link\n")
    row = summarize_run(find_runs(run_directory)[0], lookup)
    assert (row["client_limited"], row["client_limit"]) == ("true", "link")


def test_an_empty_client_limited_file_marks_the_run_without_a_resource(
    run_directory: Path,
) -> None:
    (run_directory / "server-S1-10-2" / CLIENT_LIMITED_FILE).touch()
    row = summarize_run(find_runs(run_directory)[0], lookup)
    assert (row["client_limited"], row["client_limit"]) == ("true", "")


def test_a_client_limited_file_with_an_unknown_resource_is_rejected(
    run_directory: Path,
) -> None:
    marker = run_directory / "server-S1-10-2" / CLIENT_LIMITED_FILE
    marker.write_text("memory\n")
    with pytest.raises(ValueError, match="must be one of"):
        find_runs(run_directory)


@pytest.fixture
def timed_run(tmp_path: Path) -> Path:
    """Two generators with a common start at 1000 s and a 10 s measured part.

    Generator 0 sends two measured requests of 5 s. Generator 1 sends two of
    4 s and 6 s. Every request reads 3 objects and 30 bytes from S3.
    """
    directory = tmp_path / "raw" / "server-S1-10-2"
    first = [
        request_row(0, warmup=True, seconds=2.0),
        request_row(1, seconds=5.0, dispatch=0.0),
        request_row(2, seconds=5.0, dispatch=5.0),
    ]
    second = [
        request_row(0, user=0, generator=1, warmup=True, seconds=2.0),
        request_row(1, user=0, generator=1, seconds=4.0, dispatch=0.0),
        request_row(2, user=0, generator=1, seconds=6.0, dispatch=4.0),
    ]
    write_requests(directory / "requests-0.csv", first, start_at="1000.0")
    write_requests(directory / "requests-1.csv", second, start_at="1000.0")
    link = LINK_BYTES_PER_S
    write_monitor(
        directory / "monitor-generator-0.csv",
        "generator",
        [(50.0, 0.8 * link)] + [(50.0, 0.0)] * 9,
        first_time=1001.0,
        vcpus="32",
    )
    write_monitor(
        directory / "monitor-generator-1.csv",
        "generator",
        [(25.0, 0.1 * link)] * 10,
        first_time=1001.0,
        vcpus="8",
    )
    write_monitor(
        directory / "monitor-server.csv",
        "server",
        [(100.0, 0)] + [(0.0, 0)] * 9,
        first_time=1001.0,
        memory=40.0,
    )
    return directory.parent


def test_the_sustained_columns_cover_the_measured_part_of_the_run(
    timed_run: Path,
) -> None:
    row = summarize_run(find_runs(timed_run)[0], lookup)
    assert row["server_cpu_max"] == pytest.approx(100.0)
    assert row["server_cpu_sustained"] == pytest.approx(10.0)
    assert row["server_memory_max"] == pytest.approx(40.0)
    assert row["generator_cpu_sustained"] == pytest.approx(0.5)
    assert row["generator_link_sustained"] == pytest.approx(0.1)


def test_generator_cpu_seconds_per_request_is_computed_from_vcpus_and_use(
    timed_run: Path,
) -> None:
    row = summarize_run(find_runs(timed_run)[0], lookup)
    generator_0 = 0.50 * 32 * 10
    generator_1 = 0.25 * 8 * 10
    assert row["generator_cpu_s_per_request"] == pytest.approx(
        (generator_0 + generator_1) / 4
    )
    assert row["generator_cpu_s_per_request"] == pytest.approx(45.0)


def test_generator_cpu_seconds_per_request_is_empty_without_the_vcpu_count(
    timed_run: Path,
) -> None:
    write_monitor(
        timed_run / "server-S1-10-2" / "monitor-generator-1.csv",
        "generator",
        [(25.0, 0.0)] * 10,
        first_time=1001.0,
    )
    row = summarize_run(find_runs(timed_run)[0], lookup)
    assert row["generator_cpu_s_per_request"] == ""
    assert row["generator_cpu_sustained"] == pytest.approx(0.5)


def test_s3_rates_divide_the_measured_requests_by_the_measured_seconds(
    timed_run: Path,
) -> None:
    row = summarize_run(find_runs(timed_run)[0], lookup)
    assert row["s3_requests_per_second"] == pytest.approx(4 * 3 / 10)
    assert row["s3_bytes_per_second"] == pytest.approx(4 * 30 / 10)


def test_s3_rates_leave_out_warmup_requests_and_need_no_start_time(
    run_directory: Path,
) -> None:
    row = summarize_run(find_runs(run_directory)[0], lookup)
    assert row["s3_requests_per_second"] == pytest.approx(8 * 3 / 185.5)
    assert row["s3_bytes_per_second"] == pytest.approx(8 * 30 / 185.5)


def test_a_one_second_link_spike_alone_does_not_mark_a_run_client_limited(
    timed_run: Path,
) -> None:
    row = summarize_run(find_runs(timed_run)[0], lookup)
    assert row["generator_link_sustained"] == pytest.approx(0.1)
    assert (row["client_limited"], row["client_limit"]) == ("false", "")


def test_the_measured_span_runs_from_the_first_measured_dispatch_to_the_last_response(
    tmp_path: Path,
) -> None:
    write_requests(
        tmp_path / "requests-0.csv",
        [
            request_row(0, warmup=True, dispatch=0.0, seconds=9.0),
            request_row(1, dispatch=4.0, seconds=2.0),
            request_row(2, dispatch=6.0, seconds=3.0),
        ],
        start_at="1000.0",
    )
    write_requests(
        tmp_path / "requests-1.csv",
        [
            request_row(0, warmup=True, dispatch=0.0, seconds=1.0),
            request_row(1, dispatch=3.0, seconds=8.0),
            request_row(2, dispatch=5.0, seconds=""),
        ],
        start_at="1000.0",
    )
    paths = [tmp_path / "requests-0.csv", tmp_path / "requests-1.csv"]
    assert measured_span(paths) == (1003.0, 1011.0)


def test_the_measured_span_is_none_without_a_start_time_or_a_measured_request(
    tmp_path: Path,
) -> None:
    write_requests(tmp_path / "undated.csv", [request_row(1)])
    write_requests(
        tmp_path / "warmup.csv",
        [request_row(0, warmup=True)],
        start_at="1000.0",
    )
    assert measured_span([tmp_path / "undated.csv"]) is None
    assert measured_span([tmp_path / "warmup.csv"]) is None


def test_a_sustained_mean_is_the_highest_mean_of_ten_consecutive_samples(
    tmp_path: Path,
) -> None:
    path = tmp_path / "monitor.csv"
    values = [(0.0, 0.0)] * 5 + [(100.0, 0.0)] * 6 + [(10.0, 0.0)] * 9
    write_monitor(path, "generator", values)
    span = (-1.0, 100.0)
    assert sustained_mean(path, "cpu_percent", span) == pytest.approx(
        (6 * 100.0 + 4 * 10.0) / 10
    )
    assert sustained_mean(path, "cpu_percent", span, window_s=3) == (
        pytest.approx(100.0)
    )


def test_a_sustained_mean_over_fewer_samples_than_the_window_is_their_mean(
    tmp_path: Path,
) -> None:
    path = tmp_path / "monitor.csv"
    write_monitor(path, "generator", [(10.0, 0.0), (30.0, 0.0)])
    assert sustained_mean(path, "cpu_percent", (-1.0, 100.0)) == 20.0
    assert sustained_mean(path, "cpu_percent", (500.0, 600.0)) is None


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


def write_not_run(root: Path, name: str, failed: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / NOT_RUN_FILE).write_text(f"{failed}\n")
    return directory


def test_a_directory_with_only_a_not_run_file_is_a_run(
    run_directory: Path,
) -> None:
    write_not_run(run_directory, "server-S1-10-20", "server-S1-10-2")
    runs = find_runs(run_directory)
    assert [run.run for run in runs] == ["server-S1-10-2", "server-S1-10-20"]
    assert [run.not_run for run in runs] == [False, True]
    assert runs[1].requests == ()


def test_a_not_run_row_has_the_run_identity_and_no_statistics(
    tmp_path: Path,
) -> None:
    write_not_run(tmp_path, "precomputed-R3-1h-600-100", "x")
    row = summarize_run(find_runs(tmp_path)[0], lookup)
    assert set(row) == set(RESAMPLING_RUNS)
    assert row["not_run"] == "true"
    assert row["method"] == "precomputed"
    assert row["recording"] == "R3-1h"
    assert row["window_s"] == "600"
    assert row["users"] == 100
    assert row["client_limited"] == "false"
    assert row["repeated"] == "false"
    empty = [
        "requests",
        "failures",
        "failure_rate",
        "median_s",
        "p5_s",
        "p95_s",
        "median_bytes",
        "requests_per_second",
    ]
    assert [row[name] for name in empty] == [""] * len(empty)


def test_a_run_that_ran_is_not_marked_not_run(run_directory: Path) -> None:
    row = summarize_run(find_runs(run_directory)[0], lookup)
    assert row["not_run"] == "false"


def test_a_not_run_directory_with_requests_counts_as_a_run(
    run_directory: Path,
) -> None:
    (run_directory / "server-S1-10-2" / NOT_RUN_FILE).write_text("x\n")
    (run,) = find_runs(run_directory)
    assert run.not_run is False


def test_a_window_with_a_decimal_point_survives_in_a_not_run_row(
    tmp_path: Path,
) -> None:
    write_not_run(tmp_path, "server-R3-0.5-10", "server-R3-0.5-1")
    row = summarize_run(find_runs(tmp_path)[0], lookup)
    assert (row["window_s"], row["users"]) == ("0.5", 10)


def test_summarize_writes_a_not_run_row_and_counts_no_requests_for_it(
    run_directory: Path, tmp_path: Path
) -> None:
    write_not_run(run_directory, "server-S1-10-20", "server-S1-10-2")
    results = tmp_path / "results"
    requests, runs = summarize(run_directory, results, lookup)
    assert (requests, runs) == (10, 2)
    frame = pd.read_csv(
        results / RESAMPLING_RUNS_FILE, comment="#", dtype=str
    ).fillna("")
    assert list(frame["not_run"]) == ["false", "true"]
    assert frame.loc[1, "median_s"] == ""
    assert frame.loc[1, "users"] == "20"


def test_the_median_charged_seconds_count_a_failure_as_the_timeout(
    run_directory: Path,
) -> None:
    paths = sorted((run_directory / "server-S1-10-2").glob("requests-*.csv"))
    seconds = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 180.5, REQUEST_TIMEOUT_S]
    assert median_charged_seconds(paths) == pytest.approx(np.median(seconds))


def test_the_median_charged_seconds_are_none_without_measured_requests(
    tmp_path: Path,
) -> None:
    write_requests(tmp_path / "requests-0.csv", [request_row(0, warmup=True)])
    assert median_charged_seconds([tmp_path / "requests-0.csv"]) is None
    assert median_charged_seconds([]) is None
