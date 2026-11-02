import json
from dataclasses import replace
from pathlib import Path

import pytest

from resampling_benchmark.calibrate import calibrate, main
from resampling_benchmark.load.generator import LINK_BYTES_PER_S
from tszbench.csvmeta import CsvWriter, read_header, read_rows
from tszbench.recordings import RECORDINGS, RecordingSpec
from tszbench.schemas import (
    CALIBRATION_GENERATOR,
    CALIBRATION_GENERATOR_FILE,
    CALIBRATION_SERVER,
    CALIBRATION_SERVER_FILE,
    MONITOR,
    NOT_RUN_FILE,
    RESAMPLING_REQUESTS,
    SERVER_METADATA_FILE,
)

SPEC = replace(RECORDINGS["R3"], id="S1", channels=3, rate_hz=100)


def lookup(name: str) -> RecordingSpec:
    assert name == "S1"
    return SPEC


def write_run(
    root: Path,
    method: str,
    users: int,
    *,
    task_threads: int | None = None,
    cpu: float = 25.0,
    seconds: float = 5.0,
    recording: str = "S1",
) -> Path:
    """Write one run: two measured requests of `seconds` over a 2 s span.

    The generator has 8 vCPU and the measured part starts at 1000 s.
    """
    name = f"{method}-{recording}-10-{users}"
    directory = root / name
    directory.mkdir(parents=True)
    rows = [
        {
            "run": name,
            "method": method,
            "recording": recording,
            "window_s": "10",
            "users": users,
            "generator": 0,
            "user": 0,
            "sequence": sequence,
            "start_us": 0,
            "dispatch_s": float(sequence),
            "seconds": seconds,
            "bytes": 1000,
            "s3_requests": 4,
            "s3_bytes": 40,
            "status": "ok",
            "warmup": "false",
            "level": 0,
            "read_seconds": 0.5,
            "reduce_seconds": 0.25,
        }
        for sequence in range(2)
    ]
    with CsvWriter(
        directory / "requests-0.csv",
        RESAMPLING_REQUESTS,
        {"start_at": "1000.0"},
    ) as writer:
        writer.write_all(rows)
    for role, file_name, header in (
        ("server", "monitor-server.csv", {}),
        ("generator", "monitor-generator-0.csv", {"vcpus": "8"}),
    ):
        with CsvWriter(directory / file_name, MONITOR, header) as writer:
            for index in range(7):
                writer.write(
                    {
                        "run": name,
                        "host": "h",
                        "role": role,
                        "time_s": 1001.0 + index,
                        "cpu_percent": cpu,
                        "rx_bytes_per_s": 0.1 * LINK_BYTES_PER_S,
                        "tx_bytes_per_s": 0.0,
                        "memory_percent": 30.0,
                    }
                )
    if task_threads is not None:
        (directory / SERVER_METADATA_FILE).write_text(
            json.dumps({"task_threads": str(task_threads), "instance": "x"})
        )
    return directory


def test_server_rows_hold_the_task_threads_and_the_summarize_columns(
    tmp_path: Path,
) -> None:
    root = tmp_path / "threads-4"
    write_run(root, "server", 10, task_threads=4, cpu=40.0)
    results = tmp_path / "results"
    assert calibrate([root], results, "server", lookup) == 1
    [row] = read_rows(results / CALIBRATION_SERVER_FILE)
    assert tuple(row) == CALIBRATION_SERVER
    assert row["task_threads"] == "4"
    assert row["run"] == "server-S1-10-10"
    assert row["recording"] == "S1"
    assert row["window_s"] == "10"
    assert row["users"] == "10"
    assert float(row["median_s"]) == 5.0
    assert float(row["server_cpu_sustained"]) == pytest.approx(40.0)
    assert float(row["server_memory_max"]) == pytest.approx(30.0)
    assert float(row["requests_per_second"]) == pytest.approx(2 / 6.0)
    assert float(row["s3_requests_per_second"]) == pytest.approx(8 / 6.0)


def test_generator_rows_hold_the_sustained_load_and_the_cpu_cost(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    write_run(root, "precomputed", 5, cpu=50.0)
    results = tmp_path / "results"
    assert calibrate([root], results, "generator", lookup) == 1
    [row] = read_rows(results / CALIBRATION_GENERATOR_FILE)
    assert tuple(row) == CALIBRATION_GENERATOR
    assert row["run"] == "precomputed-S1-10-5"
    assert float(row["generator_cpu_sustained"]) == pytest.approx(0.5)
    assert float(row["generator_link_sustained"]) == pytest.approx(0.1)
    # 0.5 x 8 vCPU x 6 s measured / 2 requests.
    assert float(row["generator_cpu_s_per_request"]) == pytest.approx(12.0)


def test_each_kind_takes_only_the_runs_of_its_method(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    write_run(root, "server", 1, task_threads=1)
    write_run(root, "precomputed", 1)
    results = tmp_path / "results"
    calibrate([root], results, "server", lookup)
    calibrate([root], results, "generator", lookup)
    [server] = read_rows(results / CALIBRATION_SERVER_FILE)
    [generator] = read_rows(results / CALIBRATION_GENERATOR_FILE)
    assert server["run"] == "server-S1-10-1"
    assert generator["run"] == "precomputed-S1-10-1"


def test_rows_of_several_roots_keep_their_own_task_threads(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / f"threads-{value}" for value in (1, 8)]
    for root, value in zip(roots, (1, 8), strict=True):
        write_run(root, "server", 10, task_threads=value)
    results = tmp_path / "results"
    assert calibrate(roots, results, "server", lookup) == 2
    rows = read_rows(results / CALIBRATION_SERVER_FILE)
    assert [row["task_threads"] for row in rows] == ["1", "8"]


def test_task_threads_is_empty_without_server_metadata(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    write_run(root, "server", 10)
    results = tmp_path / "results"
    calibrate([root], results, "server", lookup)
    [row] = read_rows(results / CALIBRATION_SERVER_FILE)
    assert row["task_threads"] == ""


def test_a_run_that_was_skipped_gets_no_row(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    write_run(root, "server", 10, task_threads=1)
    skipped = root / "server-S1-10-100"
    skipped.mkdir()
    (skipped / NOT_RUN_FILE).write_text("server-S1-10-10\n")
    results = tmp_path / "results"
    assert calibrate([root], results, "server", lookup) == 1


def test_the_csv_opens_with_the_run_metadata(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    write_run(root, "server", 10, task_threads=1)
    results = tmp_path / "results"
    calibrate([root], results, "server", lookup)
    header = read_header(results / CALIBRATION_SERVER_FILE)
    assert {"date", "commit", "python", "version.zarr"} <= set(header)
    assert header["server.task_threads"] == "1"


def test_main_writes_the_file_and_reports_the_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "raw"
    write_run(root, "precomputed", 1, recording="R3")
    results = tmp_path / "results"
    main(["generator", "--input", str(root), "--results", str(results)])
    assert capsys.readouterr().out == "1 generator calibration rows\n"
    [row] = read_rows(results / CALIBRATION_GENERATOR_FILE)
    assert row["recording"] == "R3"
