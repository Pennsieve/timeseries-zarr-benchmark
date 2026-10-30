import os
import subprocess
import sys
from pathlib import Path

import pytest

from smoke.__main__ import (
    FORMAT_MAX_REPEATS,
    FORMAT_MIN_READS,
    FORMAT_READ_BUDGET_S,
    SmokeError,
    check_format_benchmark_budget,
    environment,
    run_step,
    stop_process_group,
)
from tszbench import schemas
from tszbench.csvmeta import CsvWriter

SLEEPER = "import subprocess, sys, time; " + (
    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
    "time.sleep(60)"
)
"""A process that starts a child and both sleep."""


def test_the_environment_holds_moto_credentials_and_no_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_PROFILE", "work")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://elsewhere")
    variables = environment()
    assert "AWS_PROFILE" not in variables
    assert "AWS_ENDPOINT_URL" not in variables
    assert variables["AWS_ACCESS_KEY_ID"] == "testing"
    assert variables["PATH"] == os.environ["PATH"]


def test_a_failed_step_raises_with_its_label() -> None:
    command = [sys.executable, "-c", "raise SystemExit(3)"]
    with pytest.raises(SmokeError, match="format_benchmark: exit status 3"):
        run_step("format_benchmark", command, dict(os.environ))


def test_a_step_that_exits_with_zero_passes(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    command = [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]
    run_step("touch", command, dict(os.environ))
    assert marker.exists()


def test_stopping_a_group_stops_the_children_too() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", SLEEPER], start_new_session=True
    )
    try:
        stop_process_group(process)
        assert process.returncode is not None
        with pytest.raises(ProcessLookupError):
            os.killpg(process.pid, 0)
    finally:
        if process.poll() is None:
            process.kill()


def write_format_benchmark(
    results: Path,
    opens: list[tuple[int, int]],
    read_count: int,
    read_budget_s: float = FORMAT_READ_BUDGET_S,
    min_reads: int = FORMAT_MIN_READS,
) -> None:
    """Write the two the format benchmark CSVs with one open row per (reads, planned)."""
    header = {"read_budget_s": str(read_budget_s), "min_reads": str(min_reads)}
    with CsvWriter(
        results / schemas.FORMAT_READS_FILE, schemas.FORMAT_READS, header
    ) as writer:
        for _ in range(read_count):
            writer.write(dict.fromkeys(schemas.FORMAT_READS, "0"))
    with CsvWriter(
        results / schemas.FORMAT_OPENS_FILE, schemas.FORMAT_OPENS, header
    ) as writer:
        for reads, planned in opens:
            row = dict.fromkeys(schemas.FORMAT_OPENS, "0")
            row["reads"] = str(reads)
            row["planned_reads"] = str(planned)
            writer.write(row)


def test_consistent_format_benchmark_csvs_pass_the_budget_check(
    tmp_path: Path,
) -> None:
    opens = [
        (FORMAT_MAX_REPEATS, FORMAT_MAX_REPEATS),
        (FORMAT_MIN_READS, FORMAT_MAX_REPEATS),
        (1, 1),
    ]
    write_format_benchmark(
        tmp_path, opens, FORMAT_MAX_REPEATS + FORMAT_MIN_READS + 1
    )
    check_format_benchmark_budget(tmp_path)


def test_a_process_below_the_minimum_reads_fails_the_budget_check(
    tmp_path: Path,
) -> None:
    write_format_benchmark(
        tmp_path, [(FORMAT_MIN_READS - 1, FORMAT_MAX_REPEATS)], 1
    )
    with pytest.raises(SmokeError, match="planned_reads"):
        check_format_benchmark_budget(tmp_path)


def test_a_process_above_the_planned_reads_fails_the_budget_check(
    tmp_path: Path,
) -> None:
    write_format_benchmark(
        tmp_path, [(FORMAT_MAX_REPEATS, FORMAT_MIN_READS)], 3
    )
    with pytest.raises(SmokeError, match="planned_reads"):
        check_format_benchmark_budget(tmp_path)


def test_planned_reads_above_the_maximum_fail_the_budget_check(
    tmp_path: Path,
) -> None:
    planned = FORMAT_MAX_REPEATS + 1
    write_format_benchmark(tmp_path, [(planned, planned)], planned)
    with pytest.raises(SmokeError, match="planned_reads"):
        check_format_benchmark_budget(tmp_path)


def test_a_read_row_count_that_differs_from_the_reads_fails_the_check(
    tmp_path: Path,
) -> None:
    write_format_benchmark(
        tmp_path, [(FORMAT_MIN_READS, FORMAT_MAX_REPEATS)], 5
    )
    with pytest.raises(SmokeError, match="counts 2 reads"):
        check_format_benchmark_budget(tmp_path)


def test_a_header_with_another_budget_fails_the_budget_check(
    tmp_path: Path,
) -> None:
    write_format_benchmark(
        tmp_path,
        [(FORMAT_MIN_READS, FORMAT_MIN_READS)],
        FORMAT_MIN_READS,
        read_budget_s=FORMAT_READ_BUDGET_S * 2,
    )
    with pytest.raises(SmokeError, match="read_budget_s"):
        check_format_benchmark_budget(tmp_path)


def test_a_header_with_another_minimum_fails_the_budget_check(
    tmp_path: Path,
) -> None:
    write_format_benchmark(
        tmp_path,
        [(FORMAT_MIN_READS, FORMAT_MIN_READS)],
        FORMAT_MIN_READS,
        min_reads=FORMAT_MIN_READS + 1,
    )
    with pytest.raises(SmokeError, match="min_reads"):
        check_format_benchmark_budget(tmp_path)
