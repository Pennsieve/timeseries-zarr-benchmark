"""Run every stage end to end on this machine: `python -m smoke [directory]`.

The run generates two small recordings, measures them, uploads them to a local
moto S3 server, and pulls them back as the file server copy. It runs
the format benchmark over local disk, a local HTTP server, and S3, and the resampling benchmark
against a local resampling server. Then it checks, summarizes, and plots. No
AWS account is used. Every server the run starts stops when the run ends.
"""

import argparse
import contextlib
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import FrameType

import httpx

from format_benchmark.positions import (
    MIN_READS_KEY,
    READ_BUDGET_KEY,
    SUBSET_WINDOWS_S,
)
from resampling_benchmark.load.orchestrate import USERS_PER_GENERATOR
from tszbench import schemas
from tszbench.csvmeta import read_header, read_rows
from tszbench.rangeserver import free_port, serve_directory

RECORDINGS = ("R1-1h", "R2-1h")
"""Recordings of the run. Every format of each is written."""

PLOT_RECORDING = "R2-1h"
"""Recording of Figures 2, 3, and 5, Table 3, and the cost paragraph. It is both the
format recording and the resampling recording of the plot stage."""

BUCKET = "tszbench-smoke"
PREFIX = "data"
REGION = "us-east-1"

SUBSET_WINDOW_S = SUBSET_WINDOWS_S[0]
"""Window of the subset reads of the format benchmark."""

FORMAT_WINDOWS_S = (1, 10, 60)
"""Windows of the format benchmark. Figure 1 needs 10 s."""

FORMAT_MAX_REPEATS = 3
"""Reads per the format benchmark process."""

FORMAT_READ_BUDGET_S = 0.05
"""Read budget of a format benchmark process, in seconds. Small enough that slow
combinations stop early and fast ones make every planned read."""

FORMAT_MIN_READS = 2
"""Minimum reads of a format benchmark process."""

RESAMPLING_WINDOWS_S = (10, 60)
RESAMPLING_USERS = (1, 10)
RESAMPLING_MEASURED_REQUESTS = 2
"""Measured requests per user, after one warm-up request."""

RESAMPLING_PROCESSES = 2
"""Processes of each load generator."""

RESAMPLING_TIME_LIMIT_S = 300.0
"""Seconds after which the users of a resampling benchmark run stop sending."""

CHECK_WINDOWS_S = (10, 60, 600)

SERVICE_PROCESSES = 4
SERVICE_START_TIMEOUT_S = 120.0
STOP_TIMEOUT_S = 20.0

EXPECTED_OUTPUTS = (
    "fig1.pdf",
    "fig2.pdf",
    "fig3.pdf",
    "fig4.pdf",
    "fig5.pdf",
    "table1.csv",
    "table2.csv",
    "table3.csv",
    "table4.csv",
    "methods.txt",
    "cost.txt",
    "captions.txt",
)
"""Files the plot stage must write for the run to pass."""

S3_ENVIRONMENT = {
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_DEFAULT_REGION": REGION,
    "AWS_REGION": REGION,
    "AWS_EC2_METADATA_DISABLED": "true",
}
"""Credentials moto accepts. They replace any AWS profile of the caller."""


class SmokeError(RuntimeError):
    """A stage of the run failed."""


def environment() -> dict[str, str]:
    """Return the environment of every stage: this one with moto credentials."""
    variables = dict(os.environ)
    for name in ("AWS_PROFILE", "AWS_SESSION_TOKEN", "AWS_ENDPOINT_URL"):
        variables.pop(name, None)
    variables.update(S3_ENVIRONMENT)
    return variables


def python_module(module: str, *arguments: str) -> list[str]:
    """Return the command line that runs module with this interpreter."""
    return [sys.executable, "-m", module, *arguments]


def run_step(label: str, command: Sequence[str], env: dict[str, str]) -> None:
    """Run one stage command and print its wall time.

    The command runs in a process group of its own. When the run stops before
    the command ends, the whole group is stopped.

    Raises:
        SmokeError: The command exits with a nonzero status.
    """
    print(f"== {label}", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(list(command), env=env, start_new_session=True)
    try:
        status = process.wait()
    except BaseException:
        stop_process_group(process)
        raise
    seconds = time.monotonic() - started
    if status != 0:
        raise SmokeError(f"{label}: exit status {status} after {seconds:.0f} s")
    print(f"== {label}: {seconds:.1f} s", flush=True)


@contextmanager
def moto_server() -> Iterator[str]:
    """Run a moto S3 server on a thread and yield its endpoint URL."""
    from moto.server import ThreadedMotoServer

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    port = free_port()
    server = ThreadedMotoServer(
        ip_address="127.0.0.1", port=port, verbose=False
    )
    server.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.stop()


@contextmanager
def http_server(directory: Path) -> Iterator[str]:
    """Serve directory with byte ranges and yield its base URL."""
    running = serve_directory(directory)
    try:
        yield running.url
    finally:
        running.stop()


def stop_process_group(process: "subprocess.Popen[bytes]") -> None:
    """Stop every process of the group that process leads.

    The group gets SIGTERM, and SIGKILL after `STOP_TIMEOUT_S` for any process
    that remains.
    """
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(STOP_TIMEOUT_S)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


@contextmanager
def resampling_service(
    s3_root: str, endpoint: str, env: dict[str, str]
) -> Iterator[str]:
    """Run `python -m resampling_benchmark.service` and yield its URL once it answers.

    Raises:
        SmokeError: The service exits or does not answer in time.
    """
    port = free_port()
    command = python_module(
        "resampling_benchmark.service",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--processes", str(SERVICE_PROCESSES),
        "--s3-root", s3_root,
        "--s3-endpoint-url", endpoint,
        "--recordings", ",".join(RECORDINGS),
    )  # fmt: skip
    print("== resampling service", flush=True)
    process = subprocess.Popen(command, env=env, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + SERVICE_START_TIMEOUT_S
        while True:
            if process.poll() is not None:
                raise SmokeError(
                    f"resampling service: exit status {process.returncode}"
                )
            try:
                if httpx.get(f"{url}/health", timeout=2.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise SmokeError(
                    "resampling service: no answer in "
                    f"{SERVICE_START_TIMEOUT_S:.0f} s"
                )
            time.sleep(0.5)
        yield url
    finally:
        stop_process_group(process)


def _raise_on_terminate(number: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def check_format_benchmark_budget(results: Path) -> None:
    """Check that the format benchmark CSVs agree with the read budget of the run.

    Every open row holds `reads` between the smaller of `FORMAT_MIN_READS` and
    `planned_reads`, and `planned_reads`, which is at most `FORMAT_MAX_REPEATS`.
    The read CSV holds as many rows as the open rows count reads. Both headers
    record `FORMAT_READ_BUDGET_S` and `FORMAT_MIN_READS`.

    Raises:
        SmokeError: A CSV disagrees with the constants or with the other CSV.
    """
    reads_path = results / schemas.FORMAT_READS_FILE
    open_path = results / schemas.FORMAT_OPENS_FILE
    for path in (reads_path, open_path):
        header = read_header(path)
        if float(header[READ_BUDGET_KEY]) != FORMAT_READ_BUDGET_S:
            raise SmokeError(
                f"format_benchmark: {path.name} records {READ_BUDGET_KEY} "
                f"{header[READ_BUDGET_KEY]}, expected {FORMAT_READ_BUDGET_S}"
            )
        if int(header[MIN_READS_KEY]) != FORMAT_MIN_READS:
            raise SmokeError(
                f"format_benchmark: {path.name} records {MIN_READS_KEY} "
                f"{header[MIN_READS_KEY]}, expected {FORMAT_MIN_READS}"
            )
    opens = read_rows(open_path)
    if not opens:
        raise SmokeError(f"format_benchmark: {open_path.name} holds no rows")
    total = 0
    for row in opens:
        reads = int(row["reads"])
        planned = int(row["planned_reads"])
        lowest = min(FORMAT_MIN_READS, planned)
        if not lowest <= reads <= planned <= FORMAT_MAX_REPEATS:
            raise SmokeError(
                f"format_benchmark: {open_path.name} row {row}: reads must satisfy "
                f"{lowest} <= reads <= planned_reads <= {FORMAT_MAX_REPEATS}"
            )
        total += reads
    count = len(read_rows(reads_path))
    if count != total:
        raise SmokeError(
            f"format_benchmark: {reads_path.name} holds {count} rows, but "
            f"{open_path.name} counts {total} reads"
        )


def check_resampling_runs(results: Path) -> None:
    """Check the resampling benchmark CSVs against the schema and the user caps.

    Every request CSV of the raw directory records the users per generator of
    its method in the `method_users_per_generator` header key. The run CSV has
    one column per entry of `schemas.RESAMPLING_RUNS`.

    Raises:
        SmokeError: A request CSV records another cap, or the run CSV has other
            columns.
    """
    raw = results / schemas.RESAMPLING_RAW_DIRECTORY
    paths = sorted(raw.glob(f"*/{schemas.REQUESTS_PATTERN}"))
    if not paths:
        raise SmokeError(f"resampling_benchmark: no request CSV under {raw}")
    for path in paths:
        header = read_header(path)
        expected = USERS_PER_GENERATOR[header["method"]]
        recorded = header.get("method_users_per_generator")
        if recorded != str(expected):
            raise SmokeError(
                f"resampling_benchmark: {path.name} of {header['run']} records "
                f"method_users_per_generator {recorded}, expected {expected}"
            )
    runs = read_rows(results / schemas.RESAMPLING_RUNS_FILE)
    if not runs or list(runs[0]) != list(schemas.RESAMPLING_RUNS):
        raise SmokeError(
            f"resampling_benchmark: {schemas.RESAMPLING_RUNS_FILE} must hold "
            f"the columns {list(schemas.RESAMPLING_RUNS)}"
        )


def check_subset_reads(results: Path) -> None:
    """Check that the format benchmark measured the subset reads of the plot recording.

    The plot recording has more channels than the 4 and 16 of the first two
    subset kinds, so each has rows at 10 s.

    Raises:
        SmokeError: A subset kind has no read of the plot recording at 10 s.
    """
    rows = read_rows(results / schemas.FORMAT_READS_FILE)
    for kind in schemas.SUBSET_KINDS[:2]:
        found = any(
            row["recording"] == PLOT_RECORDING
            and row["read_kind"] == kind
            and int(row["window_s"]) == SUBSET_WINDOW_S
            for row in rows
        )
        if not found:
            raise SmokeError(
                f"format_benchmark: {schemas.FORMAT_READS_FILE} has no "
                f"{kind} read of {PLOT_RECORDING} at {SUBSET_WINDOW_S} s"
            )


def numbers(values: Sequence[int]) -> str:
    """Return values as a comma-separated list."""
    return ",".join(str(value) for value in values)


def run(out: Path) -> None:
    """Run every stage with its output under out.

    Raises:
        SmokeError: A stage fails or the plot stage leaves out an output.
    """
    root = out / "root"
    pulled = out / "pulled"
    results = out / "results"
    figures = out / "figures"
    env = environment()
    recordings = ",".join(RECORDINGS)
    s3_root = f"{BUCKET}/{PREFIX}"
    for name in RECORDINGS:
        run_step(
            f"generate {name}",
            python_module(
                "generate",
                "--recording", name,
                "--root", str(root),
                "--results", str(results),
                "--overwrite",
            ),
            env,
        )  # fmt: skip
    run_step(
        "storage",
        python_module(
            "upload", "storage",
            "--root", str(root),
            "--recordings", recordings,
            "--results", str(results),
        ),
        env,
    )  # fmt: skip
    with ExitStack() as servers:
        endpoint = servers.enter_context(moto_server())
        store = ["--bucket", BUCKET, "--prefix", PREFIX]
        store += ["--s3-endpoint-url", endpoint]
        run_step(
            "create bucket",
            ["aws", "s3", "mb", f"s3://{BUCKET}", "--endpoint-url", endpoint],
            env,
        )
        for name in RECORDINGS:
            run_step(
                f"upload {name}",
                python_module(
                    "upload", "s3",
                    "--root", str(root),
                    "--recording", name,
                    "--results", str(results),
                    *store,
                ),
                env,
            )  # fmt: skip
        run_step(
            "pull",
            python_module("upload", "pull", "--root", str(pulled), *store),
            env,
        )
        http_root = servers.enter_context(http_server(pulled))
        run_step(
            "format_benchmark",
            python_module(
                "format_benchmark",
                "--recordings", recordings,
                "--windows", numbers(FORMAT_WINDOWS_S),
                "--max-repeats", str(FORMAT_MAX_REPEATS),
                "--read-budget-s", str(FORMAT_READ_BUDGET_S),
                "--min-reads", str(FORMAT_MIN_READS),
                "--results", str(results),
                "--local-root", str(root),
                "--http-root", http_root,
                "--s3-root", s3_root,
                "--s3-endpoint-url", endpoint,
            ),
            env,
        )  # fmt: skip
        check_format_benchmark_budget(results)
        check_subset_reads(results)
        service = servers.enter_context(
            resampling_service(s3_root, endpoint, env)
        )
        run_step(
            "resampling_benchmark",
            python_module(
                "resampling_benchmark.load.orchestrate",
                "--local",
                "--server-url", service,
                "--s3-root", s3_root,
                "--s3-endpoint-url", endpoint,
                "--out", str(results / schemas.RESAMPLING_RAW_DIRECTORY),
                "--recordings", recordings,
                "--windows", numbers(RESAMPLING_WINDOWS_S),
                "--users", numbers(RESAMPLING_USERS),
                "--measured-requests", str(RESAMPLING_MEASURED_REQUESTS),
                "--processes", str(RESAMPLING_PROCESSES),
                "--time-limit-s", str(RESAMPLING_TIME_LIMIT_S),
                "--start-delay-s", "1",
            ),
            env,
        )  # fmt: skip
        run_step(
            "check",
            python_module(
                "resampling_benchmark.check",
                "--server-url", service,
                "--s3-root", s3_root,
                "--s3-endpoint-url", endpoint,
                "--recordings", recordings,
                "--windows", numbers(CHECK_WINDOWS_S),
                "--results", str(results),
            ),
            env,
        )  # fmt: skip
    run_step(
        "summarize",
        python_module(
            "resampling_benchmark.summarize",
            "--input", str(results / schemas.RESAMPLING_RAW_DIRECTORY),
            "--results", str(results),
        ),
        env,
    )  # fmt: skip
    check_resampling_runs(results)
    run_step(
        "plot",
        python_module(
            "plot",
            "--results", str(results),
            "--out", str(figures),
            "--recording", PLOT_RECORDING,
            "--format-recording", PLOT_RECORDING,
        ),
        env,
    )  # fmt: skip
    missing = [
        name for name in EXPECTED_OUTPUTS if not (figures / name).exists()
    ]
    if missing:
        raise SmokeError(f"plot: wrote no {missing} under {figures}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke run and return the exit status."""
    parser = argparse.ArgumentParser(
        prog="python -m smoke", description=__doc__
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="directory of every output (default a new temporary directory)",
    )
    arguments = parser.parse_args(argv)
    out: Path = arguments.directory or Path(
        tempfile.mkdtemp(prefix="tszbench-smoke-")
    )
    out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _raise_on_terminate)
    print(f"smoke run in {out}", flush=True)
    started = time.monotonic()
    try:
        run(out)
    except SmokeError as error:
        print(f"smoke run failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("smoke run stopped", file=sys.stderr)
        return 130
    print(
        f"smoke run passed in {time.monotonic() - started:.0f} s; "
        f"results in {out / 'results'}, figures in {out / 'figures'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
