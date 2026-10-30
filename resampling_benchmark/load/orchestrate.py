"""Run the resampling benchmark grid: `python -m resampling_benchmark.load.orchestrate`.

Runs on the benchmark client. For every run it starts monitors and load
generators on the hosts of a hosts file over SSH, waits for the generators,
and copies their CSVs back. The resampling server is already running.
"""

import argparse
import contextlib
import json
import logging
import math
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
import pandas as pd

from resampling_benchmark.load.generator import (
    LINK_BYTES_PER_S,
    MEASURED_REQUESTS,
    REQUEST_TIMEOUT_S,
    RUN_TIME_LIMIT_S,
    WARMUP_REQUESTS,
    format_window,
)
from resampling_benchmark.summarize import median_charged_seconds
from tszbench.recordings import RECORDINGS, resolve
from tszbench.schemas import (
    COMPLETE_FILE,
    IDLE_WAIT_FILE,
    LINK_LIMITED_FILE,
    METHODS,
    NOT_RUN_FILE,
    REQUESTS_PATTERN,
    RESAMPLING_RAW_DIRECTORY,
    SERVER_METADATA_FILE,
    SERVER_MONITOR_FILE,
    generator_monitor_file,
    requests_file,
)

LOGGER = logging.getLogger("resampling_benchmark.orchestrate")

USERS_PER_GENERATOR = 100
START_DELAY_S = 15.0
"""Seconds between launching a run and its common start time."""

STOP_GRACE_S = 120.0
"""Seconds past the run limit and one request timeout before a generator is
stopped."""

WINDOWS_S = (10, 60, 600, 3_600, 86_400)
USERS_FULL = (1, 10, 100, 1_000)
USERS_SINGLE = (1,)
USER_COUNTS = {"R3": USERS_FULL, "R4": (1, 100)}
"""User counts to run for each recording that does not run `USERS_SINGLE`."""

HEALTH_TIMEOUT_S = 300.0

IDLE_TIMEOUT_S = 1_800.0
"""Seconds the server has to finish the work of the run before."""

IDLE_POLL_S = 2.0
"""Seconds between two checks of the server's `in_flight` count."""

REMOTE_PYTHON = "timeseries-zarr-benchmark/.venv/bin/python"
"""Python of the benchmark clone on every host, relative to the login home.
The Terraform user data clones the repository and makes the venv there."""


@dataclass(frozen=True)
class RunSpec:
    """One cell of the grid.

    Attributes:
        method: `server` or `precomputed`.
        recording: Recording id.
        window_s: Window length in seconds.
        users: Users of the whole run.
    """

    method: str
    recording: str
    window_s: int
    users: int

    @property
    def name(self) -> str:
        """Run id, such as `server-R3-600-100`."""
        return f"{self.method}-{self.recording}-{format_window(self.window_s)}-{self.users}"


@dataclass(frozen=True)
class Hosts:
    """Machines of a resampling benchmark deployment.

    Attributes:
        server: Private address of the resampling server.
        generators: Private addresses of the load generators.
    """

    server: str
    generators: tuple[str, ...]


@dataclass(frozen=True)
class RunResult:
    """What the orchestrator recorded for one run.

    Attributes:
        spec: The grid cell.
        run: Run id of the files, `<spec name>-repeat` for a repeat.
        directory: Local directory that holds the run's CSVs.
        generators: Generators used.
        users_per_generator: Users on the busiest generator.
        peak_rx_bytes_per_s: Highest receive rate of any generator, or None
            when no monitor ran.
        repeated: True for a repeat with half the users per generator.
    """

    spec: RunSpec
    run: str
    directory: Path
    generators: int
    users_per_generator: int
    peak_rx_bytes_per_s: float | None
    repeated: bool


SERVER_ROLE = "resampling_server"
"""Role of the resampling server in a hosts file."""

GENERATOR_ROLE = "load_generator"
"""Role of a load generator in a hosts file."""


def parse_hosts(text: str) -> Hosts:
    """Parse a hosts file of `role private_ip` lines.

    Blank lines and lines that start with `#` are skipped. The roles are
    `resampling_server`, once, and `load_generator`, one or more, as the
    Terraform output `hosts_file` writes them. Other roles are ignored.

    Raises:
        ValueError: A line is malformed, or the server or every generator is
            missing.
    """
    server: list[str] = []
    generators: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"hosts line {number}: must read `role private_ip`, got {line!r}"
            )
        role, address = parts
        if role == SERVER_ROLE:
            server.append(address)
        elif role == GENERATOR_ROLE:
            generators.append(address)
    if len(server) != 1 or not generators:
        raise ValueError(
            f"hosts file: needs exactly one {SERVER_ROLE} and at least one "
            f"{GENERATOR_ROLE}, got {len(server)} and {len(generators)}"
        )
    return Hosts(server[0], tuple(generators))


def plan_runs(
    methods: Iterable[str] = METHODS,
    recordings: Iterable[str] = tuple(RECORDINGS),
    windows: Iterable[int] = WINDOWS_S,
    users: Iterable[int] | None = None,
) -> list[RunSpec]:
    """Return the runs in plan order.

    The order nests method, recording, window, and user count. R1 to R4 come
    first in that order, then any other recording, such as a 1 h cut, in the
    order given. R3 runs 1, 10, 100, and 1000 users, and R4 runs 1 and 100.
    Every other recording runs 1 user. A window longer than the recording is
    left out.

    Args:
        methods: Methods to run.
        recordings: Recordings to run, by any name that
            `tszbench.recordings.resolve` accepts.
        windows: Window lengths in seconds to run.
        users: When given, user counts that replace the plan's counts.

    Raises:
        ValueError: A recording name is unknown.
    """
    runs = []
    chosen_recordings = list(dict.fromkeys(recordings))
    ordered = [name for name in RECORDINGS if name in chosen_recordings]
    ordered += [name for name in chosen_recordings if name not in RECORDINGS]
    durations = {name: resolve(name).duration_s for name in ordered}
    chosen_windows = set(windows)
    for method in methods:
        for recording in ordered:
            counts = USER_COUNTS.get(recording, USERS_SINGLE)
            for window in WINDOWS_S:
                if window not in chosen_windows:
                    continue
                if window > durations[recording]:
                    continue
                for count in counts if users is None else tuple(users):
                    runs.append(RunSpec(method, recording, window, count))
    return runs


def split_generators(users: int, available: int) -> tuple[int, int]:
    """Return the generators to use and the users on the busiest one.

    Uses `ceil(users / USERS_PER_GENERATOR)` generators, or all that are
    available when fewer.
    """
    generators = max(1, min(available, math.ceil(users / USERS_PER_GENERATOR)))
    return generators, math.ceil(users / generators)


def users_on(users: int, generators: int, index: int) -> int:
    """Return the users of generator `index` when users are dealt in turn."""
    return users // generators + (1 if index < users % generators else 0)


def generator_command(
    python: str,
    spec: RunSpec,
    run: str,
    users: int,
    generators: int,
    index: int,
    start_at: float,
    server_url: str,
    s3_root: str,
    s3_endpoint_url: str | None,
    out: str,
    repeated: bool,
    processes: int | None = None,
    warmup_requests: int = WARMUP_REQUESTS,
    measured_requests: int = MEASURED_REQUESTS,
    time_limit_s: float = RUN_TIME_LIMIT_S,
    duration_s: float | None = None,
) -> list[str]:
    """Return the command line of one load generator."""
    command = [
        python,
        "-m",
        "resampling_benchmark.load.generator",
        "--method",
        spec.method,
        "--recording",
        spec.recording,
        "--window-s",
        str(spec.window_s),
        "--users",
        str(users_on(users, generators, index)),
        "--total-users",
        str(users),
        "--generators",
        str(generators),
        "--generator-index",
        str(index),
        "--run",
        run,
        "--start-at",
        repr(start_at),
        "--s3-root",
        s3_root,
        "--out",
        out,
        "--warmup-requests",
        str(warmup_requests),
        "--measured-requests",
        str(measured_requests),
        "--time-limit-s",
        str(time_limit_s),
    ]
    if spec.method == "server":
        command += ["--server-url", server_url]
    if s3_endpoint_url:
        command += ["--s3-endpoint-url", s3_endpoint_url]
    if processes is not None:
        command += ["--processes", str(processes)]
    if duration_s is not None:
        command += ["--duration-s", str(duration_s)]
    if repeated:
        command.append("--repeated")
    return command


def monitor_command(python: str, run: str, role: str, out: str) -> list[str]:
    """Return the command line of one monitor."""
    return [
        python,
        "-m",
        "resampling_benchmark.load.monitor",
        "--run",
        run,
        "--role",
        role,
        "--out",
        out,
    ]


class Job(Protocol):
    """A command running on a host."""

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the command and return its exit status.

        Raises:
            subprocess.TimeoutExpired: The command outlasted timeout.
        """
        ...

    def stop(self) -> None:
        """Ask the command to stop."""
        ...


class Runner(Protocol):
    """Runs commands on hosts and copies files back."""

    def launch(self, host: str, argv: Sequence[str]) -> Job:
        """Start argv on host without waiting."""
        ...

    def prepare(self, host: str, directory: str) -> None:
        """Create directory on host."""
        ...

    def fetch(self, host: str, remote: str, local: Path) -> bool:
        """Copy a file from host to local. Returns False when it is absent."""
        ...


class _ProcessJob:
    def __init__(self, process: "subprocess.Popen[bytes]") -> None:
        self._process = process

    def wait(self, timeout: float | None = None) -> int:
        return self._process.wait(timeout)

    def stop(self) -> None:
        self._process.terminate()


class LocalRunner:
    """Runs every command as a subprocess of this machine."""

    def launch(self, host: str, argv: Sequence[str]) -> Job:
        """Start argv locally. The host is ignored."""
        return _ProcessJob(subprocess.Popen(list(argv)))

    def prepare(self, host: str, directory: str) -> None:
        """Create directory."""
        Path(directory).mkdir(parents=True, exist_ok=True)

    def fetch(self, host: str, remote: str, local: Path) -> bool:
        """Copy remote to local."""
        source = Path(remote)
        if not source.exists():
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != local.resolve():
            shutil.copyfile(source, local)
        return True


SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new")


def kill_pattern(argv: Sequence[str]) -> str:
    """Return the quoted `pkill -f` pattern that matches only the command argv.

    The first letter of the module name sits in brackets so that the pattern
    does not match the shell that runs `pkill`.
    """
    module = next(
        part for part in argv if part.startswith("resampling_benchmark.")
    )
    run = argv[argv.index("--run") + 1]
    return shlex.quote(f"[{module[0]}]{module[1:]} .*--run {run}( |$)")


class _SshJob:
    def __init__(
        self, process: "subprocess.Popen[bytes]", host: str, pattern: str
    ) -> None:
        self._process = process
        self._host = host
        self._pattern = pattern

    def wait(self, timeout: float | None = None) -> int:
        return self._process.wait(timeout)

    def stop(self) -> None:
        subprocess.run(
            [
                "ssh",
                *SSH_OPTIONS,
                self._host,
                f"pkill -TERM -f {self._pattern}",
            ],
            check=False,
        )


class SshRunner:
    """Runs commands over SSH and copies files with scp."""

    def __init__(self, directory: str = ".") -> None:
        """Run commands in `directory` on the hosts."""
        self._directory = directory

    def launch(self, host: str, argv: Sequence[str]) -> Job:
        """Start argv on host in the working directory."""
        command = f"cd {shlex.quote(self._directory)} && {shlex.join(argv)}"
        process = subprocess.Popen(["ssh", *SSH_OPTIONS, host, command])
        return _SshJob(process, host, kill_pattern(argv))

    def prepare(self, host: str, directory: str) -> None:
        """Create directory on host."""
        subprocess.run(
            ["ssh", *SSH_OPTIONS, host, f"mkdir -p {shlex.quote(directory)}"],
            check=True,
        )

    def fetch(self, host: str, remote: str, local: Path) -> bool:
        """Copy remote from host to local."""
        local.parent.mkdir(parents=True, exist_ok=True)
        done = subprocess.run(
            ["scp", "-q", *SSH_OPTIONS, f"{host}:{remote}", str(local)],
            check=False,
        )
        return done.returncode == 0


@dataclass(frozen=True)
class Settings:
    """Fixed inputs of every run.

    Attributes:
        hosts: Machines to use.
        server_url: Base URL of the resampling server.
        s3_root: `bucket/prefix` of the bundles.
        s3_endpoint_url: S3-compatible endpoint, or None for AWS.
        out: Local directory that receives one subdirectory per run.
        remote_dir: Directory on each host that receives the CSVs.
        python: Python executable on the hosts.
        monitors: Whether to run monitors.
        processes: Generator processes per generator, None for the default.
        warmup_requests: Warm-up requests per user.
        measured_requests: Measured requests per user.
        time_limit_s: Run limit in seconds.
        start_delay_s: Seconds from launch to the common start time.
        duration_s: Recording length to pass to the generators, or None to
            look it up from the recording id.
        idle_check: Called with `server_url` before each server run. Returns
            the seconds spent until the server had no task in flight. None
            skips the wait.
        server_metadata: Called with `server_url` at the start of each run.
            Returns the run metadata of the server, which the run writes to
            `SERVER_METADATA_FILE`. None skips the file.
    """

    hosts: Hosts
    server_url: str
    s3_root: str
    s3_endpoint_url: str | None
    out: Path
    remote_dir: str
    python: str = "python"
    monitors: bool = True
    processes: int | None = None
    warmup_requests: int = WARMUP_REQUESTS
    measured_requests: int = MEASURED_REQUESTS
    time_limit_s: float = RUN_TIME_LIMIT_S
    start_delay_s: float = START_DELAY_S
    duration_s: float | None = None
    idle_check: Callable[[str], float] | None = None
    server_metadata: Callable[[str], dict[str, str]] | None = None


def peak_receive_rate(paths: Iterable[Path]) -> float | None:
    """Return the highest `rx_bytes_per_s` in monitor CSVs, or None if no rows."""
    peak: float | None = None
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path, comment="#")
        if len(frame):
            value = float(frame["rx_bytes_per_s"].max())
            peak = value if peak is None else max(peak, value)
    return peak


def execute_run(
    spec: RunSpec,
    run: str,
    generators: int,
    settings: Settings,
    runner: Runner,
    repeated: bool,
    clock: Callable[[], float] = time.time,
) -> RunResult:
    """Run one cell with a set number of generators and collect its CSVs.

    A server run first waits until the server has no task in flight, and
    writes the wait to `IDLE_WAIT_FILE`. Every run writes the server's run
    metadata to `SERVER_METADATA_FILE` when `Settings.server_metadata` is set.
    A failed lookup is logged and leaves the file out.

    Raises:
        RuntimeError: A load generator exits with an error.
        TimeoutError: The server stays busy for `IDLE_TIMEOUT_S`.
    """
    hosts = settings.hosts
    directory = settings.out / run
    directory.mkdir(parents=True, exist_ok=True)
    (directory / COMPLETE_FILE).unlink(missing_ok=True)
    (directory / LINK_LIMITED_FILE).unlink(missing_ok=True)
    (directory / NOT_RUN_FILE).unlink(missing_ok=True)
    (directory / SERVER_METADATA_FILE).unlink(missing_ok=True)
    if spec.method == "server" and settings.idle_check is not None:
        waited = settings.idle_check(settings.server_url)
        (directory / IDLE_WAIT_FILE).write_text(f"{waited:.3f}\n")
    if settings.server_metadata is not None:
        try:
            metadata = settings.server_metadata(settings.server_url)
        except (httpx.HTTPError, ValueError, KeyError) as error:
            LOGGER.warning("run %s: no server metadata: %s", run, error)
        else:
            (directory / SERVER_METADATA_FILE).write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
    remote = f"{settings.remote_dir}/{run}"
    machines = [
        hosts.generators[i % len(hosts.generators)] for i in range(generators)
    ]
    for host in {hosts.server, *machines}:
        runner.prepare(host, remote)
    monitors: list[tuple[str, str, Job]] = []
    if settings.monitors:
        monitor_files = [("server", hosts.server, SERVER_MONITOR_FILE)]
        monitor_files += [
            ("generator", host, generator_monitor_file(index))
            for index, host in enumerate(machines)
        ]
        for role, host, name in monitor_files:
            command = monitor_command(
                settings.python, run, role, f"{remote}/{name}"
            )
            monitors.append((host, name, runner.launch(host, command)))
    start_at = clock() + settings.start_delay_s
    jobs = []
    for index, host in enumerate(machines):
        command = generator_command(
            settings.python,
            spec,
            run,
            spec.users,
            generators,
            index,
            start_at,
            settings.server_url,
            settings.s3_root,
            settings.s3_endpoint_url,
            f"{remote}/{requests_file(index)}",
            repeated,
            settings.processes,
            settings.warmup_requests,
            settings.measured_requests,
            settings.time_limit_s,
            settings.duration_s,
        )
        jobs.append(runner.launch(host, command))
    limit = settings.start_delay_s + settings.time_limit_s + REQUEST_TIMEOUT_S
    statuses = []
    for job in jobs:
        try:
            statuses.append(job.wait(limit + STOP_GRACE_S))
        except subprocess.TimeoutExpired:
            job.stop()
            statuses.append(-1)
    for _, _, job in monitors:
        job.stop()
    for _, _, job in monitors:
        with contextlib.suppress(subprocess.TimeoutExpired):
            job.wait(STOP_GRACE_S)
    for index, host in enumerate(machines):
        runner.fetch(
            host,
            f"{remote}/{requests_file(index)}",
            directory / requests_file(index),
        )
    for host, name, _ in monitors:
        runner.fetch(host, f"{remote}/{name}", directory / name)
    if any(status != 0 for status in statuses):
        raise RuntimeError(f"run {run}: generator exit statuses {statuses}")
    peak = peak_receive_rate(
        directory / generator_monitor_file(index) for index in range(generators)
    )
    return RunResult(
        spec=spec,
        run=run,
        directory=directory,
        generators=generators,
        users_per_generator=max(
            users_on(spec.users, generators, i) for i in range(generators)
        ),
        peak_rx_bytes_per_s=peak,
        repeated=repeated,
    )


def execute_spec(
    spec: RunSpec,
    settings: Settings,
    runner: Runner,
    clock: Callable[[], float] = time.time,
) -> list[RunResult]:
    """Run one cell, and repeat it with half the users per generator if needed.

    A load generator whose peak receive rate exceeds half of
    `LINK_BYTES_PER_S` may have limited the run. The repeat uses half the
    users per generator, on twice the generators. When the hosts file holds too
    few generators for that, the shortfall is logged and only the first run
    stays. A run that stays with such a peak gets a `LINK_LIMITED_FILE`, and
    so does a repeat whose own peak is still that high. A repeat directory of
    an earlier pass moves out of the results first with `_discard`.
    """
    _discard(settings, f"{spec.name}-repeat")
    available = len(settings.hosts.generators)
    generators, per_generator = split_generators(spec.users, available)
    needed = math.ceil(spec.users / USERS_PER_GENERATOR)
    if generators < needed:
        LOGGER.warning(
            "run %s: needs %d generators, hosts file holds %d; using %d with "
            "up to %d users each",
            spec.name,
            needed,
            available,
            generators,
            per_generator,
        )
    first = execute_run(
        spec, spec.name, generators, settings, runner, False, clock
    )
    results = [first]
    peak = first.peak_rx_bytes_per_s
    if peak is None or peak <= LINK_BYTES_PER_S / 2:
        return results
    half = math.ceil(first.users_per_generator / 2)
    wanted = math.ceil(spec.users / half)
    if half == first.users_per_generator or wanted > available:
        LOGGER.warning(
            "run %s: peak receive rate %.3g B/s exceeds half the link; a "
            "repeat with %d users per generator needs %d generators and %d "
            "are available; keeping the first run",
            spec.name,
            peak,
            half,
            wanted,
            available,
        )
        (first.directory / LINK_LIMITED_FILE).touch()
        return results
    LOGGER.info(
        "run %s: repeating with %d users per generator", spec.name, half
    )
    repeat = execute_run(
        spec, f"{spec.name}-repeat", wanted, settings, runner, True, clock
    )
    results.append(repeat)
    if (
        repeat.peak_rx_bytes_per_s is not None
        and repeat.peak_rx_bytes_per_s > LINK_BYTES_PER_S / 2
    ):
        (repeat.directory / LINK_LIMITED_FILE).touch()
    return results


def wait_for_server(url: str, timeout: float = HEALTH_TIMEOUT_S) -> None:
    """Wait until the server answers `/health`.

    Raises:
        TimeoutError: The server does not answer within timeout seconds.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            if httpx.get(f"{url}/health", timeout=10.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"server at {url}: no answer in {timeout} s")
        time.sleep(2.0)


def fetch_server_metadata(url: str) -> dict[str, str]:
    """Return the run metadata that the server reports at `/health`.

    Raises:
        httpx.HTTPError: The request fails or the server answers an error.
        ValueError: The answer is not JSON.
        KeyError: The answer holds no `metadata` object.
    """
    response = httpx.get(f"{url}/health", timeout=10.0)
    response.raise_for_status()
    return {
        str(key): str(value)
        for key, value in response.json()["metadata"].items()
    }


def wait_for_idle(
    url: str,
    timeout: float = IDLE_TIMEOUT_S,
    poll_s: float = IDLE_POLL_S,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Wait until `/health` reports no task in flight.

    A request that the client gave up on leaves its running tasks behind, and
    the next run would count them.

    Returns:
        Seconds spent waiting, 0.0 when the server was already idle.

    Raises:
        TimeoutError: The server still has tasks in flight after timeout
            seconds.
    """
    started = time.monotonic()
    logged = False
    while True:
        try:
            response = httpx.get(f"{url}/health", timeout=10.0)
            in_flight = int(response.json()["in_flight"])
        except (httpx.HTTPError, ValueError, KeyError):
            in_flight = -1
        waited = time.monotonic() - started
        if in_flight == 0:
            if logged:
                LOGGER.info("server idle after %.1f s", waited)
            return waited
        if waited > timeout:
            raise TimeoutError(
                f"server at {url}: {in_flight} tasks in flight after "
                f"{timeout} s"
            )
        if not logged:
            LOGGER.info("waiting for the server: %d tasks in flight", in_flight)
            logged = True
        sleep(poll_s)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m resampling_benchmark.load.orchestrate",
        description="Run the resampling benchmark grid over the hosts of a hosts file.",
    )
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument(
        "--hosts", type=Path, help="file of `role private_ip` lines"
    )
    where.add_argument(
        "--local",
        action="store_true",
        help="run generators and monitors as local subprocesses",
    )
    parser.add_argument("--server-url", help="default http://<server>:8000")
    parser.add_argument("--s3-root", required=True)
    parser.add_argument("--s3-endpoint-url")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / RESAMPLING_RAW_DIRECTORY,
        help=f"directory of the run directories (default results/{RESAMPLING_RAW_DIRECTORY})",
    )
    parser.add_argument(
        "--remote-dir",
        default="resampling_benchmark-out",
        help="directory on each host, relative to the login home",
    )
    parser.add_argument(
        "--python",
        default=REMOTE_PYTHON,
        help=f"Python on each host (default {REMOTE_PYTHON}); --local uses this one",
    )
    parser.add_argument("--processes", type=int)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--recordings", default=",".join(RECORDINGS))
    parser.add_argument("--windows", help="comma-separated seconds")
    parser.add_argument("--users", help="comma-separated user counts")
    parser.add_argument("--warmup-requests", type=int, default=WARMUP_REQUESTS)
    parser.add_argument(
        "--measured-requests", type=int, default=MEASURED_REQUESTS
    )
    parser.add_argument("--time-limit-s", type=float, default=RUN_TIME_LIMIT_S)
    parser.add_argument("--start-delay-s", type=float, default=START_DELAY_S)
    parser.add_argument(
        "--duration-s",
        type=float,
        help="recording length in seconds, for a recording that is not R1 to R4",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help=f"skip runs whose directory holds a `{COMPLETE_FILE}` file",
    )
    return parser.parse_args(argv)


def _numbers(
    text: str | None, default: Sequence[int] | None
) -> list[int] | None:
    if text is None:
        return None if default is None else list(default)
    return [int(part) for part in text.split(",")]


def _staying_directory(spec: RunSpec, settings: Settings) -> Path:
    """Return the directory of the run of a cell that stays in the results.

    That is the repeat when the cell has one, and the first run otherwise.
    """
    repeat = settings.out / f"{spec.name}-repeat"
    return (
        repeat
        if (repeat / COMPLETE_FILE).exists()
        else settings.out / spec.name
    )


def _median_failed(directory: Path) -> bool:
    median = median_charged_seconds(sorted(directory.glob(REQUESTS_PATTERN)))
    return median is not None and median >= REQUEST_TIMEOUT_S


def _discard(settings: Settings, name: str) -> None:
    """Move the directory of a run out of the results, if it exists.

    The directory goes to `<out>-discarded/<name>-<timestamp>`, a sibling of
    the results directory, so `resampling_benchmark.summarize` no longer reads
    it and no data is deleted.
    """
    source = settings.out / name
    if not source.exists():
        return
    stamp = time.strftime("%Y%m%dT%H%M%S")
    root = settings.out.with_name(f"{settings.out.name}-discarded")
    target = root / f"{name}-{stamp}"
    count = 0
    while target.exists():
        count += 1
        target = root / f"{name}-{stamp}-{count}"
    root.mkdir(parents=True, exist_ok=True)
    shutil.move(source, target)
    LOGGER.warning("run %s: moved the earlier files to %s", name, target)


def _mark_not_run(spec: RunSpec, settings: Settings, failed: str) -> None:
    """Write the directory of a skipped run, with the name of the failed run.

    A directory of an earlier partial run, and its repeat directory, move out
    of the results with `_discard`. A directory that holds only a
    `NOT_RUN_FILE` is rewritten in place.
    """
    for name in (spec.name, f"{spec.name}-repeat"):
        held = {path.name for path in (settings.out / name).glob("*")}
        if held - {NOT_RUN_FILE}:
            _discard(settings, name)
    directory = settings.out / spec.name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / NOT_RUN_FILE).write_text(f"{failed}\n")


def run_grid(
    runs: Iterable[RunSpec],
    settings: Settings,
    runner: Runner,
    skip_done: bool = False,
) -> int:
    """Run every cell and return the number that failed.

    A cell that finishes without error gets a `COMPLETE_FILE` in each run
    directory. With `skip_done`, a cell whose run that stays in the results
    holds that file is skipped. Only complete runs are skipped.

    When the median charged request seconds of the run that stays in the
    results reach `REQUEST_TIMEOUT_S`, every cell of the same method,
    recording, and window with more users is not run. Its directory gets a
    `NOT_RUN_FILE` that holds the name of the failed run. A complete run is
    judged first, and a complete run above a failed run is kept as it is. A
    partial run above a failed run moves out of the results with `_discard`.
    The not-run cells come from the runs judged in this pass.
    """
    failures = 0
    failed_runs: dict[tuple[str, str, int], tuple[int, str]] = {}
    for spec in runs:
        cell = (spec.method, spec.recording, spec.window_s)
        above = cell in failed_runs and spec.users > failed_runs[cell][0]
        staying = _staying_directory(spec, settings)
        done = (staying / COMPLETE_FILE).exists()
        if done and (skip_done or above):
            if above:
                LOGGER.warning(
                    "run %s: a completed run above the failed run %s is kept",
                    spec.name,
                    failed_runs[cell][1],
                )
            else:
                LOGGER.info("run %s: done, skipped", spec.name)
                if _median_failed(staying):
                    failed_runs[cell] = (spec.users, staying.name)
            continue
        if above:
            failed = failed_runs[cell][1]
            LOGGER.info("run %s: not run, %s failed", spec.name, failed)
            _mark_not_run(spec, settings, failed)
            continue
        try:
            results = execute_spec(spec, settings, runner)
        except (RuntimeError, TimeoutError) as error:
            failures += 1
            LOGGER.error("%s", error)
            continue
        for result in results:
            (result.directory / COMPLETE_FILE).touch()
        if _median_failed(results[-1].directory):
            failed_runs[cell] = (spec.users, results[-1].directory.name)
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected grid and return the exit status."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    arguments = parse_arguments(argv)
    if arguments.local:
        hosts = Hosts("localhost", ("localhost",))
        runner: Runner = LocalRunner()
        server_url = arguments.server_url or "http://127.0.0.1:8000"
        monitors = Path("/proc/stat").exists()
    else:
        hosts = parse_hosts(arguments.hosts.read_text())
        runner = SshRunner()
        server_url = arguments.server_url or f"http://{hosts.server}:8000"
        monitors = True
    settings = Settings(
        hosts=hosts,
        server_url=server_url,
        s3_root=arguments.s3_root,
        s3_endpoint_url=arguments.s3_endpoint_url,
        out=arguments.out,
        remote_dir=(
            str(arguments.out.resolve() / "remote")
            if arguments.local
            else arguments.remote_dir
        ),
        python=arguments.python if not arguments.local else sys.executable,
        monitors=monitors,
        processes=arguments.processes,
        warmup_requests=arguments.warmup_requests,
        measured_requests=arguments.measured_requests,
        time_limit_s=arguments.time_limit_s,
        start_delay_s=arguments.start_delay_s,
        duration_s=arguments.duration_s,
        idle_check=wait_for_idle,
        server_metadata=fetch_server_metadata,
    )
    runs = plan_runs(
        methods=arguments.methods.split(","),
        recordings=arguments.recordings.split(","),
        windows=_numbers(arguments.windows, WINDOWS_S) or WINDOWS_S,
        users=_numbers(arguments.users, None),
    )
    wait_for_server(server_url)
    failures = run_grid(runs, settings, runner, arguments.skip_done)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
