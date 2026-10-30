"""Simulated users of the resampling benchmark: `python -m resampling_benchmark.load.generator`.

Each user is a closed loop. It sends a request at a random start, waits for the
decoded array, and sends the next request.
"""

import argparse
import multiprocessing
import os
import re
import signal
import threading
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

import httpx
import numpy as np
import numpy.typing as npt

from resampling_benchmark.load.precomputed import (
    READ_THREADS,
    PrecomputedReader,
)
from resampling_benchmark.service.store import open_recording
from tszbench.csvmeta import CsvWriter, run_metadata
from tszbench.fs import StoreRoots, make_filesystem
from tszbench.recordings import resolve
from tszbench.schemas import METHODS, RESAMPLING_REQUESTS

WARMUP_REQUESTS = 1
"""Requests each user sends first and the summary leaves out."""

MEASURED_REQUESTS = 10
"""Requests each user sends after the warm-up."""

RUN_TIME_LIMIT_S = 300.0
"""Seconds after the start of a run when users stop sending requests."""

REQUEST_TIMEOUT_S = 180.0
"""Seconds after which a request counts as a timeout."""

LINK_BYTES_PER_S = 12.5e9 / 8
"""Receive rate of a load generator link of 12.5 Gbps, in bytes per second."""

STATUS_OK = "ok"
STATUS_TIMEOUT = "timeout"
MAX_ERROR_DETAIL = 48

CHILD_JOIN_TIMEOUT_S = 5.0
"""Seconds a stopped user process has to exit before it is killed."""

Row = Mapping[str, object]


@dataclass(frozen=True)
class GeneratorConfig:
    """Everything one load generator needs.

    Attributes:
        method: `server` or `precomputed`.
        recording: Recording id.
        window_s: Window length in seconds.
        users: Users that run on this generator.
        total_users: Users of the whole run, written to every row.
        generators: Generators of the whole run.
        generator_index: Index of this generator.
        run: Run id. It also seeds the window starts.
        start_at: Unix time when every user sends its first request.
        server_url: Base URL of the server, for the `server` method.
        s3_root: `bucket/prefix`, for the `precomputed` method.
        s3_endpoint_url: S3-compatible endpoint, or None for AWS.
        duration_us: Length of the recording in microseconds.
        processes: Processes the users are split across.
        warmup_requests: Warm-up requests of each user.
        measured_requests: Measured requests of each user.
        time_limit_s: Seconds after `start_at` when new requests stop.
        repeated: True for a run repeated with half the users per generator.
    """

    method: str
    recording: str
    window_s: float
    users: int
    total_users: int
    generators: int
    generator_index: int
    run: str
    start_at: float
    server_url: str | None
    s3_root: str | None
    s3_endpoint_url: str | None
    duration_us: int
    processes: int = 1
    warmup_requests: int = WARMUP_REQUESTS
    measured_requests: int = MEASURED_REQUESTS
    time_limit_s: float = RUN_TIME_LIMIT_S
    repeated: bool = False

    @property
    def window_us(self) -> int:
        """Window length in whole microseconds."""
        return round(self.window_s * 1_000_000)


@dataclass(frozen=True)
class Outcome:
    """What one successful request returned.

    Attributes:
        bytes: Bytes the user received.
        s3_requests: Requests sent to S3 for the answer.
        s3_bytes: Payload bytes read from S3 for the answer.
        level: Level read, 0 for raw.
        read_seconds: Seconds spent reading, summed over tasks.
        reduce_seconds: Seconds spent reducing, summed over tasks.
    """

    bytes: int
    s3_requests: int
    s3_bytes: int
    level: int
    read_seconds: float
    reduce_seconds: float


class Requester(Protocol):
    """Answers one window request and decodes the array."""

    def request(self, start_us: int, end_us: int) -> Outcome:
        """Return the outcome of one request."""
        ...

    def close(self) -> None:
        """Release the connections."""
        ...


class RequestError(Exception):
    """A response with an error status."""

    def __init__(self, detail: str) -> None:
        """Keep the detail that follows `error:` in the status."""
        super().__init__(detail)
        self.detail = detail


class ServerRequester:
    """Sends requests to the resampling server."""

    def __init__(
        self, client: httpx.Client, recording: str, close_client: bool = False
    ) -> None:
        """Bind to a client, which may be shared by many users."""
        self._client = client
        self._recording = recording
        self._close_client = close_client

    def get(
        self, start_us: int, end_us: int
    ) -> tuple[npt.NDArray[np.float32], Mapping[str, str]]:
        """Fetch the minmax of a window.

        Returns:
            The decoded array of shape (channels, rows, 2) and the response
            headers.

        Raises:
            RequestError: The server answered with an error status.
        """
        response = self._client.get(
            "/minmax",
            params={
                "recording": self._recording,
                "start_us": start_us,
                "end_us": end_us,
            },
        )
        if response.status_code != httpx.codes.OK:
            raise RequestError(error_detail(response))
        headers = response.headers
        array = np.frombuffer(response.content, dtype=np.float32).reshape(
            int(headers["X-Channels"]), int(headers["X-Rows"]), 2
        )
        return array, headers

    def fetch(self, start_us: int, end_us: int) -> npt.NDArray[np.float32]:
        """Return the decoded minmax of a window."""
        return self.get(start_us, end_us)[0]

    def request(self, start_us: int, end_us: int) -> Outcome:
        """Fetch and decode the minmax of a window."""
        array, headers = self.get(start_us, end_us)
        return Outcome(
            bytes=array.nbytes,
            s3_requests=int(headers["X-S3-Requests"]),
            s3_bytes=int(headers["X-S3-Bytes"]),
            level=int(headers["X-Level"]),
            read_seconds=float(headers["X-Read-Seconds"]),
            reduce_seconds=float(headers["X-Reduce-Seconds"]),
        )

    def close(self) -> None:
        """Close the client when this requester owns it."""
        if self._close_client:
            self._client.close()


class PrecomputedRequester:
    """Reads the levels of the bundle from S3 and reduces them locally."""

    def __init__(self, reader: PrecomputedReader, close: Callable[[], None]):
        """Bind to a reader and the function that releases its resources."""
        self._reader = reader
        self._close = close

    def request(self, start_us: int, end_us: int) -> Outcome:
        """Read and reduce the window. `bytes` is what S3 returned."""
        result = self._reader.read(start_us, end_us)
        return Outcome(
            bytes=result.counts.bytes,
            s3_requests=result.counts.requests,
            s3_bytes=result.counts.bytes,
            level=result.level,
            read_seconds=result.read_seconds,
            reduce_seconds=result.reduce_seconds,
        )

    def close(self) -> None:
        """Close the filesystem and the bundle."""
        self._close()


def error_detail(response: httpx.Response) -> str:
    """Return the error class a server response names.

    The server puts the class name in the body of a 5xx. Any other body, or
    none, gives `HTTP<status>`.
    """
    text = response.text.strip()
    if re.fullmatch(rf"\w{{1,{MAX_ERROR_DETAIL}}}", text):
        return text
    return f"HTTP{response.status_code}"


def format_window(window_s: float) -> str:
    """Return a window length without a trailing `.0`."""
    return str(int(window_s)) if window_s == int(window_s) else str(window_s)


def user_rng(run: str, generator: int, user: int) -> np.random.Generator:
    """Return the random generator of one user of one run."""
    return np.random.default_rng([zlib.crc32(run.encode()), generator, user])


def _blank_row(
    config: GeneratorConfig, user: int, sequence: int, warmup: bool
) -> dict[str, object]:
    return {
        "run": config.run,
        "method": config.method,
        "recording": config.recording,
        "window_s": format_window(config.window_s),
        "users": config.total_users,
        "generator": config.generator_index,
        "user": user,
        "sequence": sequence,
        "start_us": "",
        "dispatch_s": "",
        "seconds": "",
        "bytes": 0,
        "s3_requests": 0,
        "s3_bytes": 0,
        "status": "",
        "warmup": "true" if warmup else "false",
        "level": "",
        "read_seconds": "",
        "reduce_seconds": "",
    }


def run_user(
    config: GeneratorConfig,
    user: int,
    requester: Requester,
    emit: Callable[[Row], None],
) -> None:
    """Send the warm-up and measured requests of one user.

    Waits for `start_at`, then sends requests one after the other until the
    user is done or `time_limit_s` has passed since `start_at`. A failed
    request is recorded and the user continues.
    """
    rng = user_rng(config.run, config.generator_index, user)
    span = config.duration_us - config.window_us
    total = config.warmup_requests + config.measured_requests
    wait = config.start_at - time.time()
    if wait > 0:
        time.sleep(wait)
    for sequence in range(total):
        if time.time() - config.start_at >= config.time_limit_s:
            return
        start_us = int(rng.integers(0, span + 1))
        row = _blank_row(
            config, user, sequence, sequence < config.warmup_requests
        )
        row["start_us"] = start_us
        dispatched = time.time()
        row["dispatch_s"] = dispatched - config.start_at
        began = time.perf_counter()
        try:
            outcome = requester.request(start_us, start_us + config.window_us)
        except httpx.TimeoutException:
            row["status"] = STATUS_TIMEOUT
        except RequestError as error:
            row["status"] = f"error:{error.detail}"
        except Exception as error:
            row["status"] = f"error:{type(error).__name__}"
        else:
            row.update(
                bytes=outcome.bytes,
                s3_requests=outcome.s3_requests,
                s3_bytes=outcome.s3_bytes,
                level=outcome.level,
                read_seconds=outcome.read_seconds,
                reduce_seconds=outcome.reduce_seconds,
                status=STATUS_OK,
            )
        seconds = time.perf_counter() - began
        row["seconds"] = seconds
        if seconds > REQUEST_TIMEOUT_S:
            row["status"] = STATUS_TIMEOUT
        emit(row)


def run_users(
    config: GeneratorConfig, users: Sequence[int], emit: Callable[[Row], None]
) -> None:
    """Run `users` as threads of this process until every one is done.

    A user whose setup fails records one error row and stops.
    """
    pool: ThreadPoolExecutor | None = None
    client: httpx.Client | None = None
    if config.method == "server":
        if config.server_url is None:
            raise ValueError("method server needs a server URL")
        client = httpx.Client(
            base_url=config.server_url,
            timeout=REQUEST_TIMEOUT_S,
            limits=httpx.Limits(
                max_connections=len(users), max_keepalive_connections=len(users)
            ),
        )
    else:
        pool = ThreadPoolExecutor(max_workers=READ_THREADS)

    def make_requester() -> Requester:
        if client is not None:
            return ServerRequester(client, config.recording)
        if config.s3_root is None or pool is None:
            raise ValueError("method precomputed needs an S3 root")
        roots = StoreRoots(
            local=Path(),
            http="",
            s3=config.s3_root,
            s3_endpoint_url=config.s3_endpoint_url,
        )
        filesystem = make_filesystem("s3", roots)
        try:
            bundle = open_recording(filesystem, roots, config.recording)
        except BaseException:
            filesystem.close()
            raise

        def close() -> None:
            bundle.close()
            filesystem.close()

        return PrecomputedRequester(
            PrecomputedReader(bundle, filesystem, pool), close
        )

    def work(user: int) -> None:
        try:
            requester = make_requester()
        except Exception as error:
            row = _blank_row(config, user, 0, False)
            row["status"] = f"error:{type(error).__name__}"
            emit(row)
            return
        try:
            run_user(config, user, requester, emit)
        finally:
            requester.close()

    threads = [
        threading.Thread(target=work, args=(user,), name=f"user-{user}")
        for user in users
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if client is not None:
        client.close()
    if pool is not None:
        pool.shutdown()


def split_users(users: int, processes: int) -> list[list[int]]:
    """Return the user indices of each process, dealt out in turn.

    A process that would get no user is dropped.
    """
    count = max(1, min(processes, users))
    return [list(range(index, users, count)) for index in range(count)]


def _process_main(
    config: GeneratorConfig,
    users: Sequence[int],
    queue: "multiprocessing.Queue[Row | None]",
) -> None:
    try:
        run_users(config, users, queue.put)
    finally:
        queue.put(None)


def run_generator(config: GeneratorConfig, out: Path) -> int:
    """Run every user of this generator and write the request CSV.

    Returns:
        The number of rows written.
    """
    header = run_metadata(
        run=config.run,
        method=config.method,
        recording=config.recording,
        window_s=format_window(config.window_s),
        users=config.total_users,
        generators=config.generators,
        generator_index=config.generator_index,
        users_on_generator=config.users,
        processes=config.processes,
        repeated="true" if config.repeated else "false",
    )
    lock = threading.Lock()
    rows = 0
    with CsvWriter(out, RESAMPLING_REQUESTS, header) as writer:

        def emit(row: Row) -> None:
            nonlocal rows
            with lock:
                writer.write(row)
                rows += 1

        shares = split_users(config.users, config.processes)
        if len(shares) == 1:
            run_users(config, shares[0], emit)
            return rows
        _run_children(config, shares, emit)
    return rows


def _exit_on_signal(signum: int, frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


def _run_children(
    config: GeneratorConfig,
    shares: Sequence[Sequence[int]],
    emit: Callable[[Row], None],
) -> None:
    """Run each share of users in a new process and relay its rows to emit.

    SIGTERM and SIGINT stop the children before this process exits.
    """
    context = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue[Row | None] = context.Queue()
    children = [
        context.Process(target=_process_main, args=(config, share, queue))
        for share in shares
    ]
    previous: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, _exit_on_signal)
    try:
        for child in children:
            child.start()
        finished = 0
        while finished < len(children):
            row = queue.get()
            if row is None:
                finished += 1
            else:
                emit(row)
        for child in children:
            child.join()
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            if child.pid is None:
                continue
            child.join(CHILD_JOIN_TIMEOUT_S)
            if child.is_alive():
                child.kill()
                child.join()
        for number, handler in previous.items():
            signal.signal(number, handler)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m resampling_benchmark.load.generator",
        description="Run the simulated users of one load generator.",
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--window-s", type=float, required=True)
    parser.add_argument(
        "--users", type=int, required=True, help="users on this generator"
    )
    parser.add_argument(
        "--total-users",
        type=int,
        help="users of the whole run, default --users",
    )
    parser.add_argument("--generators", type=int, default=1)
    parser.add_argument("--generator-index", type=int, default=0)
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--start-at",
        type=float,
        default=0.0,
        help="Unix time of the first request, default now",
    )
    parser.add_argument("--server-url")
    parser.add_argument("--s3-root")
    parser.add_argument("--s3-endpoint-url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--warmup-requests", type=int, default=WARMUP_REQUESTS)
    parser.add_argument(
        "--measured-requests", type=int, default=MEASURED_REQUESTS
    )
    parser.add_argument("--time-limit-s", type=float, default=RUN_TIME_LIMIT_S)
    parser.add_argument("--repeated", action="store_true")
    return parser.parse_args(argv)


def config_from_arguments(arguments: argparse.Namespace) -> GeneratorConfig:
    """Build the configuration from parsed arguments.

    Raises:
        ValueError: The recording is unknown, or the window is longer than the
            recording, or a count is below 1.
    """
    duration_s = (
        arguments.duration_s
        if arguments.duration_s is not None
        else resolve(arguments.recording).duration_s
    )
    config = GeneratorConfig(
        method=arguments.method,
        recording=arguments.recording,
        window_s=arguments.window_s,
        users=arguments.users,
        total_users=arguments.total_users or arguments.users,
        generators=arguments.generators,
        generator_index=arguments.generator_index,
        run=arguments.run,
        start_at=arguments.start_at or time.time(),
        server_url=arguments.server_url,
        s3_root=arguments.s3_root,
        s3_endpoint_url=arguments.s3_endpoint_url,
        duration_us=round(duration_s * 1_000_000),
        processes=arguments.processes,
        warmup_requests=arguments.warmup_requests,
        measured_requests=arguments.measured_requests,
        time_limit_s=arguments.time_limit_s,
        repeated=arguments.repeated,
    )
    if config.users < 1 or config.processes < 1:
        raise ValueError("users and processes must be at least 1")
    if config.window_us > config.duration_us or config.window_us < 1:
        raise ValueError(
            f"window of {config.window_s} s: must be between 1 us and the "
            f"{config.duration_us} us of {config.recording}"
        )
    return config


def main(argv: Sequence[str] | None = None) -> None:
    """Run one load generator."""
    arguments = parse_arguments(argv)
    config = config_from_arguments(arguments)
    run_generator(config, arguments.out)


if __name__ == "__main__":
    main()
