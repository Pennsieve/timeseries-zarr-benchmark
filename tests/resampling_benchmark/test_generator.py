import contextlib
import http.server
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path

import httpx
import pytest

from resampling_benchmark.load import generator
from resampling_benchmark.load.generator import (
    GeneratorConfig,
    Outcome,
    RequestError,
    ServerRequester,
    error_detail,
    format_window,
    run_generator,
    run_user,
    split_users,
    user_rng,
)
from tests.resampling_benchmark.conftest import TINY_ID, TINY_SPEC
from tszbench.csvmeta import read_header
from tszbench.fs import StoreRoots
from tszbench.rangeserver import free_port
from tszbench.schemas import RESAMPLING_REQUESTS

DURATION_US = TINY_SPEC.duration_us
WINDOW_S = 10.0


def config(**changes: object) -> GeneratorConfig:
    fields: dict[str, object] = {
        "method": "server",
        "recording": TINY_ID,
        "window_s": WINDOW_S,
        "users": 2,
        "total_users": 2,
        "generators": 1,
        "generator_index": 0,
        "run": "unit-run",
        "start_at": time.time() - 1.0,
        "server_url": None,
        "s3_root": None,
        "s3_endpoint_url": None,
        "duration_us": DURATION_US,
        "warmup_requests": 1,
        "measured_requests": 3,
    }
    fields.update(changes)
    return GeneratorConfig(**fields)  # type: ignore[arg-type]


class ScriptedRequester:
    """Returns each scripted outcome or raises it, then repeats the last."""

    def __init__(self, script: list[Outcome | Exception]) -> None:
        self.script = script
        self.windows: list[tuple[int, int]] = []
        self.closed = False

    def request(self, start_us: int, end_us: int) -> Outcome:
        self.windows.append((start_us, end_us))
        item = self.script[min(len(self.windows), len(self.script)) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


OK = Outcome(
    bytes=100,
    s3_requests=4,
    s3_bytes=4000,
    level=2,
    read_seconds=0.5,
    reduce_seconds=0.25,
)


def collect(
    settings: GeneratorConfig, requester: ScriptedRequester, user: int = 0
) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    run_user(settings, user, requester, rows.append)
    return rows


def test_a_user_sends_one_warmup_then_the_measured_requests() -> None:
    rows = collect(config(), ScriptedRequester([OK]))
    assert [row["sequence"] for row in rows] == [0, 1, 2, 3]
    assert [row["warmup"] for row in rows] == [
        "true",
        "false",
        "false",
        "false",
    ]
    assert {row["status"] for row in rows} == {"ok"}
    assert rows[0]["level"] == 2
    assert rows[0]["s3_requests"] == 4
    assert rows[0]["s3_bytes"] == 4000
    assert rows[0]["bytes"] == 100
    assert set(rows[0]) == set(RESAMPLING_REQUESTS)


def test_starts_are_whole_microseconds_inside_the_recording() -> None:
    requester = ScriptedRequester([OK])
    rows = collect(config(measured_requests=40), requester)
    window_us = round(WINDOW_S * 1_000_000)
    for (start, end), row in zip(requester.windows, rows, strict=True):
        assert 0 <= start <= DURATION_US - window_us
        assert end - start == window_us
        assert row["start_us"] == start
    assert len({start for start, _ in requester.windows}) > 30


def test_the_same_run_generator_and_user_pick_the_same_starts() -> None:
    first = ScriptedRequester([OK])
    second = ScriptedRequester([OK])
    other = ScriptedRequester([OK])
    collect(config(), first, user=3)
    collect(config(), second, user=3)
    collect(config(), other, user=4)
    assert first.windows == second.windows
    assert first.windows != other.windows


def test_the_seed_depends_on_run_generator_and_user() -> None:
    draws = {
        tuple(user_rng(run, gen, user).integers(0, 10**9, 3))
        for run in ("a", "b")
        for gen in (0, 1)
        for user in (0, 1)
    }
    assert len(draws) == 8


def test_a_window_that_fills_the_recording_always_starts_at_zero() -> None:
    requester = ScriptedRequester([OK])
    collect(config(window_s=120.0), requester)
    assert {start for start, _ in requester.windows} == {0}


def test_dispatch_seconds_count_from_the_start_of_the_run() -> None:
    before = time.time()
    rows = collect(config(start_at=before - 5.0), ScriptedRequester([OK]))
    dispatches = [float(str(row["dispatch_s"])) for row in rows]
    assert dispatches == sorted(dispatches)
    assert 5.0 <= dispatches[0] < 6.0


def test_a_user_sends_nothing_before_the_start_time() -> None:
    start_at = time.time() + 0.3
    rows = collect(
        config(start_at=start_at, measured_requests=0), ScriptedRequester([OK])
    )
    assert float(str(rows[0]["dispatch_s"])) >= 0.0


def test_a_timeout_an_error_and_a_server_error_are_recorded_and_the_user_continues() -> (
    None
):
    script: list[Outcome | Exception] = [
        httpx.ReadTimeout("slow"),
        ValueError("bad"),
        RequestError("ClientError"),
        OK,
    ]
    rows = collect(config(), ScriptedRequester(script))
    assert [row["status"] for row in rows] == [
        "timeout",
        "error:ValueError",
        "error:ClientError",
        "ok",
    ]
    assert rows[0]["bytes"] == 0
    assert rows[0]["level"] == ""


def test_a_response_slower_than_the_timeout_is_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generator, "REQUEST_TIMEOUT_S", 0.0)
    rows = collect(config(measured_requests=0), ScriptedRequester([OK]))
    assert rows[0]["status"] == "timeout"


def test_no_request_starts_after_the_time_limit() -> None:
    requester = ScriptedRequester([OK])
    rows = collect(
        config(start_at=time.time() - 10.0, time_limit_s=5.0), requester
    )
    assert rows == []
    assert requester.windows == []


def test_error_detail_uses_a_class_name_body_and_otherwise_the_status() -> None:
    class_name = httpx.Response(503, text="SlowDownError")
    message = httpx.Response(404, text="unknown recording R9")
    empty = httpx.Response(500, text="")
    assert error_detail(class_name) == "SlowDownError"
    assert error_detail(message) == "HTTP404"
    assert error_detail(empty) == "HTTP500"


def test_users_are_dealt_to_processes_in_turn() -> None:
    assert split_users(5, 2) == [[0, 2, 4], [1, 3]]
    assert split_users(2, 8) == [[0], [1]]
    assert split_users(3, 1) == [[0, 1, 2]]


def test_window_lengths_print_without_a_trailing_zero() -> None:
    assert format_window(10.0) == "10"
    assert format_window(0.5) == "0.5"


def test_arguments_resolve_the_recording_length_or_take_it_from_the_command() -> (
    None
):
    base = [
        "--method",
        "server",
        "--window-s",
        "10",
        "--users",
        "3",
        "--run",
        "r",
        "--out",
        "x.csv",
        "--server-url",
        "http://h",
    ]
    resolved = generator.config_from_arguments(
        generator.parse_arguments([*base, "--recording", "R3-1h"])
    )
    assert resolved.duration_us == 3_600_000_000
    assert resolved.total_users == 3
    given = generator.config_from_arguments(
        generator.parse_arguments(
            [*base, "--recording", "T1", "--duration-s", "120"]
        )
    )
    assert given.duration_us == 120_000_000
    with pytest.raises(ValueError, match="must be one of"):
        generator.config_from_arguments(
            generator.parse_arguments([*base, "--recording", "R9"])
        )
    with pytest.raises(ValueError, match="window of"):
        generator.config_from_arguments(
            generator.parse_arguments(
                [
                    *base[:3],
                    "500",
                    *base[4:],
                    "--recording",
                    "T1",
                    "--duration-s",
                    "120",
                ]
            )
        )


def read_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open() as file:
        lines = [line for line in file if not line.startswith("# ")]
    return list(csv.DictReader(lines))


@pytest.mark.network
def test_users_of_the_server_method_write_rows_against_a_live_server(
    server_url: str, tmp_path: Path
) -> None:
    out = tmp_path / "requests-0.csv"
    settings = config(server_url=server_url, total_users=2)
    written = run_generator(settings, out)
    rows = read_rows(out)
    assert written == len(rows) == 2 * 4
    assert {row["status"] for row in rows} == {"ok"}
    assert {row["method"] for row in rows} == {"server"}
    assert {row["users"] for row in rows} == {"2"}
    for user in ("0", "1"):
        mine = [row for row in rows if row["user"] == user]
        assert [row["warmup"] for row in mine] == [
            "true",
            "false",
            "false",
            "false",
        ]
        assert [row["sequence"] for row in mine] == ["0", "1", "2", "3"]
    assert {row["level"] for row in rows} == {"0"}
    assert all(int(row["s3_requests"]) > 0 for row in rows)
    assert all(int(row["bytes"]) == 4 * 2000 * 2 * 4 for row in rows)
    assert all(float(row["read_seconds"]) > 0 for row in rows)
    header = read_header(out)
    assert header["run"] == "unit-run"
    assert header["repeated"] == "false"
    assert float(header["start_at"]) == settings.start_at


@pytest.mark.network
def test_users_of_the_precomputed_method_write_rows_against_moto(
    tiny_roots: StoreRoots, tmp_path: Path
) -> None:
    out = tmp_path / "requests-0.csv"
    settings = config(
        method="precomputed",
        s3_root=tiny_roots.s3,
        s3_endpoint_url=tiny_roots.s3_endpoint_url,
        repeated=True,
    )
    run_generator(settings, out)
    rows = read_rows(out)
    assert len(rows) == 8
    assert {row["status"] for row in rows} == {"ok"}
    assert {row["level"] for row in rows} == {"1"}
    for row in rows:
        assert int(row["bytes"]) == int(row["s3_bytes"]) > 0
        assert int(row["s3_requests"]) > 0
    assert read_header(out)["repeated"] == "true"


@pytest.mark.network
def test_an_unreachable_server_gives_error_rows_and_the_users_finish(
    tmp_path: Path,
) -> None:
    out = tmp_path / "requests-0.csv"
    run_generator(config(server_url="http://127.0.0.1:9", users=1), out)
    rows = read_rows(out)
    assert len(rows) == 4
    assert {row["status"] for row in rows} == {"error:ConnectError"}


@pytest.mark.network
def test_an_unknown_recording_gives_http_status_errors(
    server_url: str, tmp_path: Path
) -> None:
    out = tmp_path / "requests-0.csv"
    run_generator(config(server_url=server_url, users=1, recording="R9"), out)
    assert {row["status"] for row in read_rows(out)} == {"error:HTTP404"}


@pytest.mark.network
def test_the_command_line_runs_users_across_processes(
    server_url: str, tmp_path: Path
) -> None:
    out = tmp_path / "requests-0.csv"
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "resampling_benchmark.load.generator",
            "--method",
            "server",
            "--recording",
            TINY_ID,
            "--duration-s",
            "120",
            "--window-s",
            "10",
            "--users",
            "3",
            "--processes",
            "2",
            "--run",
            "cli-run",
            "--server-url",
            server_url,
            "--measured-requests",
            "2",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    rows = read_rows(out)
    assert len(rows) == 3 * 3
    assert {row["status"] for row in rows} == {"ok"}
    assert {row["user"] for row in rows} == {"0", "1", "2"}


def spawned_children(parent: int) -> list[int]:
    """Return the ids of the `multiprocessing` spawn children of a process."""
    listing = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,command="],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = []
    for line in listing.splitlines():
        pid, ppid, command = line.split(None, 2)
        if int(ppid) == parent and "multiprocessing.spawn" in command:
            found.append(int(pid))
    return found


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def stalled_server() -> Iterator[str]:
    """Serve a port whose requests do not answer until the test ends."""
    release = threading.Event()

    class Stall(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            release.wait(120)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), Stall)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    release.set()
    server.shutdown()
    server.server_close()


@contextlib.contextmanager
def closing_server(reset: bool) -> Iterator[str]:
    """Serve a port that reads each request and drops the connection unanswered.

    With `reset`, the connection ends with a TCP reset in place of an orderly close.
    """
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()

    def serve() -> None:
        while True:
            try:
                connection, _ = listener.accept()
            except OSError:
                return
            connection.recv(65536)
            if reset:
                connection.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                )
            connection.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    listener.close()


@pytest.mark.network
@pytest.mark.parametrize(
    ("reset", "status"),
    [(False, "error:RemoteProtocolError"), (True, "error:ReadError")],
)
def test_a_connection_closed_mid_request_is_recorded_by_its_error_class(
    reset: bool, status: str
) -> None:
    with closing_server(reset) as url:
        client = httpx.Client(base_url=url, timeout=5.0)
        rows: list[Mapping[str, object]] = []
        run_user(
            config(measured_requests=2),
            0,
            ServerRequester(client, TINY_ID),
            rows.append,
        )
        client.close()
    assert [row["status"] for row in rows] == [status] * 3
    assert all(float(str(row["seconds"])) < 5.0 for row in rows)


@pytest.mark.network
@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX signals")
def test_stopping_a_generator_stops_its_user_processes(
    stalled_server: str, tmp_path: Path
) -> None:
    generator_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "resampling_benchmark.load.generator",
            "--method",
            "server",
            "--recording",
            TINY_ID,
            "--duration-s",
            "120",
            "--window-s",
            "10",
            "--users",
            "2",
            "--processes",
            "2",
            "--run",
            "stop-run",
            "--server-url",
            stalled_server,
            "--out",
            str(tmp_path / "requests-0.csv"),
        ]
    )
    children: list[int] = []
    try:
        deadline = time.monotonic() + 60
        while len(children) < 2:
            assert time.monotonic() < deadline, "user processes did not start"
            assert generator_process.poll() is None
            children = spawned_children(generator_process.pid)
            time.sleep(0.2)
        time.sleep(2.0)
        generator_process.send_signal(signal.SIGTERM)
        generator_process.wait(timeout=30)
        deadline = time.monotonic() + 15
        while any(is_running(pid) for pid in children):
            assert time.monotonic() < deadline, "a user process kept running"
            time.sleep(0.2)
    finally:
        generator_process.kill()
        for pid in children:
            if is_running(pid):
                os.kill(pid, signal.SIGKILL)
