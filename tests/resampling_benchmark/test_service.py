import functools
import os
import random
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from timeseries_zarr import open_bundle

from generate.tszarr import write_tszarr
from resampling_benchmark.service import engine as engine_module
from resampling_benchmark.service.app import (
    ENV_PROCESSES,
    ENV_RECORDINGS,
    ENV_S3_ENDPOINT_URL,
    ENV_S3_ROOT,
    ENV_TASK_THREADS,
    Settings,
    create_app,
    error_status,
    settings_from_environment,
)
from resampling_benchmark.service.engine import (
    TASK_THREADS,
    Engine,
    TaskResult,
)
from resampling_benchmark.service.groups import plan_groups
from tests.resampling_benchmark.conftest import TINY_ID, upload_directory
from tszbench.fs import StoreRoots
from tszbench.minmax import COLUMNS, column_minmax
from tszbench.rangeserver import free_port
from tszbench.recordings import RECORDINGS
from tszbench.schemas import relative_path

pytestmark = pytest.mark.network

RATE_HZ = 2000
HEADERS = (
    "X-Channels",
    "X-Rows",
    "X-Level",
    "X-Read-Seconds",
    "X-Reduce-Seconds",
    "X-S3-Requests",
    "X-S3-Bytes",
)


@pytest.fixture(scope="module")
def client(tiny_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(tiny_settings)) as test_client:
        yield test_client


def decode(response: httpx.Response) -> np.ndarray:
    assert response.status_code == 200, response.text
    return np.frombuffer(response.content, dtype=np.float32).reshape(
        int(response.headers["X-Channels"]), int(response.headers["X-Rows"]), 2
    )


def reference(
    tiny_dir: Path, start_us: int, end_us: int, channels: int = 4
) -> np.ndarray:
    """Reduce the raw samples of the local bundle channel by channel."""
    first = start_us * RATE_HZ // 1_000_000
    stop = end_us * RATE_HZ // 1_000_000
    with open_bundle(tiny_dir / TINY_ID / "tszarr.zarr") as bundle:
        rows = []
        for channel in list(bundle.continuous.values())[:channels]:
            values = np.asarray(channel.raw[first:stop], dtype=np.float32)
            rows.append(column_minmax(values, values))
    return np.stack(rows)


def test_health_lists_the_open_recordings(client: TestClient) -> None:
    body = client.get("/health").json()
    metadata = body.pop("metadata")
    assert body == {
        "recordings": [TINY_ID],
        "processes": 2,
        "task_threads": TASK_THREADS,
        "in_flight": 0,
    }
    assert metadata["processes"] == "2"
    assert metadata["task_threads"] == str(TASK_THREADS)
    assert {"date", "instance", "ami", "version.fastapi"} <= set(metadata)


def test_column_minmax_returns_a_float32_array_and_the_headers(
    client: TestClient,
) -> None:
    response = client.get(
        "/minmax",
        params={"recording": TINY_ID, "start_us": 0, "end_us": 10_000_000},
    )
    array = decode(response)
    assert array.shape == (4, COLUMNS, 2)
    assert array.dtype == np.float32
    for name in HEADERS:
        assert name in response.headers
    assert response.headers["X-Level"] == "0"
    assert int(response.headers["X-S3-Requests"]) > 0
    assert int(response.headers["X-S3-Bytes"]) > 0
    assert float(response.headers["X-Read-Seconds"]) > 0
    assert float(response.headers["X-Reduce-Seconds"]) > 0
    assert len(response.content) == array.nbytes


@pytest.mark.parametrize("start_us", [0, 7_000_000, 49_000_000])
def test_column_minmax_equals_the_reduction_of_the_raw_samples(
    client: TestClient, tiny_dir: Path, start_us: int
) -> None:
    end_us = start_us + 10_000_000
    served = decode(
        client.get(
            "/minmax",
            params={
                "recording": TINY_ID,
                "start_us": start_us,
                "end_us": end_us,
            },
        )
    )
    expected = reference(tiny_dir, start_us, end_us)
    np.testing.assert_array_equal(served, expected)


def test_column_minmax_is_nan_where_a_gap_lies(client: TestClient) -> None:
    served = decode(
        client.get(
            "/minmax",
            params={
                "recording": TINY_ID,
                "start_us": 45_000_000,
                "end_us": 55_000_000,
            },
        )
    )
    assert np.isnan(served[0]).any()
    assert not np.isnan(served[1]).any()


def test_groups_split_a_window_without_changing_the_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = {"recording": TINY_ID, "start_us": 0, "end_us": 10_000_000}
    whole = decode(client.get("/minmax", params=params))
    monkeypatch.setattr(
        engine_module,
        "plan_groups",
        functools.partial(plan_groups, group_samples=3_000),
    )
    split = decode(client.get("/minmax", params=params))
    np.testing.assert_array_equal(split, whole)
    plan = client.app.state.engine.plan(TINY_ID, 0, 10_000_000)  # type: ignore[attr-defined]
    assert plan.groups > 5


def test_a_window_below_two_samples_per_column_returns_every_sample(
    client: TestClient, tiny_dir: Path
) -> None:
    served = decode(
        client.get(
            "/minmax",
            params={"recording": TINY_ID, "start_us": 0, "end_us": 1_000_000},
        )
    )
    assert served.shape == (4, RATE_HZ, 2)
    np.testing.assert_array_equal(served, reference(tiny_dir, 0, 1_000_000))


def test_a_window_with_an_uneven_column_width_returns_every_column(
    client: TestClient,
) -> None:
    served = decode(
        client.get(
            "/minmax",
            params={"recording": TINY_ID, "start_us": 0, "end_us": 7_003_000},
        )
    )
    assert served.shape == (4, COLUMNS, 2)
    assert np.isfinite(served[1]).all()


def test_an_unknown_recording_is_404(client: TestClient) -> None:
    response = client.get(
        "/minmax", params={"recording": "R9", "start_us": 0, "end_us": 1}
    )
    assert response.status_code == 404


@pytest.mark.parametrize(
    ("start_us", "end_us"),
    [
        (5_000_000, 5_000_000),
        (10_000_000, 5_000_000),
        (115_000_000, 125_000_000),
    ],
)
def test_an_empty_or_outside_window_is_400(
    client: TestClient, start_us: int, end_us: int
) -> None:
    response = client.get(
        "/minmax",
        params={"recording": TINY_ID, "start_us": start_us, "end_us": end_us},
    )
    assert response.status_code == 400


def test_a_missing_parameter_is_422(client: TestClient) -> None:
    assert (
        client.get("/minmax", params={"recording": TINY_ID}).status_code == 422
    )


class SlowDownError(OSError):
    pass


@pytest.fixture
def thread_pool(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Run the tasks of the shared engine on threads, so tests can patch them."""
    pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(client.app.state.engine, "executor", pool)  # type: ignore[attr-defined]
    yield
    pool.shutdown(wait=True, cancel_futures=True)


@pytest.mark.usefixtures("thread_pool")
def test_an_error_while_reading_returns_503_for_slow_down(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(tasks: object) -> None:
        raise SlowDownError("SlowDown: reduce your request rate")

    monkeypatch.setattr(engine_module, "run_batch", fail)
    response = client.get(
        "/minmax",
        params={"recording": TINY_ID, "start_us": 0, "end_us": 10_000_000},
    )
    assert response.status_code == 503
    assert response.text == "SlowDownError"


@pytest.mark.usefixtures("thread_pool")
def test_another_error_while_reading_returns_500_with_the_class(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(tasks: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(engine_module, "run_batch", fail)
    response = client.get(
        "/minmax",
        params={"recording": TINY_ID, "start_us": 0, "end_us": 10_000_000},
    )
    assert response.status_code == 500
    assert response.text == "PermissionError"


def test_error_status_maps_only_slow_down_to_503() -> None:
    assert error_status(OSError("SlowDown")) == 503
    assert error_status(OSError("other")) == 500


def test_settings_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_S3_ROOT, "bucket/prefix")
    monkeypatch.setenv(ENV_S3_ENDPOINT_URL, "http://localhost:9000")
    monkeypatch.setenv(ENV_RECORDINGS, "R1, R3")
    monkeypatch.setenv(ENV_PROCESSES, "3")
    monkeypatch.setenv(ENV_TASK_THREADS, "8")
    settings = settings_from_environment()
    assert settings.roots == StoreRoots(
        local=Path(),
        http="",
        s3="bucket/prefix",
        s3_endpoint_url="http://localhost:9000",
    )
    assert settings.recordings == ("R1", "R3")
    assert settings.processes == 3
    assert settings.task_threads == 8


def test_settings_default_to_every_recording_and_the_default_pool_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_S3_ROOT, "bucket/prefix")
    monkeypatch.delenv(ENV_S3_ENDPOINT_URL, raising=False)
    monkeypatch.delenv(ENV_RECORDINGS, raising=False)
    monkeypatch.delenv(ENV_PROCESSES, raising=False)
    monkeypatch.delenv(ENV_TASK_THREADS, raising=False)
    settings = settings_from_environment()
    assert settings.recordings == ("R1", "R2", "R3", "R4")
    assert settings.processes == (os.cpu_count() or 1)
    assert settings.task_threads == TASK_THREADS
    assert settings.roots.s3_endpoint_url is None


def test_the_uvicorn_server_answers_like_the_application(
    server_url: str, client: TestClient
) -> None:
    params = {"recording": TINY_ID, "start_us": 3_000_000, "end_us": 13_000_000}
    remote = httpx.get(f"{server_url}/minmax", params=params, timeout=60)
    np.testing.assert_array_equal(
        decode(remote), decode(client.get("/minmax", params=params))
    )


@pytest.fixture(scope="module")
def one_thread_client(tiny_settings: Settings) -> Iterator[TestClient]:
    settings = replace(tiny_settings, task_threads=1)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_one_request_of_sixteen_tasks_runs_in_both_pool_processes(
    one_thread_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = engine_of(one_thread_client)
    monkeypatch.setattr(
        engine_module,
        "plan_groups",
        functools.partial(plan_groups, group_samples=5_000),
    )
    plan = engine.plan(TINY_ID, 0, 10_000_000)
    assert len(plan.tasks) == 16
    futures = engine.submit(plan.tasks)
    results: list[TaskResult] = [
        result for future in futures for result in future.result(timeout=60)
    ]
    assert len({result.pid for result in results}) == 2
    assert {result.pid for result in results} == set(engine.pids)


def test_every_pool_process_opens_each_bundle_once(client: TestClient) -> None:
    engine: Engine = client.app.state.engine  # type: ignore[attr-defined]
    assert len(engine.pids) == engine.processes == 2
    assert engine.bundles_opened == len(engine.recordings) == 1


def test_a_recording_that_does_not_open_raises_its_own_error(
    tiny_settings: Settings,
) -> None:
    with pytest.raises(Exception) as caught:  # noqa: PT011
        Engine(tiny_settings.roots, ("R9",), processes=2)
    assert type(caught.value).__name__ not in {
        "BrokenProcessPool",
        "BrokenBarrierError",
    }


def test_an_engine_needs_at_least_one_process(tiny_settings: Settings) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Engine(tiny_settings.roots, (TINY_ID,), processes=0)


GROUP_SAMPLES = 3_000
"""Group size that splits a 10 s window of the tiny recording into 7 groups."""

TASK_THREAD_COUNTS = (1, 4, 16)

TEN_SECONDS = {"recording": TINY_ID, "start_us": 0, "end_us": 10_000_000}


@pytest.fixture
def small_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        engine_module,
        "plan_groups",
        functools.partial(plan_groups, group_samples=GROUP_SAMPLES),
    )


@pytest.fixture(scope="module", params=TASK_THREAD_COUNTS)
def threaded_client(
    request: pytest.FixtureRequest, tiny_settings: Settings
) -> Iterator[TestClient]:
    settings = replace(tiny_settings, task_threads=request.param)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def engine_of(client: TestClient) -> Engine:
    engine: Engine = client.app.state.engine  # type: ignore[attr-defined]
    return engine


@pytest.fixture
def threaded_pool(
    threaded_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Run the batches of the engine on two threads, so tests can patch them."""
    pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(engine_of(threaded_client), "executor", pool)
    yield
    pool.shutdown(wait=True, cancel_futures=True)


def test_an_engine_has_the_task_threads_it_was_given(
    threaded_client: TestClient,
) -> None:
    body = threaded_client.get("/health").json()
    assert body["task_threads"] == engine_of(threaded_client).task_threads
    assert body["metadata"]["task_threads"] == str(body["task_threads"])
    assert body["in_flight"] == 0


def test_an_engine_needs_at_least_one_task_thread(
    tiny_settings: Settings,
) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Engine(tiny_settings.roots, (TINY_ID,), processes=1, task_threads=0)


def test_batches_hold_consecutive_tasks_of_at_most_the_batch_size() -> None:
    tasks = list(range(10))
    batches = engine_module.split_batches(tasks, 4)  # type: ignore[arg-type]
    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert engine_module.split_batches(tasks, 1) == [[i] for i in tasks]  # type: ignore[arg-type]
    assert engine_module.split_batches(tasks, 16) == [tasks]  # type: ignore[arg-type]
    assert engine_module.split_batches([], 4) == []


@pytest.mark.usefixtures("small_groups")
@pytest.mark.parametrize("window_us", [10_000_000, 60_000_000])
@pytest.mark.parametrize("start_us", [0, 7_000_000, 49_000_000])
def test_arrays_are_equal_at_every_task_thread_count(
    client: TestClient,
    threaded_client: TestClient,
    start_us: int,
    window_us: int,
) -> None:
    params = {
        "recording": TINY_ID,
        "start_us": start_us,
        "end_us": start_us + window_us,
    }
    serial = client.get("/minmax", params=params)
    threaded = threaded_client.get("/minmax", params=params)
    assert threaded.content == serial.content
    assert threaded.headers["X-Rows"] == serial.headers["X-Rows"]


@pytest.mark.usefixtures("small_groups")
def test_a_request_runs_as_batches_of_at_most_the_task_threads(
    threaded_client: TestClient,
) -> None:
    engine = engine_of(threaded_client)
    plan = engine.plan(TINY_ID, 0, 10_000_000)
    assert len(plan.tasks) == 4 * 7
    futures = engine.submit(plan.tasks)
    sizes = [len(future.result(timeout=60)) for future in futures]
    assert sum(sizes) == len(plan.tasks)
    assert max(sizes) == min(engine.task_threads, len(plan.tasks))
    assert len(sizes) == -(-len(plan.tasks) // engine.task_threads)


@pytest.mark.usefixtures("small_groups")
def test_assemble_gets_every_task_once_in_any_completion_order(
    threaded_client: TestClient,
) -> None:
    engine = engine_of(threaded_client)
    plan = engine.plan(TINY_ID, 0, 10_000_000)
    futures = engine.submit(plan.tasks)
    batches = [future.result(timeout=60) for future in futures]
    results = [result for batch in batches for result in batch]
    assert sorted((r.channel, r.first_column) for r in results) == sorted(
        (t.channel, t.group.first_column) for t in plan.tasks
    )
    in_order = engine_module.assemble(results, 4, plan.groups)
    generator = random.Random(len(batches))
    for _ in range(5):
        generator.shuffle(batches)
        shuffled = [result for batch in batches for result in batch]
        generator.shuffle(shuffled)
        np.testing.assert_array_equal(
            engine_module.assemble(shuffled, 4, plan.groups), in_order
        )


@pytest.mark.usefixtures("small_groups")
def test_each_task_counts_its_own_requests_and_bytes(
    threaded_client: TestClient,
) -> None:
    engine = engine_of(threaded_client)
    plan = engine.plan(TINY_ID, 0, 10_000_000)
    results = [
        result
        for future in engine.submit(plan.tasks)
        for result in future.result(timeout=60)
    ]
    assert all(result.requests > 0 and result.bytes > 0 for result in results)


@pytest.mark.usefixtures("threaded_pool", "small_groups")
def test_an_error_in_one_batch_returns_its_status_and_ends_the_others(
    threaded_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = engine_of(threaded_client)
    batches = len(
        engine_module.split_batches(
            engine.plan(TINY_ID, 0, 10_000_000).tasks, engine.task_threads
        )
    )
    started: list[int] = []
    lock = threading.Lock()

    def run(tasks: list[engine_module.Task]) -> list[TaskResult]:
        with lock:
            started.append(len(started))
            number = started[-1]
        if number == 0:
            raise SlowDownError("SlowDown: reduce your request rate")
        time.sleep(0.3)
        return []

    monkeypatch.setattr(engine_module, "run_batch", run)
    response = threaded_client.get("/minmax", params=TEN_SECONDS)
    assert response.status_code == 503
    assert response.text == "SlowDownError"
    if batches > 3:
        assert len(started) < batches
    deadline = time.monotonic() + 10
    while engine.in_flight:
        assert time.monotonic() < deadline, "in_flight did not reach 0"
        time.sleep(0.05)


@pytest.mark.usefixtures("small_groups")
def test_in_flight_counts_tasks_and_never_exceeds_those_submitted(
    threaded_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = engine_of(threaded_client)
    release = threading.Event()
    pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(engine, "executor", pool)
    monkeypatch.setattr(
        engine_module, "run_batch", lambda tasks: release.wait(30) and []
    )
    tasks = engine.plan(TINY_ID, 0, 10_000_000).tasks
    observed: list[int] = []
    try:
        futures = engine.submit(tasks)
        observed.append(engine.in_flight)
        assert observed[0] == len(tasks)
        release.set()
        for future in futures:
            future.result(timeout=30)
            observed.append(engine.in_flight)
        deadline = time.monotonic() + 10
        while engine.in_flight:
            assert time.monotonic() < deadline, "in_flight did not reach 0"
            time.sleep(0.01)
    finally:
        release.set()
        pool.shutdown(wait=True, cancel_futures=True)
    assert max(observed) <= len(tasks)
    assert observed == sorted(observed, reverse=True)


GRID_WINDOWS_S = (10, 60, 600, 3_600)
"""Windows of the benchmark that fit a recording of 1 h."""

GRID_RECORDINGS = ("R1", "R2")


@pytest.fixture(scope="module")
def grid_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("resampling_benchmark-grid")
    for name in GRID_RECORDINGS:
        spec = RECORDINGS[name].cut(3_600, "1h")
        path = root / relative_path(spec.id, "tszarr")
        path.parent.mkdir(parents=True)
        write_tszarr(spec, path)
    return root


@pytest.fixture(scope="module")
def grid_clients(
    grid_dir: Path, s3_endpoint: str
) -> Iterator[dict[int, TestClient]]:
    """One application per task thread count over R1-1h and R2-1h.

    Each pool has one process, so every task meets the same caches.
    """
    roots = upload_directory(grid_dir, s3_endpoint)
    names = tuple(
        RECORDINGS[name].cut(3_600, "1h").id for name in GRID_RECORDINGS
    )
    with ExitStack() as stack:
        yield {
            threads: stack.enter_context(
                TestClient(
                    create_app(
                        Settings(
                            roots=roots,
                            recordings=names,
                            processes=1,
                            task_threads=threads,
                        )
                    )
                )
            )
            for threads in TASK_THREAD_COUNTS
        }


@pytest.mark.parametrize("window_s", GRID_WINDOWS_S)
@pytest.mark.parametrize("name", GRID_RECORDINGS)
def test_a_grid_window_answers_alike_at_every_task_thread_count(
    grid_clients: dict[int, TestClient], name: str, window_s: int
) -> None:
    params = {
        "recording": f"{name}-1h",
        "start_us": 0,
        "end_us": window_s * 1_000_000,
    }
    answers = {}
    for threads, grid_client in grid_clients.items():
        grid_client.get("/minmax", params=params)
        answers[threads] = grid_client.get("/minmax", params=params)
    serial = answers[1]
    assert serial.status_code == 200
    for response in answers.values():
        assert response.content == serial.content
        for header in ("X-Channels", "X-Rows", "X-S3-Requests", "X-S3-Bytes"):
            assert response.headers[header] == serial.headers[header]


@pytest.fixture(params=(1, 4), ids=("one_thread", "four_threads"))
def live_app(
    request: pytest.FixtureRequest, tiny_settings: Settings
) -> Iterator[tuple[str, FastAPI]]:
    """Run the application on a real socket in a thread and yield its URL."""
    app = create_app(
        replace(tiny_settings, processes=1, task_threads=request.param)
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=free_port(), log_level="warning"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 120
    while not server.started:
        assert thread.is_alive(), "server exited during startup"
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.1)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", app
    server.should_exit = True
    thread.join(timeout=60)


def wait_for_in_flight(url: str, expected: int) -> None:
    deadline = time.monotonic() + 30
    while httpx.get(f"{url}/health").json()["in_flight"] != expected:
        assert time.monotonic() < deadline, (
            f"in_flight did not reach {expected}"
        )
        time.sleep(0.05)


@pytest.mark.usefixtures("small_groups")
def test_batches_of_a_request_whose_client_gave_up_are_canceled(
    live_app: tuple[str, FastAPI], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, app = live_app
    threads = app.state.engine.task_threads
    release = threading.Event()
    started: list[int] = []

    def slow(tasks: list[engine_module.Task]) -> list[TaskResult]:
        started.append(len(tasks))
        release.wait(30)
        return []

    pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(app.state.engine, "executor", pool)
    monkeypatch.setattr(engine_module, "run_batch", slow)
    try:
        with pytest.raises(httpx.ReadTimeout):
            httpx.get(f"{url}/minmax", params=TEN_SECONDS, timeout=1.0)
        wait_for_in_flight(url, 2 * threads)
        release.set()
        wait_for_in_flight(url, 0)
        assert started == [threads, threads]
    finally:
        release.set()
        pool.shutdown(wait=True, cancel_futures=True)


@pytest.mark.usefixtures("small_groups")
def test_in_flight_is_zero_after_a_burst_of_fifty_requests(
    live_app: tuple[str, FastAPI],
) -> None:
    url, app = live_app
    tasks = len(app.state.engine.plan(TINY_ID, 0, 10_000_000).tasks)
    peak = 0
    done = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not done.is_set():
            peak = max(peak, httpx.get(f"{url}/health").json()["in_flight"])

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        with ThreadPoolExecutor(max_workers=50) as pool:
            responses = list(
                pool.map(
                    lambda _: httpx.get(
                        f"{url}/minmax", params=TEN_SECONDS, timeout=120
                    ),
                    range(50),
                )
            )
    finally:
        done.set()
        sampler.join(timeout=30)
    assert {response.status_code for response in responses} == {200}
    assert len({response.content for response in responses}) == 1
    assert 0 < peak <= 50 * tasks
    wait_for_in_flight(url, 0)
