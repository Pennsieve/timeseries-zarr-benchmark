import functools
import os
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from timeseries_zarr import open_bundle

from resampling_benchmark.service import engine as engine_module
from resampling_benchmark.service.app import (
    ENV_PROCESSES,
    ENV_RECORDINGS,
    ENV_S3_ENDPOINT_URL,
    ENV_S3_ROOT,
    Settings,
    create_app,
    error_status,
    settings_from_environment,
)
from resampling_benchmark.service.engine import Engine, TaskResult
from resampling_benchmark.service.groups import plan_groups
from tests.resampling_benchmark.conftest import TINY_ID
from tszbench.fs import StoreRoots
from tszbench.minmax import COLUMNS, column_minmax
from tszbench.rangeserver import free_port

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
    assert body == {"recordings": [TINY_ID], "processes": 2, "in_flight": 0}
    assert metadata["processes"] == "2"
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
    def fail(task: object) -> None:
        raise SlowDownError("SlowDown: reduce your request rate")

    monkeypatch.setattr(engine_module, "run_task", fail)
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
    def fail(task: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(engine_module, "run_task", fail)
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
    settings = settings_from_environment()
    assert settings.roots == StoreRoots(
        local=Path(),
        http="",
        s3="bucket/prefix",
        s3_endpoint_url="http://localhost:9000",
    )
    assert settings.recordings == ("R1", "R3")
    assert settings.processes == 3


def test_settings_default_to_every_recording_and_one_process_per_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_S3_ROOT, "bucket/prefix")
    monkeypatch.delenv(ENV_S3_ENDPOINT_URL, raising=False)
    monkeypatch.delenv(ENV_RECORDINGS, raising=False)
    monkeypatch.delenv(ENV_PROCESSES, raising=False)
    settings = settings_from_environment()
    assert settings.recordings == ("R1", "R2", "R3", "R4")
    assert settings.processes == (os.cpu_count() or 1)
    assert settings.roots.s3_endpoint_url is None


def test_the_uvicorn_server_answers_like_the_application(
    server_url: str, client: TestClient
) -> None:
    params = {"recording": TINY_ID, "start_us": 3_000_000, "end_us": 13_000_000}
    remote = httpx.get(f"{server_url}/minmax", params=params, timeout=60)
    np.testing.assert_array_equal(
        decode(remote), decode(client.get("/minmax", params=params))
    )


def test_one_request_of_sixteen_tasks_runs_in_both_pool_processes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine: Engine = client.app.state.engine  # type: ignore[attr-defined]
    monkeypatch.setattr(
        engine_module,
        "plan_groups",
        functools.partial(plan_groups, group_samples=5_000),
    )
    plan = engine.plan(TINY_ID, 0, 10_000_000)
    assert len(plan.tasks) == 16
    futures = [engine.submit(task) for task in plan.tasks]
    results: list[TaskResult] = [
        future.result(timeout=60) for future in futures
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


@pytest.fixture
def live_app(tiny_settings: Settings) -> Iterator[tuple[str, FastAPI]]:
    """Run the application on a real socket in a thread and yield its URL."""
    app = create_app(replace(tiny_settings, processes=1))
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


def test_tasks_of_a_request_whose_client_gave_up_are_canceled(
    live_app: tuple[str, FastAPI], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, app = live_app
    release = threading.Event()
    started: list[object] = []

    def slow(task: object) -> None:
        started.append(task)
        release.wait(30)

    pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(app.state.engine, "executor", pool)
    monkeypatch.setattr(engine_module, "run_task", slow)
    params = {"recording": TINY_ID, "start_us": 0, "end_us": 10_000_000}
    try:
        with pytest.raises(httpx.ReadTimeout):
            httpx.get(f"{url}/minmax", params=params, timeout=1.0)
        wait_for_in_flight(url, 2)
        release.set()
        wait_for_in_flight(url, 0)
        assert len(started) == 2
    finally:
        release.set()
        pool.shutdown(wait=True, cancel_futures=True)
