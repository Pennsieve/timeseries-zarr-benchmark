"""FastAPI application that answers minmax requests from raw samples."""

import asyncio
import os
from collections.abc import AsyncIterator
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Query, Request, Response

from resampling_benchmark.service.engine import (
    DEFAULT_PROCESSES,
    Engine,
    TaskResult,
    assemble,
)
from tszbench.csvmeta import run_metadata
from tszbench.fs import StoreRoots
from tszbench.recordings import RECORDINGS

ENV_S3_ROOT = "RESAMPLING_S3_ROOT"
ENV_S3_ENDPOINT_URL = "RESAMPLING_S3_ENDPOINT_URL"
ENV_RECORDINGS = "RESAMPLING_RECORDINGS"
ENV_PROCESSES = "RESAMPLING_PROCESSES"

SLOW_DOWN = "SlowDown"
"""Name of the S3 error code that maps to HTTP 503."""

DISCONNECT_POLL_S = 0.25
"""Seconds between checks for a client that closed its connection."""


@dataclass(frozen=True)
class Settings:
    """Service configuration.

    Attributes:
        roots: S3 bucket, prefix, and endpoint.
        recordings: Ids of the recordings to open at startup.
        processes: Processes in the pool that reads and reduces the tasks.
    """

    roots: StoreRoots
    recordings: tuple[str, ...]
    processes: int


def settings_from_environment() -> Settings:
    """Read the service configuration from the `RESAMPLING_*` variables.

    `RESAMPLING_S3_ROOT` is required. `RESAMPLING_S3_ENDPOINT_URL` is optional.
    `RESAMPLING_RECORDINGS` is a comma-separated list and defaults to every
    recording of the benchmark. `RESAMPLING_PROCESSES` defaults to
    `DEFAULT_PROCESSES`.

    Raises:
        KeyError: `RESAMPLING_S3_ROOT` is not set.
    """
    recordings = os.environ.get(ENV_RECORDINGS) or ",".join(RECORDINGS)
    return Settings(
        roots=StoreRoots(
            local=Path(),
            http="",
            s3=os.environ[ENV_S3_ROOT],
            s3_endpoint_url=os.environ.get(ENV_S3_ENDPOINT_URL) or None,
        ),
        recordings=tuple(name.strip() for name in recordings.split(",")),
        processes=int(os.environ.get(ENV_PROCESSES) or DEFAULT_PROCESSES),
    )


def error_status(error: BaseException) -> int:
    """Return 503 for an S3 SlowDown error and 500 for any other error."""
    text = f"{type(error).__name__} {error}"
    return 503 if SLOW_DOWN in text else 500


async def cancel_on_disconnect(
    request: Request, futures: list["Future[TaskResult]"]
) -> None:
    """Cancel the tasks that have not started once the client disconnects.

    Runs until its task is canceled. A task that already runs finishes.
    """
    while not await request.is_disconnected():
        await asyncio.sleep(DISCONNECT_POLL_S)
    for future in futures:
        future.cancel()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Configuration, or None to read the environment. The pool
            starts and opens every recording when the application starts.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        chosen = settings or settings_from_environment()
        engine = await asyncio.to_thread(
            Engine, chosen.roots, chosen.recordings, chosen.processes
        )
        app.state.engine = engine
        app.state.metadata = await asyncio.to_thread(
            run_metadata,
            s3_root=chosen.roots.s3,
            s3_endpoint_url=chosen.roots.s3_endpoint_url or "none",
            recordings=",".join(chosen.recordings),
            processes=chosen.processes,
        )
        yield
        await asyncio.to_thread(engine.close)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, object]:
        engine: Engine = app.state.engine
        return {
            "recordings": list(engine.recordings),
            "processes": engine.processes,
            "in_flight": engine.in_flight,
            "metadata": app.state.metadata,
        }

    @app.get("/minmax")
    async def minmax_endpoint(
        request: Request,
        recording: str,
        start_us: int = Query(),
        end_us: int = Query(),
    ) -> Response:
        engine: Engine = app.state.engine
        if recording not in engine.recordings:
            return Response(f"unknown recording {recording}", status_code=404)
        try:
            plan = engine.plan(recording, start_us, end_us)
        except ValueError as error:
            return Response(str(error), status_code=400)
        futures = [engine.submit(task) for task in plan.tasks]
        watcher = asyncio.create_task(cancel_on_disconnect(request, futures))
        try:
            outcomes = await asyncio.gather(
                *(asyncio.wrap_future(future) for future in futures),
                return_exceptions=True,
            )
        finally:
            watcher.cancel()
        results: list[TaskResult] = []
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                return Response(
                    type(outcome).__name__, status_code=error_status(outcome)
                )
            results.append(outcome)
        array = assemble(results, plan.info.channels, plan.groups)
        return Response(
            array.tobytes(),
            media_type="application/octet-stream",
            headers={
                "X-Channels": str(array.shape[0]),
                "X-Rows": str(array.shape[1]),
                "X-Level": "0",
                "X-Read-Seconds": repr(sum(r.read_seconds for r in results)),
                "X-Reduce-Seconds": repr(
                    sum(r.reduce_seconds for r in results)
                ),
                "X-S3-Requests": str(sum(r.requests for r in results)),
                "X-S3-Bytes": str(sum(r.bytes for r in results)),
            },
        )

    return app
