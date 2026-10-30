"""A tiny recording in a local directory and in moto, and a running server."""

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import boto3
import httpx
import pytest

from generate.tszarr import write_tszarr
from resampling_benchmark.service.app import Settings
from tests.conftest import REGION, S3_PREFIX
from tszbench.fs import StoreRoots
from tszbench.rangeserver import free_port
from tszbench.recordings import RECORDINGS, Gap, RecordingSpec
from tszbench.schemas import relative_path

TINY_ID = "T1"
TINY_SPEC = replace(
    RECORDINGS["R3"],
    id=TINY_ID,
    channels=4,
    rate_hz=2000,
    duration_s=120,
    levels=3,
    gaps=(
        Gap(channel=0, start_s=20.0, duration_s=0.5),
        Gap(channel=0, start_s=50.0, duration_s=2.0),
        Gap(channel=2, start_s=80.0, duration_s=0.25),
    ),
)
"""Four channels at 2000 Hz for 2 min, with three short gaps."""

SERVER_START_TIMEOUT_S = 120.0


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory that holds `T1/tszarr.zarr`."""
    root = tmp_path_factory.mktemp("resampling_benchmark-tiny")
    path = root / relative_path(TINY_ID, "tszarr")
    path.parent.mkdir(parents=True)
    write_tszarr(TINY_SPEC, path)
    return root


def upload_directory(directory: Path, s3_endpoint: str) -> StoreRoots:
    """Upload a directory to a new moto bucket and return its store roots."""
    bucket = f"resampling_benchmark-{time.time_ns()}"
    client = boto3.client("s3", endpoint_url=s3_endpoint, region_name=REGION)
    client.create_bucket(Bucket=bucket)
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            key = f"{S3_PREFIX}/{path.relative_to(directory).as_posix()}"
            client.put_object(Bucket=bucket, Key=key, Body=path.read_bytes())
    return StoreRoots(
        local=directory,
        http="",
        s3=f"{bucket}/{S3_PREFIX}",
        s3_endpoint_url=s3_endpoint,
    )


@pytest.fixture(scope="module")
def tiny_roots(tiny_dir: Path, s3_endpoint: str) -> StoreRoots:
    """Store roots with the tiny bundle uploaded to a new moto bucket."""
    return upload_directory(tiny_dir, s3_endpoint)


@pytest.fixture(scope="module")
def tiny_settings(tiny_roots: StoreRoots) -> Settings:
    return Settings(roots=tiny_roots, recordings=(TINY_ID,), processes=2)


@pytest.fixture(scope="module")
def tiny_spec() -> RecordingSpec:
    return TINY_SPEC


@pytest.fixture(scope="module")
def server_url(tiny_roots: StoreRoots) -> Iterator[str]:
    """Start `python -m resampling_benchmark.service` over the tiny bundle."""
    port = free_port()
    environment = dict(os.environ)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "resampling_benchmark.service",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "1",
            "--processes",
            "2",
            "--s3-root",
            tiny_roots.s3,
            "--s3-endpoint-url",
            str(tiny_roots.s3_endpoint_url),
            "--recordings",
            TINY_ID,
        ],
        env=environment,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + SERVER_START_TIMEOUT_S
    try:
        while True:
            if process.poll() is not None:
                raise RuntimeError("server exited during startup")
            try:
                if httpx.get(f"{url}/health", timeout=2.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("server did not start")
            time.sleep(0.5)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
