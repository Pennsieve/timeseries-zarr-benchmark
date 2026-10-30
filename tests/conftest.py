"""Fixtures that serve a directory over local HTTP and local S3."""

import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import boto3
import pytest
from moto.server import ThreadedMotoServer

from tszbench.fs import StoreRoots
from tszbench.rangeserver import (
    RunningServer,
    free_port,
    serve_directory,
)

REGION = "us-east-1"
S3_PREFIX = "data"


@pytest.fixture
def serve_http() -> Iterator[Callable[[Path], str]]:
    """Return a function that serves a directory and returns its base URL."""
    servers: list[RunningServer] = []

    def serve(directory: Path) -> str:
        running = serve_directory(directory)
        servers.append(running)
        return running.url

    yield serve
    for running in servers:
        running.stop()


@pytest.fixture(scope="session")
def s3_endpoint() -> Iterator[str]:
    """Start a local moto S3 server and return its endpoint URL."""
    port = free_port()
    server = ThreadedMotoServer(
        ip_address="127.0.0.1", port=port, verbose=False
    )
    server.start()
    environment = {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": REGION,
        "AWS_REGION": REGION,
        "AWS_EC2_METADATA_DISABLED": "true",
    }
    with pytest.MonkeyPatch.context() as patch:
        for key, value in environment.items():
            patch.setenv(key, value)
        patch.delenv("AWS_PROFILE", raising=False)
        yield f"http://127.0.0.1:{port}"
    server.stop()


@pytest.fixture
def upload_s3(s3_endpoint: str) -> Callable[[Path], str]:
    """Return a function that uploads a directory to a new bucket."""

    def upload(directory: Path) -> str:
        bucket = f"tszbench-{uuid.uuid4().hex[:16]}"
        client = boto3.client(
            "s3", endpoint_url=s3_endpoint, region_name=REGION
        )
        client.create_bucket(Bucket=bucket)
        root = Path(directory)
        for path in sorted(root.rglob("*")):
            if path.is_file():
                key = f"{S3_PREFIX}/{path.relative_to(root).as_posix()}"
                client.put_object(
                    Bucket=bucket, Key=key, Body=path.read_bytes()
                )
        return f"{bucket}/{S3_PREFIX}"

    return upload


@pytest.fixture
def store_roots(
    serve_http: Callable[[Path], str],
    upload_s3: Callable[[Path], str],
    s3_endpoint: str,
) -> Callable[[Path], StoreRoots]:
    """Return a function that serves a directory over HTTP and S3."""

    def roots(directory: Path) -> StoreRoots:
        return StoreRoots(
            local=Path(directory),
            http=serve_http(directory),
            s3=upload_s3(directory),
            s3_endpoint_url=s3_endpoint,
        )

    return roots
