"""Launch the resampling server: `python -m resampling_benchmark.service`."""

import argparse
import os
import resource
from collections.abc import Sequence

import uvicorn

from resampling_benchmark.service.app import (
    ENV_PROCESSES,
    ENV_RECORDINGS,
    ENV_S3_ENDPOINT_URL,
    ENV_S3_ROOT,
)
from resampling_benchmark.service.engine import DEFAULT_PROCESSES
from tszbench.recordings import RECORDINGS

DEFAULT_PORT = 8000

OPEN_FILES_MINIMUM = 16384
"""Open files the service process may hold. Each connected user holds one."""


def raise_open_file_limit(minimum: int = OPEN_FILES_MINIMUM) -> int:
    """Raise the soft open file limit to `minimum`, or to the hard limit if lower.

    The pool processes inherit the new limit. A default soft limit of 1024 makes
    the server close the connections of the users beyond it.

    Returns:
        The soft limit after the call.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = minimum if hard == resource.RLIM_INFINITY else min(minimum, hard)
    if soft == resource.RLIM_INFINITY or soft >= wanted:
        return soft
    resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
    return wanted


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line. Each option falls back to its variable."""
    parser = argparse.ArgumentParser(
        prog="python -m resampling_benchmark.service",
        description="Serve minmax arrays computed from raw samples in S3.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="uvicorn worker processes, each with its own pool",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=int(os.environ.get(ENV_PROCESSES) or DEFAULT_PROCESSES),
        help="pool processes that read and reduce, one per vCPU by default",
    )
    parser.add_argument(
        "--s3-root",
        default=os.environ.get(ENV_S3_ROOT),
        help="bucket and prefix as bucket/prefix",
    )
    parser.add_argument(
        "--s3-endpoint-url", default=os.environ.get(ENV_S3_ENDPOINT_URL)
    )
    parser.add_argument(
        "--recordings",
        default=os.environ.get(ENV_RECORDINGS) or ",".join(RECORDINGS),
        help="comma-separated recording ids to open at startup",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Export the configuration to the environment and start uvicorn."""
    arguments = parse_arguments(argv)
    if not arguments.s3_root:
        raise SystemExit(f"--s3-root or {ENV_S3_ROOT} is required")
    os.environ[ENV_S3_ROOT] = arguments.s3_root
    if arguments.s3_endpoint_url:
        os.environ[ENV_S3_ENDPOINT_URL] = arguments.s3_endpoint_url
    os.environ[ENV_RECORDINGS] = arguments.recordings
    os.environ[ENV_PROCESSES] = str(arguments.processes)
    raise_open_file_limit()
    uvicorn.run(
        "resampling_benchmark.service.app:create_app",
        factory=True,
        host=arguments.host,
        port=arguments.port,
        workers=arguments.workers,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
