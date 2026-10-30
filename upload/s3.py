"""Upload the files of a recording with `aws s3 cp` and sync them back."""

import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from tszbench import schemas
from tszbench.csvmeta import CsvWriter, run_metadata
from upload.storage import measure, present_formats

Runner = Callable[[Sequence[str]], str]
"""Runs a command and returns its standard output. Raises on a nonzero exit."""

Clock = Callable[[], float]
"""Returns a time in seconds from any fixed origin."""


def run_command(command: Sequence[str]) -> str:
    """Run command and return its standard output.

    Raises:
        RuntimeError: The command exits with a nonzero status. The message holds
            the command and its standard error.
    """
    result = subprocess.run(
        list(command), capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)}: exit status {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def s3_uri(bucket: str, prefix: str, path: str) -> str:
    """Return the URI of path below prefix in bucket. An empty prefix adds nothing."""
    parts = [part for part in (prefix.strip("/"), path.strip("/")) if part]
    return f"s3://{bucket}/{'/'.join(parts)}"


def _with_options(
    command: list[str], region: str | None, endpoint_url: str | None
) -> list[str]:
    """Return command followed by the region and endpoint options that are set."""
    if region:
        command = [*command, "--region", region]
    if endpoint_url:
        command = [*command, "--endpoint-url", endpoint_url]
    return command


def copy_command(
    local: Path,
    uri: str,
    *,
    recursive: bool,
    region: str | None = None,
    endpoint_url: str | None = None,
) -> list[str]:
    """Return the `aws s3 cp` command that uploads local to uri.

    The command sets no option that changes the transfer, so the AWS CLI uses
    its default concurrency and part size.
    """
    command = ["aws", "s3", "cp"]
    if recursive:
        command.append("--recursive")
    command += [str(local), uri]
    return _with_options(command, region, endpoint_url)


def sync_command(
    uri: str,
    local: Path,
    *,
    region: str | None = None,
    endpoint_url: str | None = None,
) -> list[str]:
    """Return the `aws s3 sync` command that downloads uri to local."""
    return _with_options(
        ["aws", "s3", "sync", uri, str(local)], region, endpoint_url
    )


def upload_recording(
    root: Path,
    recording_id: str,
    bucket: str,
    prefix: str,
    output: Path,
    *,
    run: Runner = run_command,
    clock: Clock = time.perf_counter,
    region: str | None = None,
    endpoint_url: str | None = None,
) -> list[dict[str, object]]:
    """Upload every format of a recording that exists under root, and time each.

    A directory format goes up with `aws s3 cp --recursive` and a single-file
    format with plain `aws s3 cp`. The destination is
    `s3://<bucket>/<prefix>/<recording id>/<file name>`. An endpoint_url
    sends every command to an S3-compatible server. The 1-byte object of
    `schemas.ONE_BYTE_OBJECT` is uploaded first and is not timed. The rows go to
    output as `schemas.UPLOAD`, one flushed row per file.

    Returns:
        The rows written.

    Raises:
        FileNotFoundError: No format of the recording exists under root.
    """
    formats = present_formats(root, recording_id)
    header = run_metadata(
        bucket=bucket,
        prefix=prefix,
        aws_cli=run(["aws", "--version"]).strip(),
        s3_endpoint_url=endpoint_url or "none",
    )
    upload_one_byte(
        bucket, prefix, run=run, region=region, endpoint_url=endpoint_url
    )
    rows: list[dict[str, object]] = []
    with CsvWriter(output, schemas.UPLOAD, header) as writer:
        for file_format in formats:
            local = root / schemas.relative_path(recording_id, file_format)
            size, objects = measure(local)
            uri = s3_uri(
                bucket,
                prefix,
                schemas.relative_path(recording_id, file_format),
            )
            command = copy_command(
                local,
                uri,
                recursive=local.is_dir(),
                region=region,
                endpoint_url=endpoint_url,
            )
            started = clock()
            run(command)
            seconds = clock() - started
            row: dict[str, object] = {
                "recording": recording_id,
                "format": file_format,
                "seconds": seconds,
                "bytes": size,
                "objects": objects,
                "bytes_per_second": size / seconds if seconds > 0 else 0.0,
            }
            writer.write(row)
            rows.append(row)
    return rows


def upload_one_byte(
    bucket: str,
    prefix: str,
    *,
    run: Runner = run_command,
    region: str | None = None,
    endpoint_url: str | None = None,
) -> None:
    """Upload a 1-byte object to `schemas.ONE_BYTE_OBJECT` below prefix."""
    with tempfile.TemporaryDirectory() as directory:
        local = Path(directory) / "one-byte"
        local.write_bytes(b"\0")
        uri = s3_uri(bucket, prefix, schemas.ONE_BYTE_OBJECT)
        run(
            copy_command(
                local,
                uri,
                recursive=False,
                region=region,
                endpoint_url=endpoint_url,
            )
        )


def pull(
    root: Path,
    bucket: str,
    prefix: str,
    *,
    recordings: Sequence[str] | None = None,
    run: Runner = run_command,
    region: str | None = None,
    endpoint_url: str | None = None,
) -> None:
    """Sync objects below prefix in bucket to root. The sync is not timed.

    Without recordings, one sync copies everything below prefix. With
    recordings, one sync copies each `<recording>/` prefix, and one more copies
    the directory of `schemas.ONE_BYTE_OBJECT`.
    """
    root.mkdir(parents=True, exist_ok=True)
    if recordings is None:
        paths = [""]
    else:
        paths = [
            *(f"{name}/" for name in recordings),
            schemas.ONE_BYTE_OBJECT.rpartition("/")[0] + "/",
        ]
    for path in paths:
        source = s3_uri(bucket, prefix, path).rstrip("/") + "/"
        run(
            sync_command(
                source,
                root / path,
                region=region,
                endpoint_url=endpoint_url,
            )
        )
