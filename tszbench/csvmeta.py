"""CSV files that open with the run metadata as `# key: value` lines."""

import csv
import os
import platform
import subprocess
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from types import TracebackType
from typing import IO, Self

LIBRARIES = (
    "numpy",
    "scipy",
    "zarr",
    "numcodecs",
    "timeseries-zarr",
    "tsdownsample",
    "pynwb",
    "hdmf",
    "hdmf-zarr",
    "h5py",
    "hdf5plugin",
    "fsspec",
    "s3fs",
    "aiohttp",
    "httpx",
    "fastapi",
    "uvicorn",
    "pandas",
    "matplotlib",
)
"""Distributions whose version every CSV header records."""

_IMDS = "http://169.254.169.254/latest"


def _instance_metadata(item: str) -> str | None:
    """Return an EC2 instance metadata item, or None off EC2."""
    try:
        token_request = urllib.request.Request(
            f"{_IMDS}/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urllib.request.urlopen(token_request, timeout=0.5) as response:
            token = response.read().decode()
        request = urllib.request.Request(
            f"{_IMDS}/meta-data/{item}",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(request, timeout=0.5) as response:
            return str(response.read().decode())
    except OSError:
        return None


def _commit() -> str:
    """Return the benchmark repository commit, from BENCH_COMMIT or git."""
    if commit := os.environ.get("BENCH_COMMIT"):
        return commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _version(distribution: str) -> str:
    """Return the installed version of distribution, or `absent`."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "absent"


def run_metadata(**extra: object) -> dict[str, str]:
    """Return the metadata every CSV header records, followed by extra.

    The keys are `date` (UTC, ISO 8601), `commit`, `region`, `instance`,
    `ami`, `python`, one `version.<distribution>` per entry of `LIBRARIES`,
    then the keys of extra in order. Off EC2, `region` comes from AWS_REGION,
    `instance` is `local`, and `ami` is `none`.
    """
    region = _instance_metadata("placement/region") or os.environ.get(
        "AWS_REGION", "none"
    )
    fields = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _commit(),
        "region": region,
        "instance": _instance_metadata("instance-type") or "local",
        "ami": _instance_metadata("ami-id") or "none",
        "python": platform.python_version(),
    }
    fields |= {f"version.{name}": _version(name) for name in LIBRARIES}
    fields |= {key: str(value) for key, value in extra.items()}
    return fields


class CsvWriter:
    """Write a CSV with a metadata header, one flushed row at a time.

    Use as a context manager. The file opens with one `# key: value` line per
    metadata item, then the column names.
    """

    def __init__(
        self, path: Path, columns: Sequence[str], header: Mapping[str, str]
    ) -> None:
        """Create path and write the metadata lines and the column names."""
        for key, value in header.items():
            if "\n" in key or "\n" in value or ":" in key:
                raise ValueError(
                    f"metadata key {key!r}: must hold no newline or colon, and "
                    "its value no newline"
                )
        self.columns = tuple(columns)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = path.open("w", newline="")
        for key, value in header.items():
            self._file.write(f"# {key}: {value}\n")
        self._writer = csv.DictWriter(self._file, fieldnames=self.columns)
        self._writer.writeheader()
        self._file.flush()

    def write(self, row: Mapping[str, object]) -> None:
        """Write one row. Every column must be present and nothing else."""
        if set(row) != set(self.columns):
            missing = sorted(set(self.columns) - set(row))
            extra = sorted(set(row) - set(self.columns))
            raise ValueError(f"row: missing {missing}, unexpected {extra}")
        self._writer.writerow(row)
        self._file.flush()

    def write_all(self, rows: Iterable[Mapping[str, object]]) -> None:
        """Write every row of rows."""
        for row in rows:
            self.write(row)

    def close(self) -> None:
        """Close the file."""
        self._file.close()

    def __enter__(self) -> Self:
        """Return the writer."""
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the file."""
        self.close()


def read_header(path: Path) -> dict[str, str]:
    """Return the metadata lines at the top of a CSV written by CsvWriter."""
    header: dict[str, str] = {}
    with path.open() as file:
        for line in file:
            if not line.startswith("# "):
                break
            key, _, value = line[2:].rstrip("\n").partition(": ")
            header[key] = value
    return header


def read_rows(path: Path) -> list[dict[str, str]]:
    """Return the rows of a CSV written by CsvWriter, metadata lines left out."""
    with path.open(newline="") as file:
        lines = (line for line in file if not line.startswith("# "))
        return list(csv.DictReader(lines))
