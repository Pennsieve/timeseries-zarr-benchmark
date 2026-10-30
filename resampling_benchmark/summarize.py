"""Merge the CSVs of the resampling benchmark runs: `python -m resampling_benchmark.summarize`.

Reads the request CSVs of every run directory and the monitor CSVs next to
them. Writes `tszbench.schemas.RESAMPLING_REQUESTS_FILE` and
`tszbench.schemas.RESAMPLING_RUNS_FILE` to the results directory.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from resampling_benchmark.load.generator import REQUEST_TIMEOUT_S, STATUS_OK
from resampling_benchmark.load.orchestrate import LINK_BYTES_PER_S
from tszbench.csvmeta import CsvWriter, read_header, run_metadata
from tszbench.recordings import RecordingSpec, resolve
from tszbench.schemas import (
    GENERATOR_MONITOR_PATTERN,
    LINK_LIMITED_FILE,
    REQUESTS_PATTERN,
    RESAMPLING_RAW_DIRECTORY,
    RESAMPLING_REQUESTS,
    RESAMPLING_REQUESTS_FILE,
    RESAMPLING_RUNS,
    RESAMPLING_RUNS_FILE,
    SERVER_METADATA_FILE,
    SERVER_MONITOR_FILE,
)

BYTES_PER_SAMPLE = 4
LOW_PERCENTILE = 5
HIGH_PERCENTILE = 95
TRUE = "true"
FAILED_REQUEST_SECONDS_KEY = "failed_request_seconds"


@dataclass(frozen=True)
class RunFiles:
    """CSV files of one run.

    Attributes:
        run: Run id, the name of the directory.
        requests: Request CSVs, one per generator.
        server_monitors: Monitor CSVs of the server.
        generator_monitors: Monitor CSVs of the load generators.
        server_metadata: JSON file with the run metadata of the server, or
            None when the run has none.
        link_limited: True when the run directory holds a
            `tszbench.schemas.LINK_LIMITED_FILE`.
    """

    run: str
    requests: tuple[Path, ...]
    server_monitors: tuple[Path, ...]
    generator_monitors: tuple[Path, ...]
    server_metadata: Path | None = None
    link_limited: bool = False


def find_runs(root: Path) -> list[RunFiles]:
    """Return the runs under root that have at least one request CSV."""
    runs = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        requests = tuple(sorted(directory.glob(REQUESTS_PATTERN)))
        if not requests:
            continue
        runs.append(
            RunFiles(
                run=directory.name,
                requests=requests,
                server_monitors=tuple(
                    sorted(directory.glob(SERVER_MONITOR_FILE))
                ),
                generator_monitors=tuple(
                    sorted(directory.glob(GENERATOR_MONITOR_PATTERN))
                ),
                server_metadata=(
                    directory / SERVER_METADATA_FILE
                    if (directory / SERVER_METADATA_FILE).is_file()
                    else None
                ),
                link_limited=(directory / LINK_LIMITED_FILE).exists(),
            )
        )
    return runs


def read_requests(path: Path) -> Any:
    """Read a request CSV as text, with an empty field kept as ''."""
    return pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)


def _numbers(frame: Any, column: str) -> Any:
    return pd.to_numeric(frame[column], errors="coerce")


def charged_seconds(measured: Any) -> Any:
    """Return the seconds each measured request counts for in the statistics.

    An `ok` request counts its own seconds. Any other request counts the
    larger of its own seconds and `REQUEST_TIMEOUT_S`, and a missing seconds
    value counts as `REQUEST_TIMEOUT_S`.

    Args:
        measured: Measured request rows with numeric `seconds` and a `status`.
    """
    own = measured["seconds"].fillna(0.0)
    failed = np.maximum(own, REQUEST_TIMEOUT_S)
    return own.where(measured["status"] == STATUS_OK, failed)


def _monitor_peak(paths: Sequence[Path], column: str) -> float | None:
    peak: float | None = None
    for path in paths:
        frame = pd.read_csv(path, comment="#")
        if len(frame):
            value = float(frame[column].max())
            peak = value if peak is None else max(peak, value)
    return peak


def summarize_run(
    files: RunFiles,
    lookup: Callable[[str], RecordingSpec] = resolve,
) -> dict[str, object]:
    """Return the `RESAMPLING_RUNS` row of one run.

    Warm-up requests are left out. A request is a failure unless its status is
    `ok`. `median_s`, `p5_s`, and `p95_s` cover every measured request with the
    seconds of `charged_seconds`. `median_bytes` covers the successful measured
    requests. The rate is the successful measured requests over the time from
    the first measured dispatch to the last measured completion.
    `failure_rate` is the failures over every measured request.

    Args:
        files: The run's CSV files.
        lookup: Returns the spec of a recording id.

    Raises:
        ValueError: The run's request CSVs hold no rows.
    """
    frames = [read_requests(path) for path in files.requests]
    requests = pd.concat(frames, ignore_index=True)
    if requests.empty:
        raise ValueError(f"run {files.run}: no requests")
    first = requests.iloc[0]
    spec = lookup(first["recording"])
    window_s = float(first["window_s"])
    measured = requests[requests["warmup"] != TRUE].copy()
    measured["seconds"] = _numbers(measured, "seconds")
    measured["dispatch_s"] = _numbers(measured, "dispatch_s")
    measured["bytes"] = _numbers(measured, "bytes")
    good = measured[measured["status"] == STATUS_OK]
    total = len(measured)
    failures = total - len(good)
    row: dict[str, object] = {
        "run": files.run,
        "method": first["method"],
        "recording": first["recording"],
        "window_s": first["window_s"],
        "users": int(first["users"]),
        "generators": len(files.requests),
        "users_per_generator": int(
            requests.groupby("generator")["user"].nunique().max()
        ),
        "requests": total,
        "failures": failures,
        "failure_rate": failures / total if total else "",
        "requests_per_second": "",
        "median_s": "",
        "p5_s": "",
        "p95_s": "",
        "median_bytes": "",
        "raw_bytes": spec.channels
        * round(window_s * spec.rate_hz)
        * BYTES_PER_SAMPLE,
        "server_cpu_max": "",
        "generator_link_max": "",
        "repeated": (
            TRUE
            if read_header(files.requests[0]).get("repeated") == TRUE
            else "false"
        ),
        "link_limited": TRUE if files.link_limited else "false",
    }
    if total:
        seconds = charged_seconds(measured).to_numpy(dtype=float)
        low, middle, high = np.percentile(
            seconds, [LOW_PERCENTILE, 50, HIGH_PERCENTILE]
        )
        row["median_s"], row["p5_s"], row["p95_s"] = middle, low, high
    if len(good):
        row["median_bytes"] = float(np.median(good["bytes"].to_numpy(float)))
        done = measured["dispatch_s"] + measured["seconds"]
        span = float(done.max() - measured["dispatch_s"].min())
        if span > 0:
            row["requests_per_second"] = len(good) / span
    cpu = _monitor_peak(files.server_monitors, "cpu_percent")
    if cpu is not None:
        row["server_cpu_max"] = cpu
    receive = _monitor_peak(files.generator_monitors, "rx_bytes_per_s")
    if receive is not None:
        row["generator_link_max"] = receive / LINK_BYTES_PER_S
    return row


def deployment_metadata(runs: Sequence[RunFiles]) -> dict[str, str]:
    """Return the header keys that describe the machines of the runs.

    The server metadata of the first run that has any becomes one
    `server.<key>` item each. The load generator CSV headers give
    `generator.instance` and `generator.ami`, the distinct values in sorted
    order and joined by commas, and the earliest and latest `date` as
    `first_run_date` and `last_run_date`. A key whose source is absent is left
    out.
    """
    fields: dict[str, str] = {}
    for files in runs:
        if files.server_metadata is not None:
            loaded = json.loads(files.server_metadata.read_text())
            fields |= {
                f"server.{key}": str(value) for key, value in loaded.items()
            }
            break
    headers = [read_header(path) for files in runs for path in files.requests]
    for key in ("instance", "ami"):
        values = sorted({h[key] for h in headers if key in h})
        if values:
            fields[f"generator.{key}"] = ",".join(values)
    dates = sorted(h["date"] for h in headers if "date" in h)
    if dates:
        fields["first_run_date"] = dates[0]
        fields["last_run_date"] = dates[-1]
    return fields


def summarize(
    root: Path,
    results: Path,
    lookup: Callable[[str], RecordingSpec] = resolve,
) -> tuple[int, int]:
    """Write the merged request CSV and the per-run CSV.

    Args:
        root: Directory with one subdirectory per run.
        results: Directory that receives the two CSVs.
        lookup: Returns the spec of a recording id.

    Returns:
        The number of request rows and of run rows written.
    """
    runs = find_runs(root)
    header = (
        run_metadata()
        | deployment_metadata(runs)
        | {FAILED_REQUEST_SECONDS_KEY: str(REQUEST_TIMEOUT_S)}
    )
    request_rows = 0
    run_rows = []
    with CsvWriter(
        results / RESAMPLING_REQUESTS_FILE, RESAMPLING_REQUESTS, header
    ) as writer:
        for files in runs:
            for path in files.requests:
                frame = read_requests(path)
                for record in frame.to_dict("records"):
                    writer.write(record)
                    request_rows += 1
            run_rows.append(summarize_run(files, lookup))
    with CsvWriter(
        results / RESAMPLING_RUNS_FILE, RESAMPLING_RUNS, header
    ) as writer:
        writer.write_all(run_rows)
    return request_rows, len(run_rows)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m resampling_benchmark.summarize",
        description="Merge the CSVs of every resampling benchmark run.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results") / RESAMPLING_RAW_DIRECTORY,
        help=f"directory of the run directories (default results/{RESAMPLING_RAW_DIRECTORY})",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help=(
            f"directory of {RESAMPLING_REQUESTS_FILE} and {RESAMPLING_RUNS_FILE} "
            "(default results)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Write the two CSVs and print their row counts."""
    arguments = parse_arguments(argv)
    requests, runs = summarize(arguments.input, arguments.results)
    print(f"{requests} request rows, {runs} run rows")


if __name__ == "__main__":
    main()
