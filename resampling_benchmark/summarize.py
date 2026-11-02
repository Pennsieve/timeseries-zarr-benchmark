"""Merge the CSVs of the resampling benchmark runs: `python -m resampling_benchmark.summarize`.

Reads the request CSVs of every run directory and the monitor CSVs next to
them. Writes `tszbench.schemas.RESAMPLING_REQUESTS_FILE` and
`tszbench.schemas.RESAMPLING_RUNS_FILE` to the results directory.
"""

import argparse
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from resampling_benchmark.load.generator import (
    LINK_BYTES_PER_S,
    REQUEST_TIMEOUT_S,
    SATURATION_WINDOW_S,
    STATUS_OK,
)
from resampling_benchmark.load.monitor import SAMPLE_INTERVAL_S
from tszbench.csvmeta import CsvWriter, read_header, run_metadata
from tszbench.recordings import RecordingSpec, resolve
from tszbench.schemas import (
    CLIENT_LIMITED_FILE,
    CLIENT_LIMITS,
    GENERATOR_MONITOR_PATTERN,
    METHODS,
    NOT_RUN_FILE,
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
        client_limited: True when the run directory holds a
            `tszbench.schemas.CLIENT_LIMITED_FILE`.
        client_limit: The resource that file names, or an empty string when
            the file is absent or empty.
        not_run: True for the directory of a run that was skipped. It holds a
            `tszbench.schemas.NOT_RUN_FILE` and no request CSV.
    """

    run: str
    requests: tuple[Path, ...]
    server_monitors: tuple[Path, ...]
    generator_monitors: tuple[Path, ...]
    server_metadata: Path | None = None
    client_limited: bool = False
    client_limit: str = ""
    not_run: bool = False


def find_runs(root: Path) -> list[RunFiles]:
    """Return the runs under root.

    A directory counts when it holds at least one request CSV, or when it holds
    a `tszbench.schemas.NOT_RUN_FILE` and no request CSV.
    """
    runs = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        requests = tuple(sorted(directory.glob(REQUESTS_PATTERN)))
        if not requests:
            if (directory / NOT_RUN_FILE).is_file():
                runs.append(
                    RunFiles(
                        run=directory.name,
                        requests=(),
                        server_monitors=(),
                        generator_monitors=(),
                        not_run=True,
                    )
                )
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
                client_limited=(directory / CLIENT_LIMITED_FILE).exists(),
                client_limit=_client_limit(directory),
            )
        )
    return runs


def _client_limit(directory: Path) -> str:
    marker = directory / CLIENT_LIMITED_FILE
    resource = marker.read_text().strip() if marker.exists() else ""
    if resource and resource not in CLIENT_LIMITS:
        raise ValueError(
            f"{marker}: holds {resource!r}, must be one of {CLIENT_LIMITS}"
        )
    return resource


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


def measured_requests(requests: Any) -> Any:
    """Return the measured rows of request rows, with numeric columns.

    Warm-up rows are left out. `seconds`, `dispatch_s`, `bytes`,
    `s3_requests`, and `s3_bytes` become numbers, and a value that is not a
    number becomes NaN.
    """
    measured = requests[requests["warmup"] != TRUE].copy()
    for column in (
        "seconds",
        "dispatch_s",
        "bytes",
        "s3_requests",
        "s3_bytes",
    ):
        measured[column] = _numbers(measured, column)
    return measured


def median_charged_seconds(paths: Sequence[Path]) -> float | None:
    """Return the median of `charged_seconds` over the measured requests.

    Args:
        paths: Request CSVs of one run.

    Returns:
        The median, or None when the CSVs hold no measured request.
    """
    frames = [read_requests(path) for path in paths]
    if not frames:
        return None
    measured = measured_requests(pd.concat(frames, ignore_index=True))
    if measured.empty:
        return None
    return float(np.median(charged_seconds(measured).to_numpy(dtype=float)))


def _monitor_peak(paths: Sequence[Path], column: str) -> float | None:
    peak: float | None = None
    for path in paths:
        frame = pd.read_csv(path, comment="#")
        if len(frame) and column in frame:
            value = float(frame[column].max())
            peak = value if peak is None else max(peak, value)
    return peak


def measured_span(paths: Sequence[Path]) -> tuple[float, float] | None:
    """Return the Unix times that bound the measured part of a run.

    The measured part starts at the first measured dispatch of any generator
    and ends at the last measured response. Each request CSV gives its
    dispatch times from the `start_at` of its header.

    Args:
        paths: Request CSVs of one run.

    Returns:
        The start and the end, or None when no CSV has a `start_at` and a
        measured request that was sent.
    """
    starts: list[float] = []
    ends: list[float] = []
    for path in paths:
        origin = read_header(path).get("start_at")
        if origin is None:
            continue
        measured = measured_requests(read_requests(path))
        dispatched = measured["dispatch_s"].dropna()
        finished = (measured["dispatch_s"] + measured["seconds"]).dropna()
        if dispatched.empty or finished.empty:
            continue
        starts.append(float(origin) + float(dispatched.min()))
        ends.append(float(origin) + float(finished.max()))
    if not starts:
        return None
    return min(starts), max(ends)


def monitor_samples(path: Path, column: str, span: tuple[float, float]) -> Any:
    """Return the values of a monitor column over the samples of a span.

    A sample at `time_s` covers the `SAMPLE_INTERVAL_S` before it. A sample
    counts when that interval overlaps the span.

    Args:
        path: Monitor CSV.
        column: Column of `tszbench.schemas.MONITOR`.
        span: Start and end in Unix seconds.
    """
    frame = pd.read_csv(path, comment="#")
    inside = frame[
        (frame["time_s"] > span[0])
        & (frame["time_s"] - SAMPLE_INTERVAL_S < span[1])
    ]
    return inside[column].to_numpy(dtype=float)


def mean_over_span(
    path: Path, column: str, span: tuple[float, float]
) -> float | None:
    """Return the mean of a monitor column over a span, or None without samples."""
    values = monitor_samples(path, column, span)
    return float(values.mean()) if values.size else None


def sustained_mean(
    path: Path,
    column: str,
    span: tuple[float, float],
    window_s: int = SATURATION_WINDOW_S,
) -> float | None:
    """Return the highest rolling mean of a monitor column over a span.

    The mean covers `window_s` consecutive samples. A span with fewer samples
    gives the mean of all of them.

    Returns:
        The value, or None when the span holds no sample.
    """
    values = monitor_samples(path, column, span)
    if values.size == 0:
        return None
    if values.size <= window_s:
        return float(values.mean())
    sums = np.convolve(values, np.ones(window_s), mode="valid")
    return float(sums.max() / window_s)


class SustainedLoad(NamedTuple):
    """The highest sustained load of any load generator of a run.

    Attributes:
        cpu: Sustained CPU use as a fraction of the generator's CPU, 0 to 1.
        link: Sustained receive rate as a fraction of `LINK_BYTES_PER_S`.
    """

    cpu: float
    link: float


def sustained_load(
    request_paths: Iterable[Path], monitor_paths: Iterable[Path]
) -> SustainedLoad | None:
    """Return the highest sustained CPU use and receive rate of any generator.

    Each value is the highest `SATURATION_WINDOW_S` rolling mean over the
    measured part of the run, as `measured_span` bounds it.

    Args:
        request_paths: Request CSVs of the run.
        monitor_paths: Monitor CSVs of the load generators. A path that does
            not exist is skipped.

    Returns:
        The two fractions, or None when no monitor sample falls inside the
        measured part.
    """
    span = measured_span(list(request_paths))
    if span is None:
        return None
    cpu: list[float] = []
    link: list[float] = []
    for path in monitor_paths:
        if not path.exists():
            continue
        busy = sustained_mean(path, "cpu_percent", span)
        received = sustained_mean(path, "rx_bytes_per_s", span)
        if busy is not None and received is not None:
            cpu.append(busy / 100.0)
            link.append(received / LINK_BYTES_PER_S)
    if not cpu:
        return None
    return SustainedLoad(max(cpu), max(link))


def generator_cpu_seconds(
    monitor_paths: Sequence[Path], span: tuple[float, float]
) -> float | None:
    """Return the CPU time that the load generators used over a span.

    The unit is seconds of one vCPU. Each monitor CSV gives the vCPU count of
    its generator in the `vcpus` header key.

    Returns:
        The sum over the generators, or None when a monitor has no `vcpus`
        key or no sample in the span, or when there is no monitor.
    """
    if not monitor_paths:
        return None
    total = 0.0
    for path in monitor_paths:
        vcpus = read_header(path).get("vcpus")
        busy = mean_over_span(path, "cpu_percent", span)
        if vcpus is None or busy is None:
            return None
        total += busy / 100.0 * int(vcpus) * (span[1] - span[0])
    return total


def not_run_row(run: str) -> dict[str, object]:
    """Return the `RESAMPLING_RUNS` row of a run that was skipped.

    The method, recording, window, and users come from the run name, such as
    `server-R3-1h-600-100`. Every statistic is empty.

    Raises:
        ValueError: The run name does not have the form
            `<method>-<recording>-<window>-<users>`.
    """
    method, _, rest = run.partition("-")
    recording, window_s, users = (
        rest.rsplit("-", 2) if rest.count("-") >= 2 else ("", "", "")
    )
    if method not in METHODS or not users.isdecimal():
        raise ValueError(f"run {run}: not a run name")
    row: dict[str, object] = dict.fromkeys(RESAMPLING_RUNS, "")
    row |= {
        "run": run,
        "method": method,
        "recording": recording,
        "window_s": window_s,
        "users": int(users),
        "repeated": "false",
        "client_limited": "false",
        "not_run": TRUE,
    }
    return row


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

    A run that was skipped gets `not_run_row`.

    Args:
        files: The run's CSV files.
        lookup: Returns the spec of a recording id.

    Raises:
        ValueError: The run's request CSVs hold no rows.
    """
    if files.not_run:
        return not_run_row(files.run)
    frames = [read_requests(path) for path in files.requests]
    requests = pd.concat(frames, ignore_index=True)
    if requests.empty:
        raise ValueError(f"run {files.run}: no requests")
    first = requests.iloc[0]
    spec = lookup(first["recording"])
    window_s = float(first["window_s"])
    measured = measured_requests(requests)
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
        "repeated": (
            TRUE
            if read_header(files.requests[0]).get("repeated") == TRUE
            else "false"
        ),
        "client_limited": TRUE if files.client_limited else "false",
        "client_limit": files.client_limit,
        "not_run": "false",
    }
    row |= dict.fromkeys(
        (
            "server_cpu_sustained",
            "server_memory_max",
            "generator_link_sustained",
            "generator_cpu_sustained",
            "generator_cpu_s_per_request",
            "s3_requests_per_second",
            "s3_bytes_per_second",
        ),
        "",
    )
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
    memory = _monitor_peak(files.server_monitors, "memory_percent")
    if memory is not None:
        row["server_memory_max"] = memory
    seconds = _measured_seconds(measured)
    if seconds is not None:
        row["s3_requests_per_second"] = (
            float(measured["s3_requests"].sum()) / seconds
        )
        row["s3_bytes_per_second"] = float(measured["s3_bytes"].sum()) / seconds
    unix_span = measured_span(files.requests)
    if unix_span is not None:
        _add_sustained(row, files, unix_span, total)
    return row


def _measured_seconds(measured: Any) -> float | None:
    """Return the seconds from the first measured dispatch to the last response."""
    done = (measured["dispatch_s"] + measured["seconds"]).dropna()
    started = measured["dispatch_s"].dropna()
    if done.empty or started.empty:
        return None
    seconds = float(done.max() - started.min())
    return seconds if seconds > 0 else None


def _add_sustained(
    row: dict[str, object],
    files: RunFiles,
    span: tuple[float, float],
    measured_total: int,
) -> None:
    """Fill the columns that need the Unix time of the measured part."""
    sustained = [
        value
        for path in files.server_monitors
        if (value := sustained_mean(path, "cpu_percent", span)) is not None
    ]
    if sustained:
        row["server_cpu_sustained"] = max(sustained)
    load = sustained_load(files.requests, files.generator_monitors)
    if load is not None:
        row["generator_cpu_sustained"] = load.cpu
        row["generator_link_sustained"] = load.link
    cost = generator_cpu_seconds(files.generator_monitors, span)
    if cost is not None and measured_total:
        row["generator_cpu_s_per_request"] = cost / measured_total


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
