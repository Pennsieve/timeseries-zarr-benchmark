from collections.abc import Callable
from pathlib import Path

import pytest

from plot.data import load_csv
from tszbench.csvmeta import CsvWriter
from tszbench.recordings import RECORDINGS
from tszbench.schemas import (
    CHECK,
    FORMAT_OPENS,
    FORMAT_READS,
    FORMAT_ROWS,
    GENERATION,
    READ_FORMATS,
    READ_KINDS,
    RESAMPLING_REQUESTS,
    RESAMPLING_RUNS,
    STORAGE,
    STORES,
    UPLOAD,
    generation_file,
    upload_file,
)

HEADER = {"date": "2026-10-23T00:00:00+00:00", "region": "us-east-1"}
READ_BUDGET_S = 3600
MIN_READS = 3
BUDGET_HEADER = HEADER | {
    "read_budget_s": str(READ_BUDGET_S),
    "min_reads": str(MIN_READS),
}
FAILED_REQUEST_SECONDS = 150
"""Differs from `REQUEST_TIMEOUT_S` so a test can tell the header from the constant."""
STOPPED_READS = 7
"""Reads of a combination in `write_open` that stopped at the read budget."""
WINDOWS = (1, 10, 60)
REPEATS = 21
"""Reads per combination. The median of 21 reads is the 11th, 10 steps above the first."""
SPREAD = 0.01
"""Each repeat adds this fraction of the base seconds."""
MEDIAN_FACTOR = 1 + SPREAD * (REPEATS - 1) / 2


def base_seconds(
    file_format: str, store: str, read_kind: str, window_s: int
) -> float:
    format_index = FORMAT_ROWS.index(file_format) + 1
    store_index = STORES.index(store) + 1
    return 0.001 * format_index * store_index * window_s + (
        0.01 if read_kind == "all" else 0.0
    )


def base_bytes(file_format: str, window_s: int) -> int:
    return 1000 * (FORMAT_ROWS.index(file_format) + 1) * window_s


def overhead_base(store: str) -> float:
    return 0.005 * (STORES.index(store) + 1)


def write_format_benchmark(path: Path) -> None:
    with CsvWriter(path, FORMAT_READS, BUDGET_HEADER) as writer:
        for recording in ("R3-1h", "R1-1h"):
            for store in STORES:
                for kind in READ_KINDS:
                    for window in WINDOWS:
                        for name in (*READ_FORMATS, "zarr-direct"):
                            base = base_seconds(name, store, kind, window)
                            for repeat in range(REPEATS):
                                writer.write(
                                    {
                                        "recording": recording,
                                        "store": store,
                                        "format": name,
                                        "read_kind": kind,
                                        "window_s": window,
                                        "repeat": repeat,
                                        "channel": -1 if kind == "all" else 0,
                                        "start_us": repeat * 1_000_000,
                                        "seconds": base * (1 + SPREAD * repeat),
                                        "bytes": base_bytes(name, window),
                                        "requests": 2,
                                        "samples": 100,
                                    }
                                )
        for store in STORES:
            for repeat in range(REPEATS):
                writer.write(
                    {
                        "recording": "R3-1h",
                        "store": store,
                        "format": "overhead",
                        "read_kind": "all",
                        "window_s": 10,
                        "repeat": repeat,
                        "channel": -1,
                        "start_us": 0,
                        "seconds": overhead_base(store) * (1 + SPREAD * repeat),
                        "bytes": 1,
                        "requests": 1,
                        "samples": 0,
                    }
                )


RUN_WINDOWS = (10, 600)
USERS = (1, 10, 100, 1000)


def run_row(
    method: str,
    recording: str,
    window: int,
    users: int,
    *,
    median: float,
    failure_rate: float = 0.0,
    rate: float = 1.0,
    repeated: bool = False,
    link_limited: bool = False,
    requests: int = 90,
    median_bytes: float = 1000.0,
) -> dict[str, object]:
    return {
        "run": f"{method}-{recording}-{window}-{users}",
        "method": method,
        "recording": recording,
        "window_s": window,
        "users": users,
        "generators": 1,
        "users_per_generator": users,
        "requests": requests,
        "failures": round(failure_rate * requests),
        "failure_rate": failure_rate,
        "requests_per_second": rate,
        "median_s": median,
        "p5_s": median / 2,
        "p95_s": median * 3,
        "median_bytes": median_bytes,
        "raw_bytes": RECORDINGS[recording].raw_bytes * window // 86_400,
        "server_cpu_max": 50.0,
        "generator_link_max": 0.1,
        "repeated": "true" if repeated else "false",
        "link_limited": "true" if link_limited else "false",
        "not_run": "false",
    }


def run_rows() -> list[dict[str, object]]:
    rows = []
    for recording in ("R3", "R1"):
        for window in RUN_WINDOWS:
            for users in USERS:
                server_fails = recording == "R3" and (
                    users == 1000 or (window == 600 and users == 100)
                )
                rows.append(
                    run_row(
                        "server",
                        recording,
                        window,
                        users,
                        median=window * users / 100,
                        failure_rate=0.2 if server_fails else 0.0,
                        rate=float(users) ** 0.5 + window / 100,
                    )
                )
                rows.append(
                    run_row(
                        "precomputed",
                        recording,
                        window,
                        users,
                        median=0.5 * window**0.5 * users**0.25,
                        rate=users * 2.0,
                        median_bytes=4_000_000 + window,
                    )
                )
    rows.append(
        run_row(
            "server",
            "R3",
            10,
            100,
            median=7.0,
            repeated=True,
            rate=5.0,
            requests=45,
        )
    )
    return rows


def write_runs(path: Path) -> None:
    with CsvWriter(
        path,
        RESAMPLING_RUNS,
        HEADER | {"failed_request_seconds": str(FAILED_REQUEST_SECONDS)},
    ) as writer:
        writer.write_all(run_rows())


def write_requests(path: Path) -> None:
    with CsvWriter(path, RESAMPLING_REQUESTS, HEADER) as writer:
        for method, gets in (("server", 4), ("precomputed", 3)):
            for window in RUN_WINDOWS:
                for sequence, (warmup, status) in enumerate(
                    (
                        ("true", "ok"),
                        ("false", "ok"),
                        ("false", "ok"),
                        ("false", "timeout"),
                    )
                ):
                    writer.write(
                        {
                            "run": "r",
                            "method": method,
                            "recording": "R3",
                            "window_s": window,
                            "users": 1,
                            "generator": 0,
                            "user": 0,
                            "sequence": sequence,
                            "start_us": 0,
                            "dispatch_s": 0.0,
                            "seconds": 1.0,
                            "bytes": 10,
                            "s3_requests": gets if status == "ok" else 99,
                            "s3_bytes": 10,
                            "status": status,
                            "warmup": warmup,
                            "level": 0,
                            "read_seconds": 0.5,
                            "reduce_seconds": 0.5,
                        }
                    )


VERSION_HEADER = HEADER | {
    "instance": "c6i.xlarge",
    "version.timeseries-zarr": "0.1.0a7",
    "version.tsdownsample": "0.1.5.1",
    "version.pynwb": "absent",
}


def write_upload(path: Path) -> None:
    with CsvWriter(path, UPLOAD, VERSION_HEADER) as writer:
        for name, rate in (("edf", 100e6), ("tszarr", 40e6)):
            for seconds in (1.0, 2.0, 3.0):
                writer.write(
                    {
                        "recording": "R3",
                        "format": name,
                        "seconds": seconds,
                        "bytes": 1000,
                        "objects": 1,
                        "bytes_per_second": rate * seconds / 2,
                    }
                )


def write_open(path: Path, *, stopped: int = 1, header: bool = True) -> None:
    """Write 12 opens. The first `stopped` of each store ends at the budget."""
    meta = VERSION_HEADER | {"instance": "c6i.2xlarge"}
    if header:
        meta |= BUDGET_HEADER
    with CsvWriter(path, FORMAT_OPENS, meta) as writer:
        for store in STORES:
            for index, seconds in enumerate((0.1, 0.3, 0.2)):
                writer.write(
                    {
                        "recording": "R3",
                        "store": store,
                        "format": "tszarr",
                        "read_kind": "one",
                        "window_s": 1,
                        "seconds": seconds * (STORES.index(store) + 1),
                        "bytes": 2000,
                        "requests": 2,
                        "reads": STOPPED_READS if index < stopped else REPEATS,
                        "planned_reads": REPEATS,
                    }
                )
            writer.write(
                {
                    "recording": "R3",
                    "store": store,
                    "format": "overhead",
                    "read_kind": "one",
                    "window_s": 1,
                    "seconds": 0.0,
                    "bytes": 0,
                    "requests": 0,
                    "reads": REPEATS,
                    "planned_reads": REPEATS,
                }
            )


def write_generation(path: Path, recording: str, scale: float) -> None:
    with CsvWriter(path, GENERATION, VERSION_HEADER) as writer:
        for name, seconds in (("edf", 10.0), ("tszarr", 20.0)):
            writer.write(
                {
                    "recording": recording,
                    "format": name,
                    "seconds": seconds * scale,
                    "bytes": 1000,
                    "objects": 1,
                }
            )


def write_check(path: Path, equal: tuple[str, ...] = ("true",) * 3) -> None:
    with CsvWriter(path, CHECK, VERSION_HEADER) as writer:
        for window, value in zip((10, 60, 600), equal, strict=True):
            writer.write(
                {
                    "recording": "R3",
                    "window_s": window,
                    "start_us": 0,
                    "end_us": window * 1_000_000,
                    "level": 2,
                    "bins_per_column": 4,
                    "rows": 100,
                    "equal": value,
                }
            )


def write_storage(path: Path) -> None:
    raw = RECORDINGS["R3"].raw_bytes
    rows = [
        ("edf", "", raw // 2, 1),
        ("nwb-hdf5-time", "", raw, 1),
        ("tszarr", "", 4 * raw, 500),
        ("tszarr", "raw", raw, 100),
        ("tszarr", "minmax", raw // 2, 50),
        ("tszarr", "mean", raw // 4, 40),
        ("tszarr", "var", raw // 4, 40),
        ("tszarr", "count", raw // 8, 30),
        ("tszarr", "metadata", 1000, 3),
        ("tszarr-raw", "", raw + 500, 90),
    ]
    with CsvWriter(path, STORAGE, HEADER) as writer:
        for name, member, size, objects in rows:
            writer.write(
                {
                    "recording": "R3",
                    "format": name,
                    "member": member,
                    "bytes": size,
                    "objects": objects,
                }
            )


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "results"
    directory.mkdir()
    write_format_benchmark(directory / "format-reads.csv")
    write_runs(directory / "resampling-runs.csv")
    write_requests(directory / "resampling-requests.csv")
    write_storage(directory / "storage.csv")
    write_upload(directory / upload_file("R3"))
    write_open(directory / "format-opens.csv")
    write_generation(directory / generation_file("R3-1h"), "R3-1h", 1.0)
    write_generation(directory / generation_file("R3"), "R3", 24.0)
    write_check(directory / "resampling-check.csv")
    return directory


@pytest.fixture
def format_benchmark(results_dir: Path):
    return load_csv(results_dir / "format-reads.csv", FORMAT_READS)


@pytest.fixture
def runs(results_dir: Path):
    return load_csv(results_dir / "resampling-runs.csv", RESAMPLING_RUNS)


@pytest.fixture
def requests(results_dir: Path):
    return load_csv(
        results_dir / "resampling-requests.csv", RESAMPLING_REQUESTS
    )


@pytest.fixture
def storage(results_dir: Path):
    return load_csv(results_dir / "storage.csv", STORAGE)


@pytest.fixture
def seconds() -> Callable[[str, str, str, int], float]:
    return base_seconds


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch):
    """Return a function that records the figure a module saves."""
    from plot import style

    def install(module: object) -> list:
        saved: list = []

        def save(figure, out_dir, stem):
            saved.append(figure)
            return style.save_figure(figure, out_dir, stem)

        monkeypatch.setattr(module, "save_figure", save)
        return saved

    return install


def trim_reads(frame, count: int):
    """Return a copy of a format-reads frame where one combination has `count` reads."""
    first = frame.iloc[0]
    same = (
        (frame["recording"] == first["recording"])
        & (frame["store"] == first["store"])
        & (frame["format"] == first["format"])
        & (frame["read_kind"] == first["read_kind"])
        & (frame["window_s"] == first["window_s"])
    )
    return frame[~(same & (frame["repeat"] >= count))]


def mark_link_limited(
    frame, method: str, recording: str, window: int, users: int
):
    """Return a copy of a runs frame with one run marked link limited."""
    changed = frame.copy()
    mask = (
        (changed["method"] == method)
        & (changed["recording"] == recording)
        & (changed["window_s"] == window)
        & (changed["users"] == users)
        & (changed["repeated"] == "false")
    )
    changed.loc[mask, "link_limited"] = "true"
    return changed


def ring_points(axes) -> list[tuple[float, float]]:
    """Return the points of the open rings that mark link limited runs."""
    return [
        point
        for axis in axes
        for line in axis.get_lines()
        if line.get_marker() == "o" and line.get_markerfacecolor() == "none"
        for point in zip(line.get_xdata(), line.get_ydata(), strict=True)
    ]


NOT_RUN_BLANKED = (
    "generators",
    "users_per_generator",
    "requests",
    "failures",
    "failure_rate",
    "requests_per_second",
    "median_s",
    "p5_s",
    "p95_s",
    "median_bytes",
    "raw_bytes",
    "server_cpu_max",
    "generator_link_max",
)


def mark_not_run(frame, method: str, recording: str, window: int, users: int):
    """Return a copy of a runs frame with one run turned into a not-run row."""
    changed = frame.copy()
    mask = (
        (changed["method"] == method)
        & (changed["recording"] == recording)
        & (changed["window_s"] == window)
        & (changed["users"] == users)
        & (changed["repeated"] == "false")
    )
    changed.loc[mask, "not_run"] = "true"
    changed.loc[mask, list(NOT_RUN_BLANKED)] = float("nan")
    return changed
