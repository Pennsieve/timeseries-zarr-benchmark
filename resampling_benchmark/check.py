"""Equality check of the two the resampling benchmark methods: `python -m resampling_benchmark.check`.

For each recording and window, the server and the precomputed read must return
equal arrays, with NaN equal to NaN. The check window holds a whole number of
level bins in every column and starts on a bin edge, so that both methods
reduce the same samples to every column. It is centered on a planted gap so
that the channel with gaps takes part.
"""

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import numpy.typing as npt

from resampling_benchmark.load.generator import ServerRequester
from resampling_benchmark.load.orchestrate import R4_LONGEST_WINDOW_S, WINDOWS_S
from resampling_benchmark.load.precomputed import (
    READ_THREADS,
    PrecomputedReader,
)
from resampling_benchmark.service.groups import sample_bounds_us
from resampling_benchmark.service.store import open_recording
from tszbench.csvmeta import CsvWriter, run_metadata
from tszbench.fs import StoreRoots, make_filesystem
from tszbench.minmax import COLUMNS
from tszbench.recordings import RECORDINGS, RecordingSpec, resolve
from tszbench.schemas import CHECK, CHECK_FILE

DECIMATION = 4
"""Raw samples folded by one bin of level 1, and the factor between levels."""

DEFAULT_TIMEOUT_S = 3_600.0
"""Seconds the server has to answer one check window. A whole-recording window
holds more than 100 GB of raw samples."""

Array = npt.NDArray[np.float32]


@dataclass(frozen=True)
class CheckWindow:
    """A window for the equality check.

    Attributes:
        window_s: Nominal window length in seconds.
        level: Level the precomputed read uses, 0 for raw samples.
        bins_per_column: Level bins in each column, or samples for level 0.
        first: Index of the first sample.
        stop: Index after the last sample.
        start_us: Start of the window in microseconds from sample 0.
        end_us: End of the window in microseconds, exclusive.
    """

    window_s: float
    level: int
    bins_per_column: int
    first: int
    stop: int
    start_us: int
    end_us: int


def nominal_entries(window_s: float, rate_hz: int, level: int) -> int:
    """Return the entries a window `[0, window_s)` holds at a level.

    An entry is a sample at level 0 and a bin at a higher level. A bin that the
    window covers in part counts as one entry.
    """
    period_us = int(DECIMATION**level) * 1_000_000
    return -(-round(window_s * 1_000_000) * rate_hz // period_us)


def edge_samples(rate_hz: int, bin_samples: int) -> int:
    """Return the spacing in samples of the allowed window starts.

    A start is a multiple of `bin_samples` whose time is a whole number of
    microseconds. Then the reader and `sample_range` agree on the first sample.
    """
    whole = rate_hz // math.gcd(rate_hz, 1_000_000)
    return math.lcm(bin_samples, whole)


def gap_ranges(spec: RecordingSpec) -> list[tuple[int, int]]:
    """Return the half-open sample range of every planted gap."""
    return [gap.sample_range(spec.rate_hz) for gap in spec.gaps]


def plan_window(
    spec: RecordingSpec,
    window_s: float,
    level: int,
    columns: int = COLUMNS,
) -> CheckWindow:
    """Return the check window of a nominal window and the level it reads.

    Every column of the window holds the same whole number of entries. An
    entry is a level bin, or a sample at level 0. That number is the entries of
    the nominal window per column, rounded to a whole number and at least 1. A
    recording too short for it gets the largest number that fits. The window
    starts on a level bin edge at a whole microsecond (see `edge_samples`). It
    is centered on the longest planted gap shorter than the window, or on the
    shortest gap when none is shorter, and moves to fit the recording.

    Args:
        spec: The recording.
        window_s: Nominal window length in seconds.
        level: Level the precomputed read chooses for the nominal window.
        columns: Pixel columns.

    Raises:
        ValueError: The recording holds fewer than `columns` whole bins.
    """
    entries = nominal_entries(window_s, spec.rate_hz, level)
    bin_samples = int(DECIMATION**level)
    usable = spec.samples_per_channel // bin_samples * bin_samples
    per_column = min(
        max(1, round(entries / columns)), usable // (columns * bin_samples)
    )
    count = columns * per_column * bin_samples
    edge = edge_samples(spec.rate_hz, bin_samples)
    if count > usable or per_column < 1:
        raise ValueError(
            f"{spec.id}: a check window of {count} samples at level {level} "
            f"does not fit the {spec.samples_per_channel} samples"
        )
    first = 0
    gaps = gap_ranges(spec)
    if gaps:
        shorter = [gap for gap in gaps if gap[1] - gap[0] < count]
        gap = (
            max(shorter, key=lambda item: item[1] - item[0])
            if shorter
            else min(gaps, key=lambda item: item[1] - item[0])
        )
        first = (gap[0] + gap[1]) // 2 - count // 2
        first -= first % edge
        latest = (usable - count) // edge * edge
        first = min(max(first, 0), latest)
    stop = first + count
    start_us, end_us = sample_bounds_us(first, stop, spec.rate_hz)
    return CheckWindow(
        window_s, level, per_column, first, stop, start_us, end_us
    )


def arrays_equal(left: Array, right: Array) -> bool:
    """Return True when the shapes match and every value matches or both are NaN."""
    return left.shape == right.shape and bool(
        np.array_equal(left, right, equal_nan=True)
    )


def check_recording(
    spec: RecordingSpec,
    windows: Sequence[float],
    choose_level: Callable[[int, int], int],
    server: Callable[[int, int], Array],
    precomputed: Callable[[int, int, int], Array],
) -> list[dict[str, object]]:
    """Return one `CHECK` row per window.

    A window whose read or comparison raises gets a row with `equal` set to
    `error:<class name>`, and the other windows still run.

    Args:
        spec: The recording.
        windows: Nominal window lengths in seconds.
        choose_level: Given `start_us` and `end_us` of a nominal window,
            returns the level the precomputed read chooses.
        server: Given `start_us` and `end_us`, returns the server array.
        precomputed: Given `start_us`, `end_us`, and a level, returns the
            array of the precomputed read.
    """
    rows = []
    for window_s in windows:
        row: dict[str, object] = {
            "recording": spec.id,
            "window_s": f"{window_s:g}",
            "start_us": "",
            "end_us": "",
            "level": "",
            "bins_per_column": "",
            "rows": "",
            "equal": "",
        }
        try:
            level = choose_level(0, round(window_s * 1_000_000))
            window = plan_window(spec, window_s, level)
            row.update(
                start_us=window.start_us,
                end_us=window.end_us,
                level=window.level,
                bins_per_column=window.bins_per_column,
            )
            left = server(window.start_us, window.end_us)
            row["rows"] = left.shape[1] if left.ndim == 3 else ""
            right = precomputed(window.start_us, window.end_us, window.level)
            row["equal"] = "true" if arrays_equal(left, right) else "false"
        except Exception as error:
            row["equal"] = f"error:{type(error).__name__}"
        rows.append(row)
    return rows


def windows_for(
    spec: RecordingSpec, chosen: Sequence[int] | None = None
) -> list[int]:
    """Return the benchmark windows that apply to a recording.

    Args:
        spec: The recording.
        chosen: When given, only these windows are kept.
    """
    limit = R4_LONGEST_WINDOW_S if spec.id == "R4" else spec.duration_s
    return [
        window
        for window in WINDOWS_S
        if window <= limit and (chosen is None or window in chosen)
    ]


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m resampling_benchmark.check",
        description="Check that the server and the precomputed read agree.",
    )
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--s3-root", required=True)
    parser.add_argument("--s3-endpoint-url")
    parser.add_argument(
        "--recordings",
        default=",".join(RECORDINGS),
        help="comma-separated recordings, R1 to R4 or R1-1h to R3-1h",
    )
    parser.add_argument(
        "--windows",
        help="comma-separated window lengths in seconds (default every window)",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="seconds to wait for each server answer (default %(default)s)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help=f"directory of {CHECK_FILE} (default results)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check over the chosen recordings.

    Returns:
        0 when every row is equal, 1 when a window differs or fails.
    """
    arguments = parse_arguments(argv)
    chosen = (
        None
        if arguments.windows is None
        else [int(window) for window in arguments.windows.split(",")]
    )
    roots = StoreRoots(
        local=Path(),
        http="",
        s3=arguments.s3_root,
        s3_endpoint_url=arguments.s3_endpoint_url or None,
    )
    failures = 0
    with (
        CsvWriter(
            arguments.results / CHECK_FILE, CHECK, run_metadata()
        ) as writer,
        httpx.Client(
            base_url=arguments.server_url, timeout=arguments.timeout_s
        ) as client,
        ThreadPoolExecutor(max_workers=READ_THREADS) as pool,
    ):
        for name in arguments.recordings.split(","):
            spec = resolve(name)
            filesystem = make_filesystem("s3", roots)
            bundle = open_recording(filesystem, roots, name)
            reader = PrecomputedReader(bundle, filesystem, pool)
            requester = ServerRequester(client, name)

            def server(
                start_us: int,
                end_us: int,
                requester: ServerRequester = requester,
            ) -> Array:
                return requester.fetch(start_us, end_us)

            def precomputed(
                start_us: int,
                end_us: int,
                level: int,
                reader: PrecomputedReader = reader,
            ) -> Array:
                return reader.read(start_us, end_us, level).array

            def choose(
                start_us: int, end_us: int, reader: PrecomputedReader = reader
            ) -> int:
                return reader.choose_level(start_us, end_us)

            try:
                rows = check_recording(
                    spec,
                    windows_for(spec, chosen),
                    choose,
                    server,
                    precomputed,
                )
            finally:
                bundle.close()
                filesystem.close()
            for row in rows:
                writer.write(row)
                if row["equal"] != "true":
                    failures += 1
                    reason = (
                        "arrays differ"
                        if row["equal"] == "false"
                        else str(row["equal"])
                    )
                    print(
                        f"{row['recording']} {row['window_s']} s: {reason}",
                        file=sys.stderr,
                    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
