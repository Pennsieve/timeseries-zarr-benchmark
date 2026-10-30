"""Run one combination of the format benchmark in this process.

The process opens the file untimed by the reads, times the open once, then
times each read from the call to the decoded array in memory.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from format_benchmark.combinations import Combination, expected_rows, recording
from format_benchmark.positions import (
    MIN_READS,
    MIN_READS_KEY,
    READ_BUDGET_KEY,
    READ_BUDGET_S,
    positions,
)
from format_benchmark.readers import make_reader
from tszbench import schemas
from tszbench.csvmeta import CsvWriter
from tszbench.fs import StoreRoots, make_filesystem

ALL_CHANNELS = -1
"""Value of the `channel` column of an all-channel read."""


def add_root_options(parser: argparse.ArgumentParser) -> None:
    """Add the options that locate the three stores."""
    parser.add_argument("--local-root", help="Directory of the local store.")
    parser.add_argument("--http-root", help="Base URL of the HTTP store.")
    parser.add_argument(
        "--s3-root", help="Bucket and prefix of the S3 store, as bucket/prefix."
    )
    parser.add_argument(
        "--s3-endpoint-url",
        help="Endpoint of an S3-compatible server. Omit it for AWS.",
    )


def roots_of(arguments: argparse.Namespace) -> StoreRoots:
    """Return the store roots the options name. A missing root is empty."""
    return StoreRoots(
        local=Path(arguments.local_root or ""),
        http=arguments.http_root or "",
        s3=arguments.s3_root or "",
        s3_endpoint_url=arguments.s3_endpoint_url,
    )


def add_budget_options(parser: argparse.ArgumentParser) -> None:
    """Add the options that set the read budget and the minimum reads."""
    parser.add_argument(
        "--read-budget-s",
        type=float,
        default=READ_BUDGET_S,
        help=(
            "A combination stops after the read that brings its timed read "
            f"seconds to at least this value. Default: {READ_BUDGET_S}."
        ),
    )
    parser.add_argument(
        "--min-reads",
        type=int,
        default=MIN_READS,
        help=(
            "Reads a combination makes before the budget can stop it. "
            f"Default: {MIN_READS}."
        ),
    )


def check_budget_options(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> None:
    """Report through parser an option value that `run` would reject."""
    if not arguments.read_budget_s > 0:
        parser.error("--read-budget-s: must be above 0")
    if arguments.min_reads < 1:
        parser.error("--min-reads: must be at least 1")


def _commit(writer: CsvWriter, partial: Path, path: Path) -> None:
    """Close writer and move its partial file to path."""
    writer.close()
    partial.replace(path)


def run(
    combination: Combination,
    roots: StoreRoots,
    reads_path: Path,
    open_path: Path,
    max_repeats: int | None = None,
    read_budget_s: float = READ_BUDGET_S,
    min_reads: int = MIN_READS,
) -> None:
    """Run a combination and write its two CSVs.

    The worker makes the planned reads in order. It stops after read k when
    the timed seconds of reads 1 to k sum to at least read_budget_s and k is
    at least min_reads. A combination that stops early holds a prefix of the
    planned reads. A combination with fewer planned reads than min_reads makes
    them all.

    Each CSV is written under a `.partial` name and renamed when whole. A
    path that exists holds every row. Both headers record read_budget_s and
    min_reads.

    Args:
        combination: What to run.
        roots: Where the files of each store live.
        reads_path: CSV of the timed reads, `schemas.FORMAT_READS`.
        open_path: CSV of the open, one `schemas.FORMAT_OPENS` row.
        max_repeats: Cap on the number of reads, or None for the plan's count.
        read_budget_s: Seconds of timed reads at which the combination stops.
        min_reads: Reads made before read_budget_s can stop the combination.

    Raises:
        ValueError: read_budget_s is not above 0, or min_reads is below 1.
    """
    if not read_budget_s > 0:
        raise ValueError(f"read_budget_s is {read_budget_s}: must be above 0")
    if min_reads < 1:
        raise ValueError(f"min_reads is {min_reads}: must be at least 1")
    spec = recording(combination.recording)
    reads = positions(spec, combination.window_s, max_repeats)
    window_us = combination.window_s * 1_000_000
    header = {
        "combination": combination.stem,
        READ_BUDGET_KEY: str(read_budget_s),
        MIN_READS_KEY: str(min_reads),
    }
    key = {
        "recording": combination.recording,
        "store": combination.store,
        "format": combination.file_format,
        "read_kind": combination.read_kind,
        "window_s": combination.window_s,
    }
    filesystem = make_filesystem(combination.store, roots)
    reader = make_reader(
        combination.file_format, filesystem, combination.store, roots, spec
    )
    opened = False
    reads_partial = reads_path.with_name(reads_path.name + ".partial")
    open_partial = open_path.with_name(open_path.name + ".partial")
    reads_writer = CsvWriter(reads_partial, schemas.FORMAT_READS, header)
    open_writer = CsvWriter(open_partial, schemas.FORMAT_OPENS, header)
    try:
        before = filesystem.counts()
        started = perf_counter()
        reader.open()
        open_seconds = perf_counter() - started
        opened = True
        open_used = filesystem.counts() - before
        timed = 0.0
        made = 0
        for read in reads:
            channel = None if combination.read_kind == "all" else read.channel
            before = filesystem.counts()
            started = perf_counter()
            samples = reader.read(
                read.start_us, read.start_us + window_us, channel
            )
            seconds = perf_counter() - started
            used = filesystem.counts() - before
            reads_writer.write(
                key
                | {
                    "repeat": read.repeat,
                    "channel": ALL_CHANNELS if channel is None else channel,
                    "start_us": read.start_us,
                    "seconds": seconds,
                    "bytes": used.bytes,
                    "requests": used.requests,
                    "samples": samples.size,
                }
            )
            del samples
            timed += seconds
            made += 1
            if made >= min_reads and timed >= read_budget_s:
                break
        open_writer.write(
            key
            | {
                "seconds": open_seconds,
                "bytes": open_used.bytes,
                "requests": open_used.requests,
                "reads": made,
                "planned_reads": len(reads),
            }
        )
        _commit(open_writer, open_partial, open_path)
        _commit(reads_writer, reads_partial, reads_path)
    except BaseException:
        reads_partial.unlink(missing_ok=True)
        open_partial.unlink(missing_ok=True)
        raise
    finally:
        reads_writer.close()
        open_writer.close()
        if opened:
            reader.close()
        filesystem.close()


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of the worker command line."""
    parser = argparse.ArgumentParser(
        prog="python -m format_benchmark.worker",
        description="Run one combination of the format benchmark in this process.",
    )
    parser.add_argument("--recording", required=True)
    parser.add_argument("--store", required=True, choices=schemas.STORES)
    parser.add_argument("--format", required=True, choices=schemas.FORMAT_ROWS)
    parser.add_argument(
        "--read-kind", required=True, choices=schemas.READ_KINDS
    )
    parser.add_argument("--window-s", required=True, type=int)
    parser.add_argument(
        "--output", required=True, type=Path, help="CSV of the reads."
    )
    parser.add_argument(
        "--open-output", required=True, type=Path, help="CSV of the open."
    )
    parser.add_argument(
        "--max-repeats", type=int, help="Cap on the number of reads."
    )
    add_budget_options(parser)
    add_root_options(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the exit status."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    combination = Combination(
        arguments.recording,
        arguments.store,
        arguments.format,
        arguments.read_kind,
        arguments.window_s,
    )
    try:
        expected_rows(combination, arguments.max_repeats)
    except ValueError as error:
        parser.error(str(error))
    check_budget_options(parser, arguments)
    run(
        combination,
        roots_of(arguments),
        arguments.output,
        arguments.open_output,
        arguments.max_repeats,
        arguments.read_budget_s,
        arguments.min_reads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
