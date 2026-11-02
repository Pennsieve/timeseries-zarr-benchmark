"""Run every combination of the format benchmark, each in a fresh process.

Before each process the command line runs every `--drop-caches` command. A
combination whose CSVs are whole is skipped. A combination stops early when the
seconds of its timed reads reach `--read-budget-s`, once it made `--min-reads`
reads. A rerun with the same two options skips it. At the end the whole CSVs of
every combination in the results directory, including the ones that earlier
commands ran, merge into the two result CSVs of the format benchmark.
"""

import argparse
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from format_benchmark.combinations import (
    Combination,
    complete_parts,
    is_complete,
    plan,
    recording,
)
from format_benchmark.positions import MIN_READS_KEY, READ_BUDGET_KEY, WINDOWS_S
from format_benchmark.worker import (
    add_budget_options,
    add_root_options,
    check_budget_options,
    roots_of,
)
from tszbench import schemas
from tszbench.csvmeta import CsvWriter, read_rows, run_metadata
from tszbench.fs import StoreRoots
from tszbench.recordings import FORMAT_RECORDINGS

SERVER_TIMEOUT_S = 10.0
"""Timeout of the request that reads the Server header."""

UNKNOWN = "unknown"


def server_header(http_root: str) -> str:
    """Return the `Server` header of the file server, or `unknown`.

    The header names the nginx build. The request is a HEAD of the 1-byte
    object.
    """
    url = http_root.rstrip("/") + "/" + schemas.ONE_BYTE_OBJECT
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(
            request, timeout=SERVER_TIMEOUT_S
        ) as response:
            return str(response.headers.get("Server", UNKNOWN))
    except urllib.error.HTTPError as error:
        return str(error.headers.get("Server", UNKNOWN))
    except (OSError, ValueError):
        return UNKNOWN


def nginx_build(http_root: str) -> str:
    """Return the first line of the file server's `nginx -V` output, or `unknown`.

    The line comes from `schemas.NGINX_VERSION_FILE` under the HTTP root.
    """
    url = http_root.rstrip("/") + "/" + schemas.NGINX_VERSION_FILE
    try:
        with urllib.request.urlopen(url, timeout=SERVER_TIMEOUT_S) as response:
            lines = response.read().decode(errors="replace").splitlines()
    except (OSError, ValueError):
        return UNKNOWN
    return lines[0].strip() if lines and lines[0].strip() else UNKNOWN


def worker_command(
    combination: Combination,
    roots: StoreRoots,
    results: Path,
    max_repeats: int | None,
    read_budget_s: float,
    min_reads: int,
) -> list[str]:
    """Return the command line that runs one combination in a new process."""
    command = [
        sys.executable,
        "-m",
        "format_benchmark.worker",
        "--recording",
        combination.recording,
        "--store",
        combination.store,
        "--format",
        combination.file_format,
        "--read-kind",
        combination.read_kind,
        "--window-s",
        str(combination.window_s),
        "--output",
        str(combination.reads_path(results)),
        "--open-output",
        str(combination.open_path(results)),
        "--local-root",
        str(roots.local),
        "--http-root",
        roots.http,
        "--s3-root",
        roots.s3,
        "--read-budget-s",
        str(read_budget_s),
        "--min-reads",
        str(min_reads),
    ]
    if roots.s3_endpoint_url is not None:
        command += ["--s3-endpoint-url", roots.s3_endpoint_url]
    if max_repeats is not None:
        command += ["--max-repeats", str(max_repeats)]
    return command


def run_worker(
    combination: Combination,
    roots: StoreRoots,
    results: Path,
    max_repeats: int | None,
    read_budget_s: float,
    min_reads: int,
) -> int:
    """Run one combination in a fresh process and return its exit status."""
    combination.reads_path(results).parent.mkdir(parents=True, exist_ok=True)
    command = worker_command(
        combination, roots, results, max_repeats, read_budget_s, min_reads
    )
    return subprocess.run(command, check=False).returncode


def merge(
    results: Path,
    combinations: Sequence[Combination],
    header: dict[str, str],
) -> tuple[int, int]:
    """Merge the CSVs of combinations into the two result CSVs.

    A combination whose CSVs do not exist is left out. Rows keep the order of
    combinations, then of reads. The result CSVs hold only these combinations.

    Returns:
        The number of read rows and of open rows written.
    """
    counts = []
    for name, columns, path_of in (
        (
            schemas.FORMAT_READS_FILE,
            schemas.FORMAT_READS,
            Combination.reads_path,
        ),
        (
            schemas.FORMAT_OPENS_FILE,
            schemas.FORMAT_OPENS,
            Combination.open_path,
        ),
    ):
        count = 0
        with CsvWriter(results / name, columns, header) as writer:
            for combination in combinations:
                path = path_of(combination, results)
                if path.is_file():
                    rows = read_rows(path)
                    writer.write_all(rows)
                    count += len(rows)
        counts.append(count)
    return counts[0], counts[1]


def parse_list(
    parser: argparse.ArgumentParser,
    name: str,
    text: str,
    choices: Sequence[str] | None,
) -> list[str]:
    """Return the values of a comma-separated list, each one of choices.

    Any value is accepted when choices is None. The parser reports an error
    and exits for an empty list or a value outside choices.
    """
    values = [item for item in text.split(",") if item]
    if not values:
        parser.error(f"{name}: names no value")
    for value in values:
        if choices is not None and value not in choices:
            parser.error(f"{name} {value!r}: must be one of {list(choices)}")
    return values


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m format_benchmark",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--recordings",
        default=",".join(FORMAT_RECORDINGS),
        help=(
            "Comma-separated recordings, R1 to R4 or R1-1h to R3-1h. "
            f"Default: {','.join(FORMAT_RECORDINGS)}."
        ),
    )
    parser.add_argument(
        "--stores",
        default=",".join(schemas.STORES),
        help="Comma-separated stores. Default: all three.",
    )
    parser.add_argument(
        "--formats",
        default=",".join(schemas.FORMAT_ROWS),
        help="Comma-separated rows of the format benchmark. Default: all.",
    )
    parser.add_argument(
        "--read-kinds",
        default=",".join(schemas.READ_KINDS),
        help=(
            "Comma-separated read kinds: one, channels4, channels16, "
            "channels64, and all. Default: all five."
        ),
    )
    parser.add_argument(
        "--windows",
        default=",".join(str(window) for window in WINDOWS_S),
        help="Comma-separated window lengths in seconds. Default: all four.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help=(
            f"Directory of {schemas.FORMAT_READS_FILE}, {schemas.FORMAT_OPENS_FILE}, "
            f"and {schemas.FORMAT_PARTS_DIRECTORY}. Default: results."
        ),
    )
    parser.add_argument(
        "--drop-caches",
        action="append",
        default=[],
        metavar="COMMAND",
        help=(
            "Shell command that drops a page cache. It runs before every "
            "process. Repeatable."
        ),
    )
    parser.add_argument(
        "--max-repeats",
        type=int,
        help="Cap on the reads of each combination, for smoke runs.",
    )
    add_budget_options(parser)
    add_root_options(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the exit status."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    recordings = parse_list(parser, "recording", arguments.recordings, None)
    for name in recordings:
        try:
            recording(name)
        except ValueError as error:
            parser.error(str(error))
    stores = parse_list(parser, "store", arguments.stores, schemas.STORES)
    formats = parse_list(
        parser, "format", arguments.formats, schemas.FORMAT_ROWS
    )
    kinds = parse_list(
        parser, "read kind", arguments.read_kinds, schemas.READ_KINDS
    )
    windows = [
        int(window)
        for window in parse_list(
            parser,
            "window",
            arguments.windows,
            [str(window) for window in WINDOWS_S],
        )
    ]
    if arguments.max_repeats is not None and arguments.max_repeats < 1:
        parser.error("--max-repeats: must be at least 1")
    check_budget_options(parser, arguments)
    read_budget_s: float = arguments.read_budget_s
    min_reads: int = arguments.min_reads
    for store, option in (
        ("local", "local_root"),
        ("http", "http_root"),
        ("s3", "s3_root"),
    ):
        if store in stores and not getattr(arguments, option):
            parser.error(f"store {store}: needs --{option.replace('_', '-')}")
    roots = roots_of(arguments)
    results: Path = arguments.results
    measured, skipped = plan(recordings, stores, formats, kinds, windows)
    for combination, reason in skipped:
        print(
            f"skip {combination.stem} of {combination.recording}: {reason}",
            file=sys.stderr,
        )
    failures: list[Combination] = []
    for number, combination in enumerate(measured, start=1):
        label = f"{combination.recording}/{combination.stem}"
        if is_complete(
            combination,
            results,
            arguments.max_repeats,
            read_budget_s,
            min_reads,
        ):
            print(f"[{number}/{len(measured)}] done {label}", file=sys.stderr)
            continue
        for command in arguments.drop_caches:
            status = subprocess.run(command, shell=True, check=False).returncode
            if status != 0:
                print(
                    f"drop-caches command {command!r} exited with status {status}",
                    file=sys.stderr,
                )
                return 1
        print(f"[{number}/{len(measured)}] run {label}", file=sys.stderr)
        status = run_worker(
            combination,
            roots,
            results,
            arguments.max_repeats,
            read_budget_s,
            min_reads,
        )
        if status != 0:
            print(
                f"{label}: worker exited with status {status}", file=sys.stderr
            )
            failures.append(combination)
    merged = complete_parts(
        results, arguments.max_repeats, read_budget_s, min_reads
    )
    merged_recordings = list(
        dict.fromkeys(combination.recording for combination in merged)
    )
    header = run_metadata(
        recordings=",".join(merged_recordings) or "none",
        local_root=roots.local if "local" in stores else "none",
        http_root=roots.http if "http" in stores else "none",
        s3_root=roots.s3 if "s3" in stores else "none",
        s3_endpoint_url=roots.s3_endpoint_url or "none",
        http_server=server_header(roots.http) if "http" in stores else "none",
        nginx_build=nginx_build(roots.http) if "http" in stores else "none",
        max_repeats=arguments.max_repeats or "none",
        **{READ_BUDGET_KEY: read_budget_s, MIN_READS_KEY: min_reads},
        drop_caches="; ".join(arguments.drop_caches) or "none",
        **{f"seed.{name}": recording(name).seed for name in merged_recordings},
    )
    reads, opens = merge(results, merged, header)
    print(
        f"merged {reads} reads and {opens} opens into {results}",
        file=sys.stderr,
    )
    if failures:
        print(f"{len(failures)} combinations failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
