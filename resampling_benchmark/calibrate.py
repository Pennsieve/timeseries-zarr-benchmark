"""Write the calibration CSVs: `python -m resampling_benchmark.calibrate`.

Reads the run directories of one or more orchestrator output roots. The
`server` kind writes `tszbench.schemas.CALIBRATION_SERVER_FILE` from the server
method runs. The `generator` kind writes
`tszbench.schemas.CALIBRATION_GENERATOR_FILE` from the precomputed method runs.
Each row is the `summarize.summarize_run` row of one run.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from resampling_benchmark.summarize import (
    RunFiles,
    deployment_metadata,
    find_runs,
    summarize_run,
)
from tszbench.csvmeta import CsvWriter, run_metadata
from tszbench.recordings import RecordingSpec, resolve
from tszbench.schemas import (
    CALIBRATION_GENERATOR,
    CALIBRATION_GENERATOR_FILE,
    CALIBRATION_SERVER,
    CALIBRATION_SERVER_FILE,
    RESAMPLING_RAW_DIRECTORY,
)

KINDS = {
    "server": ("server", CALIBRATION_SERVER, CALIBRATION_SERVER_FILE),
    "generator": (
        "precomputed",
        CALIBRATION_GENERATOR,
        CALIBRATION_GENERATOR_FILE,
    ),
}
"""For each kind, the method of its runs, its columns, and its result file."""


def _task_threads(files: RunFiles) -> str:
    """Return `task_threads` from the run's server metadata, or an empty string."""
    if files.server_metadata is None:
        return ""
    loaded = json.loads(files.server_metadata.read_text())
    return str(loaded.get("task_threads", ""))


def calibration_rows(
    roots: Sequence[Path],
    kind: str,
    lookup: Callable[[str], RecordingSpec] = resolve,
) -> tuple[list[RunFiles], list[dict[str, object]]]:
    """Return the runs of a kind and their calibration rows.

    The runs come in the order of roots, then in the order of their names. A
    run that was skipped or has another method is left out.

    Args:
        roots: Directories with one subdirectory per run.
        kind: A key of `KINDS`.
        lookup: Returns the spec of a recording id.

    Returns:
        The run files and one row of the kind's columns per run.
    """
    method, columns, _ = KINDS[kind]
    runs: list[RunFiles] = []
    rows: list[dict[str, object]] = []
    for root in roots:
        for files in find_runs(root):
            if files.not_run:
                continue
            summary = summarize_run(files, lookup)
            if summary["method"] != method:
                continue
            row = {column: summary.get(column, "") for column in columns}
            if kind == "server":
                row["task_threads"] = _task_threads(files)
            runs.append(files)
            rows.append(row)
    return runs, rows


def calibrate(
    roots: Sequence[Path],
    results: Path,
    kind: str,
    lookup: Callable[[str], RecordingSpec] = resolve,
) -> int:
    """Write the calibration CSV of a kind to the results directory.

    Returns:
        The number of rows written.
    """
    _, columns, name = KINDS[kind]
    runs, rows = calibration_rows(roots, kind, lookup)
    header = run_metadata() | deployment_metadata(runs)
    with CsvWriter(results / name, columns, header) as writer:
        writer.write_all(rows)
    return len(rows)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m resampling_benchmark.calibrate",
        description="Write the calibration CSV of a sweep of resampling runs.",
    )
    parser.add_argument(
        "kind",
        choices=sorted(KINDS),
        help=(
            f"server writes {CALIBRATION_SERVER_FILE}, "
            f"generator writes {CALIBRATION_GENERATOR_FILE}"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[Path("results") / RESAMPLING_RAW_DIRECTORY],
        help=(
            "orchestrator output roots, one per sweep value or all in one "
            f"(default results/{RESAMPLING_RAW_DIRECTORY})"
        ),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="directory that receives the CSV (default results)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Write the calibration CSV and print its row count."""
    arguments = parse_arguments(argv)
    count = calibrate(arguments.input, arguments.results, arguments.kind)
    print(f"{count} {arguments.kind} calibration rows")


if __name__ == "__main__":
    main()
