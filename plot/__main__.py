"""Command line: write every figure, table, caption, and paragraph of numbers."""

import argparse
import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from plot.cost import PRICES_FILE, load_prices, write_cost
from plot.data import (
    NoDataError,
    Output,
    load_csv,
    recording_order,
)
from plot.figure1 import figure1
from plot.figure2 import figure2
from plot.figure3 import figure3
from plot.figure4 import figure4
from plot.figure5 import figure5
from plot.methods import write_methods
from plot.tables import table1, table2, table3, table4, write_table
from tszbench import schemas
from tszbench.schemas import (
    FORMAT_READS_FILE,
    RESAMPLING_REQUESTS_FILE,
    RESAMPLING_RUNS_FILE,
    STORAGE_FILE,
)

DEFAULT_RECORDING = "R3"
DEFAULT_FORMAT_RECORDING = "R3-1h"
CAPTIONS_FILE = "captions.txt"


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    """Return the parsed command line."""
    parser = argparse.ArgumentParser(
        prog="python -m plot",
        description="Write the figures and tables of a benchmark run.",
    )
    parser.add_argument(
        "--results", type=Path, required=True, help="directory of the CSVs"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="directory to write into"
    )
    parser.add_argument(
        "--recording",
        default=DEFAULT_RECORDING,
        help="recording of Figure 3, Table 3, and the cost paragraph",
    )
    parser.add_argument(
        "--format-recording",
        default=DEFAULT_FORMAT_RECORDING,
        help="recording of Figures 2 and 5 of the format benchmark",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=PRICES_FILE,
        help="JSON file of the prices of the cost paragraph",
    )
    return parser.parse_args(argv)


def _supplement(recordings: Sequence[str], main: str) -> list[str]:
    """Return the recordings other than main, in recording order."""
    return [name for name in recording_order(recordings) if name != main]


def main(argv: Sequence[str] | None = None) -> int:
    """Write every output that the CSVs in `--results` allow.

    An output whose CSV is absent, or whose CSV holds no rows for it, is
    skipped with a message on standard error. Figures 2, 3, and 5 are also
    written for each other recording present, as `fig2-<id>`, `fig3-<id>`, and
    `fig5-<id>`.

    Table 1 needs no CSV and is always written. `methods.txt` holds the Methods
    sentences that the upload, open, read budget, generation, check, and header
    data allow.

    Returns:
        0, or 1 when none of `format-reads.csv`, `resampling-runs.csv`, and
        `storage.csv` exists.
    """
    args = _parse(argv)
    results: Path = args.results
    out: Path = args.out
    recording: str = args.recording
    format_recording: str = args.format_recording
    outputs: list[Output] = []

    def run(label: str, job: Callable[[], Output]) -> None:
        try:
            outputs.append(job())
        except NoDataError as error:
            print(f"skipped {label}: {error}", file=sys.stderr)

    def csv(name: str, columns: Sequence[str]) -> Any | None:
        path = results / name
        if not path.exists():
            print(f"skipped outputs of {name}: file not found", file=sys.stderr)
            return None
        return load_csv(path, columns)

    format_benchmark = csv(FORMAT_READS_FILE, schemas.FORMAT_READS)
    runs = csv(RESAMPLING_RUNS_FILE, schemas.RESAMPLING_RUNS)
    storage = csv(STORAGE_FILE, schemas.STORAGE)
    requests_path = results / RESAMPLING_REQUESTS_FILE
    requests = (
        load_csv(requests_path, schemas.RESAMPLING_REQUESTS)
        if requests_path.exists()
        else None
    )
    if format_benchmark is not None:
        run("Figure 1", lambda: figure1(format_benchmark, out))
        run(
            "Figure 2",
            lambda: figure2(format_benchmark, out, format_recording),
        )
        for name in _supplement(
            format_benchmark["recording"], format_recording
        ):
            run(
                f"Figure 2 of {name}",
                partial(figure2, format_benchmark, out, name, f"fig2-{name}"),
            )
    if runs is not None:
        run("Figure 3", lambda: figure3(runs, out, recording))
        for name in _supplement(runs["recording"], recording):
            run(
                f"Figure 3 of {name}",
                partial(figure3, runs, out, name, f"fig3-{name}"),
            )
        run("Figure 4", lambda: figure4(runs, out))
    if format_benchmark is not None:
        run(
            "Figure 5",
            lambda: figure5(format_benchmark, out, format_recording),
        )
        for name in _supplement(
            format_benchmark["recording"], format_recording
        ):
            run(
                f"Figure 5 of {name}",
                partial(figure5, format_benchmark, out, name, f"fig5-{name}"),
            )
    run("Table 1", lambda: write_table(table1(), out))
    if storage is not None:
        run("Table 2", lambda: write_table(table2(storage), out))
    if runs is not None:
        run("Table 3", lambda: write_table(table3(runs, recording), out))
    if format_benchmark is not None:
        run("Table 4", lambda: write_table(table4(format_benchmark), out))
    if runs is not None:
        run(
            "cost paragraph",
            lambda: write_cost(
                runs, requests, load_prices(args.prices), out, recording
            ),
        )
    run(
        "Methods sentences",
        lambda: write_methods(
            results,
            out,
            lambda message: print(f"skipped {message}", file=sys.stderr),
        ),
    )
    captions = [o.caption for o in outputs if o.caption]
    if captions:
        (out / CAPTIONS_FILE).write_text("\n\n".join(captions) + "\n")
    return (
        0
        if any(x is not None for x in (format_benchmark, runs, storage))
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
