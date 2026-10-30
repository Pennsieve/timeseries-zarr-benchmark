"""The Methods sentences that quote measured numbers: `methods.txt`.

Each paragraph reads its own CSVs from the results directory: the uploads, the
timed opens, the read budget, the generation times, the equality check, and
the CSV headers. A paragraph whose CSV is absent is skipped and reported.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from format_benchmark.positions import READ_BUDGET_KEY
from plot.data import (
    NoDataError,
    Output,
    as_bool,
    budget_label,
    count_noun,
    format_bytes,
    format_short,
    header_number,
    join_words,
    load_csv,
    recording_order,
    spell_count,
)
from plot.style import FORMAT_LABELS, STORE_LABELS
from tszbench import schemas
from tszbench.csvmeta import read_header

METHODS_FILE = "methods.txt"
GENERATION_RECORDINGS = ("R3-1h", "R3", "R4")
"""Recordings whose generation seconds the Methods quote, in this order."""

VERSIONED_LIBRARIES = ("timeseries-zarr", "tsdownsample")
"""Libraries whose version the Methods quote from the CSV headers."""

ABSENT_VERSION = "absent"
"""Value that a CSV header gives to a library that is not installed."""

Skip = Callable[[str], None]


def _label(name: str) -> str:
    """Return the display name of a format."""
    return (
        "zarr-direct"
        if name == "zarr-direct"
        else FORMAT_LABELS.get(name, name)
    )


def _ordered(names: Any, order: tuple[str, ...]) -> list[str]:
    """Return the distinct names in the order of `order`, then the rest sorted."""
    distinct = set(names)
    first = [name for name in order if name in distinct]
    return first + sorted(distinct - set(first))


def _frames(results: Path, pattern: str, columns: tuple[str, ...]) -> Any:
    """Return the rows of every CSV that matches pattern, or raise.

    Raises:
        NoDataError: No file matches, or the files hold no rows.
    """
    paths = sorted(results.glob(pattern))
    frames = [load_csv(path, columns) for path in paths]
    frame = pd.concat(frames, ignore_index=True) if frames else None
    if frame is None or frame.empty:
        raise NoDataError(f"no rows in {pattern}")
    return frame


def upload_paragraph(results: Path) -> str:
    """Return the median upload rate of each format.

    Raises:
        NoDataError: No `upload-*.csv` has a row.
    """
    frame = _frames(results, schemas.upload_file("*"), schemas.UPLOAD)
    medians = frame.groupby("format")["bytes_per_second"].median().dropna()
    if medians.empty:
        raise NoDataError("no upload rate in upload-*.csv")
    rates = [
        f"{format_bytes(medians[name])}/s for {_label(name)}"
        for name in _ordered(medians.index, schemas.FORMATS)
    ]
    return (
        "The median upload rate of `aws s3 cp` at default settings is "
        f"{join_words(rates)}."
    )


def open_paragraph(results: Path) -> str:
    """Return the median seconds and requests of the open of each format.

    Raises:
        NoDataError: `format-opens.csv` is absent or has no row.
    """
    path = results / schemas.FORMAT_OPENS_FILE
    if not path.exists():
        raise NoDataError(f"{schemas.FORMAT_OPENS_FILE} not found")
    frame = load_csv(path, schemas.FORMAT_OPENS)
    # The overhead row reads a 1-byte object and opens no file.
    frame = frame[frame["format"] != "overhead"]
    if frame.empty:
        raise NoDataError(f"no rows in {schemas.FORMAT_OPENS_FILE}")
    medians = frame.groupby(["format", "store"])[
        ["seconds", "requests"]
    ].median()
    sentences = []
    for name in _ordered(frame["format"], schemas.FORMAT_ROWS):
        stores = [s for s in schemas.STORES if (name, s) in medians.index]
        parts = [
            f"{format_short(medians.loc[(name, s), 'seconds'])} s and "
            f"{format_short(medians.loc[(name, s), 'requests'])} requests on "
            f"{STORE_LABELS[s]}"
            for s in stores
        ]
        if parts:
            sentences.append(
                f"The open of {_label(name)} takes a median of "
                f"{join_words(parts)}."
            )
    if not sentences:
        raise NoDataError(f"no known store in {schemas.FORMAT_OPENS_FILE}")
    return " ".join(sentences)


def budget_paragraph(results: Path) -> str:
    """Return how many combinations stopped at the read budget.

    A combination stopped at the read budget when `reads` is below
    `planned_reads`. The budget comes from the header of `format-opens.csv`
    and is left out when the header has no `READ_BUDGET_KEY`.

    Raises:
        NoDataError: `format-opens.csv` is absent or has no row with reads.
    """
    path = results / schemas.FORMAT_OPENS_FILE
    if not path.exists():
        raise NoDataError(f"{schemas.FORMAT_OPENS_FILE} not found")
    frame = load_csv(path, schemas.FORMAT_OPENS).dropna(
        subset=["reads", "planned_reads"]
    )
    if frame.empty:
        raise NoDataError(f"no reads in {schemas.FORMAT_OPENS_FILE}")
    budget = header_number(frame.attrs["header"], READ_BUDGET_KEY)
    named = "" if budget is None else f" of {budget_label(budget)}"
    stopped = frame[frame["reads"] < frame["planned_reads"]]
    if stopped.empty:
        return f"No combination reached the read budget{named}."
    fewest = int(stopped["reads"].min())
    return (
        f"Of {count_noun(len(frame), 'combination')}, {spell_count(len(stopped))} "
        f"stopped at the read budget{named}, "
        f"{'each after at least ' if len(stopped) > 1 else 'after '}"
        f"{count_noun(fewest, 'read')}."
    )


def generation_paragraph(results: Path) -> str:
    """Return the seconds that `generate` took for each format of a recording.

    Raises:
        NoDataError: None of `GENERATION_RECORDINGS` has a generation CSV row.
    """
    sentences = []
    for recording in GENERATION_RECORDINGS:
        path = results / schemas.generation_file(recording)
        if not path.exists():
            continue
        frame = load_csv(path, schemas.GENERATION)
        seconds = frame.groupby("format")["seconds"].median()
        parts = [
            f"{format_short(seconds[name])} s for {_label(name)}"
            for name in _ordered(seconds.index, schemas.FORMATS)
        ]
        if parts:
            sentences.append(
                f"Generating {recording} takes {join_words(parts)}."
            )
    if not sentences:
        raise NoDataError(
            "no rows in "
            + join_words(
                [schemas.generation_file(r) for r in GENERATION_RECORDINGS]
            )
        )
    return " ".join(sentences)


def check_paragraph(results: Path) -> str:
    """Return how many windows gave equal arrays under the two methods.

    Raises:
        NoDataError: `resampling-check.csv` is absent or has no row.
    """
    path = results / schemas.CHECK_FILE
    if not path.exists():
        raise NoDataError(f"{schemas.CHECK_FILE} not found")
    frame = load_csv(path, schemas.CHECK)
    if frame.empty:
        raise NoDataError(f"no rows in {schemas.CHECK_FILE}")
    equal = int(as_bool(frame["equal"]).sum())
    names = join_words(recording_order(frame["recording"]))
    return (
        f"The server and the precomputed read return equal arrays in "
        f"{spell_count(equal)} of {spell_count(len(frame))} windows, on {names}."
    )


def _header_paths(results: Path) -> list[Path]:
    """Return the result CSVs that exist, in a fixed order."""
    fixed = (
        schemas.FORMAT_READS_FILE,
        schemas.FORMAT_OPENS_FILE,
        schemas.RESAMPLING_RUNS_FILE,
        schemas.RESAMPLING_REQUESTS_FILE,
        schemas.CHECK_FILE,
        schemas.STORAGE_FILE,
    )
    paths = [results / name for name in fixed if (results / name).exists()]
    for pattern in (schemas.generation_file("*"), schemas.upload_file("*")):
        paths += sorted(results.glob(pattern))
    return paths


def versions_paragraph(results: Path) -> str:
    """Return the library versions and instance types of the CSV headers.

    A version that the files disagree on lists every value. The instance types
    group the files that ran on each.

    Raises:
        NoDataError: No result CSV exists, or no header names a version or an
            instance type.
    """
    headers = {path.name: read_header(path) for path in _header_paths(results)}
    sentences = []
    versions = []
    for library in VERSIONED_LIBRARIES:
        values = list(
            dict.fromkeys(
                h[f"version.{library}"]
                for h in headers.values()
                if h.get(f"version.{library}", ABSENT_VERSION) != ABSENT_VERSION
            )
        )
        if values:
            shown = values[0] if len(values) == 1 else f"({', '.join(values)})"
            versions.append(f"{library} {shown}")
    if versions:
        sentences.append(f"The CSV headers record {join_words(versions)}.")
    by_instance: dict[str, list[str]] = {}
    for name, header in headers.items():
        if "instance" in header:
            by_instance.setdefault(header["instance"], []).append(name)
    if by_instance:
        parts = [
            f"{instance} for {join_words(names)}"
            for instance, names in by_instance.items()
        ]
        sentences.append(f"The instance types are {join_words(parts)}.")
    if not sentences:
        raise NoDataError(
            "no library version or instance type in any CSV header"
        )
    return " ".join(sentences)


PARAGRAPHS: tuple[tuple[str, Callable[[Path], str]], ...] = (
    ("upload sentence", upload_paragraph),
    ("open sentence", open_paragraph),
    ("read budget sentence", budget_paragraph),
    ("generation sentence", generation_paragraph),
    ("equality check sentence", check_paragraph),
    ("versions sentence", versions_paragraph),
)
"""Name and writer of each paragraph of `methods.txt`, in file order."""


def write_methods(results: Path, out_dir: Path, skip: Skip) -> Output:
    """Write `methods.txt`, one paragraph per measured Methods statement.

    Args:
        results: Directory of the result CSVs.
        out_dir: Directory to write into.
        skip: Called with a message for each paragraph that has no data.

    Returns:
        The file, with an empty caption.

    Raises:
        NoDataError: Every paragraph lacks data.
    """
    paragraphs = []
    for name, build in PARAGRAPHS:
        try:
            paragraphs.append(build(results))
        except NoDataError as error:
            skip(f"Methods {name}: {error}")
    if not paragraphs:
        raise NoDataError("no result CSV holds data for a Methods sentence")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / METHODS_FILE
    path.write_text("\n\n".join(paragraphs) + "\n")
    return Output((path,), "")
