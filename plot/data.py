"""Loading and selecting the rows of the result CSVs.

The helpers here hold no plotting code. Figures and tables share them so that
every output counts the same rows.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from format_benchmark.positions import MIN_READS_KEY, READ_BUDGET_KEY
from resampling_benchmark.load.generator import REQUEST_TIMEOUT_S
from resampling_benchmark.summarize import FAILED_REQUEST_SECONDS_KEY
from tszbench.csvmeta import read_header
from tszbench.recordings import RECORDINGS
from tszbench.schemas import READ_FORMATS

OVERHEAD = "overhead"
"""Value of `format` for the reads of the 1-byte object."""

PLOTTED_FORMATS = (*READ_FORMATS, OVERHEAD)
"""Rows of Figures 1 and 2. `zarr-direct` is left out."""

TEXT_COLUMNS = frozenset(
    {
        "run",
        "method",
        "recording",
        "store",
        "format",
        "read_kind",
        "member",
        "status",
        "generator",
        "user",
        "repeated",
        "link_limited",
        "not_run",
        "warmup",
        "equal",
    }
)
"""Columns kept as text. Every other column is numeric."""

SMALL_NUMBER_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3_600

COMBINATION = ["recording", "store", "format", "read_kind", "window_s"]
"""Columns of `format-reads.csv` that name one combination of the format benchmark."""


class NoDataError(ValueError):
    """Raised when the rows an output needs are absent from its CSV."""


@dataclass(frozen=True)
class Output:
    """Files an output wrote and the caption that describes them.

    Attributes:
        paths: Files written.
        caption: One paragraph for captions.txt, or an empty string when the
            output has no caption.
    """

    paths: tuple[Path, ...]
    caption: str


def load_csv(path: Path, columns: Sequence[str]) -> Any:
    """Read a result CSV into a data frame.

    The `# key: value` header lines are skipped. Columns named in
    `TEXT_COLUMNS` stay text, with an empty field read as an empty string.
    Every other column is numeric, and an empty field reads as NaN.

    Args:
        path: CSV written by `tszbench.csvmeta.CsvWriter`.
        columns: Column names the file must hold, from `tszbench.schemas`.

    Returns:
        A data frame with the columns of the file in file order. Its `attrs`
        key `header` holds the `# key: value` lines as a dictionary.

    Raises:
        ValueError: A column of `columns` is absent from the file.
    """
    frame = pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for name in frame.columns:
        if name not in TEXT_COLUMNS:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame.attrs["header"] = read_header(path)
    return frame


def header_number(header: dict[str, str], key: str) -> float | None:
    """Return a finite number from a CSV header, or None when it has none."""
    try:
        value = float(header[key])
    except (KeyError, ValueError):
        return None
    return value if math.isfinite(value) else None


def budget_label(seconds: float) -> str:
    """Return a read budget as `1 h`, `30 min`, or `90 s`.

    A budget that is not a whole number of seconds keeps its decimals.
    """
    if seconds == int(seconds):
        return window_label(seconds)
    return f"{format_short(seconds)} s"


def reads_sentence(rows: Any, header: dict[str, str] | None = None) -> str:
    """Return the caption sentences on the reads per combination.

    The first sentence gives the fewest and the most rows that any combination
    holds in `rows`. When the header has `READ_BUDGET_KEY`, a second sentence
    gives the read budget, and `MIN_READS_KEY` adds the reads that precede it.
    The text starts with a space.

    Args:
        rows: The rows of `format-reads.csv` that the output shows.
        header: Header of `format-reads.csv`. The `header` key of `rows.attrs` is
            the default.
    """
    counts = rows.groupby(COMBINATION).size()
    fewest, most = int(counts.min()), int(counts.max())
    if fewest == most:
        text = f" Each combination holds {count_noun(most, 'read')}."
    else:
        text = (
            f" Each combination holds between {spell_count(fewest)} and "
            f"{count_noun(most, 'read')}."
        )
    if header is None:
        header = rows.attrs.get("header", {})
    budget = header_number(header, READ_BUDGET_KEY)
    if budget is None:
        return text
    text += (
        " A combination stops after the read that brings the sum of its read "
        f"seconds to at least the read budget of {budget_label(budget)}"
    )
    minimum = header_number(header, MIN_READS_KEY)
    if minimum is not None:
        text += (
            f", once it has made at least {count_noun(int(minimum), 'read')}"
        )
    return text + "."


def failed_request_sentence(runs: Any, *, percentiles: bool) -> str:
    """Return the caption sentence on the seconds of a failed request.

    The minimum seconds come from `FAILED_REQUEST_SECONDS_KEY` in the header of
    `resampling-runs.csv`, or from `REQUEST_TIMEOUT_S` when the header lacks it.
    A failed request that lasted longer counts as its own seconds. The
    sentence starts with a space.

    Args:
        runs: Data frame of `resampling-runs.csv`.
        percentiles: Whether the output also shows percentiles.
    """
    seconds = header_number(
        runs.attrs.get("header", {}), FAILED_REQUEST_SECONDS_KEY
    )
    if seconds is None:
        seconds = REQUEST_TIMEOUT_S
    subject = "medians and percentiles" if percentiles else "medians"
    return (
        f" The {subject} count each failed request as at least "
        f"{format_short(seconds)} s."
    )


def as_bool(series: Any) -> Any:
    """Return a boolean series from text that reads `true` in any case."""
    text = series.astype(str).str.strip().str.lower()
    return text.isin(("true", "1", "yes"))


def recording_order(ids: Iterable[str]) -> list[str]:
    """Return the distinct ids with R1 to R4 first, then the rest sorted."""
    known = list(RECORDINGS)
    distinct = set(ids)
    first = [name for name in known if name in distinct]
    return first + sorted(distinct - set(first))


def select_runs(runs: Any) -> Any:
    """Return one run per method, recording, window, and user count.

    A run repeated with half the users per generator replaces the run it
    repeats, because the first run met a load generator limit. When several
    rows remain, the last one in file order wins.
    """
    frame = runs.copy()
    frame["repeated"] = as_bool(frame["repeated"])
    frame["order"] = range(len(frame))
    key = ["method", "recording", "window_s", "users"]
    frame = frame.sort_values(["repeated", "order"])
    frame = frame.drop_duplicates(key, keep="last")
    frame = frame.sort_values(key).drop(columns="order")
    return frame.reset_index(drop=True)


def link_limited(rows: Any) -> Any:
    """Return a boolean series that is true for the runs of `link_limited`.

    A run is link limited when a load generator received more than half of its
    link and no repeat with fewer users per generator replaced the run.
    """
    return as_bool(rows["link_limited"])


def not_run(rows: Any) -> Any:
    """Return a boolean series that is true for the runs of `not_run`.

    A run is not run when the orchestrator skipped it because a run of the same
    method, recording, and window with fewer users had a median request that
    failed. Such a row has no statistics.
    """
    return as_bool(rows["not_run"])


def without_not_run(rows: Any) -> Any:
    """Return the rows of runs that were run, with a fresh index."""
    return rows[~not_run(rows)].reset_index(drop=True)


def link_limited_sentence(rows: Any) -> str:
    """Return the caption sentence for the link limited runs of rows.

    The sentence starts with a space. It is empty when no run is link limited.
    """
    count = int(link_limited(rows).sum())
    if not count:
        return ""
    return (
        f" A ring marks {count_noun(count, 'run')} in which a load generator "
        "received more than half of its link and no repeat with fewer users "
        "per generator replaced the run."
    )


def overhead_rows(format_benchmark: Any, recording: str) -> Any:
    """Return the reads of the 1-byte object that belong to a recording panel.

    Rows that name the recording win. When none do, every overhead row
    applies, because the 1-byte object does not depend on the recording.
    """
    rows = format_benchmark[format_benchmark["format"] == OVERHEAD]
    own = rows[rows["recording"] == recording]
    return own if len(own) else rows


def window_label(seconds: float) -> str:
    """Return a window length as `10 s`, `1 min`, or `24 h`."""
    value = int(round(seconds))
    if value >= SECONDS_PER_HOUR and value % SECONDS_PER_HOUR == 0:
        return f"{value // SECONDS_PER_HOUR} h"
    if value >= SECONDS_PER_MINUTE and value % SECONDS_PER_MINUTE == 0:
        return f"{value // SECONDS_PER_MINUTE} min"
    return f"{value} s"


def format_sig(value: float, digits: int = 3) -> str:
    """Return a number with `digits` significant digits and no exponent.

    Numbers of 1000 and above keep thousands separators and all integer
    digits.
    """
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    magnitude = math.floor(math.log10(abs(value)))
    decimals = max(0, digits - 1 - magnitude)
    return f"{round(value, decimals):,.{decimals}f}"


def format_short(value: float, digits: int = 3) -> str:
    """Return `format_sig` of a number without trailing zeros after the point."""
    text = format_sig(value, digits)
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_exact(value: float) -> str:
    """Return a number as written, in plain notation with no exponent."""
    return format(Decimal(str(value)), "f")


def join_words(words: Sequence[str]) -> str:
    """Return words as `a`, `a and b`, or `a, b, and c`."""
    if len(words) <= 2:
        return " and ".join(words)
    return ", ".join(words[:-1]) + f", and {words[-1]}"


def format_bytes(count: float) -> str:
    """Return a byte count in decimal units, such as `88.5 MB`."""
    units = ("B", "kB", "MB", "GB", "TB")
    value = float(count)
    unit = 0
    while abs(value) >= 1000 and unit < len(units) - 1:
        value /= 1000
        unit += 1
    return f"{format_short(value)} {units[unit]}"


def spell_count(count: int) -> str:
    """Return a count as a word below 10 and as grouped digits from 10."""
    if 0 <= count < len(SMALL_NUMBER_WORDS):
        return SMALL_NUMBER_WORDS[count]
    return f"{count:,}"


def count_noun(count: int, singular: str, plural: str | None = None) -> str:
    """Return `three reads` or `one read` for a count and a noun."""
    noun = singular if count == 1 else plural or f"{singular}s"
    return f"{spell_count(count)} {noun}"
