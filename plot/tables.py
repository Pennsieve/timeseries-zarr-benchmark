"""Tables 1 to 4 as data frames, CSV files, and LaTeX tabular files."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from plot.data import (
    NoDataError,
    Output,
    client_limited,
    count_noun,
    format_bytes,
    format_short,
    format_sig,
    not_run,
    reads_sentence,
    recording_order,
    select_runs,
    window_label,
)
from plot.style import FORMAT_LABELS, READ_KIND_LABELS, STORE_LABELS
from tszbench.recordings import RECORDINGS, resolve
from tszbench.schemas import FORMATS

NOT_RUN = "not run"
"""Cell of Table 3 where the server run or the precomputed run was not run."""

FAIL = "fail"
"""Cell of Table 3 where the server run had failures."""

FAIL_PRECOMPUTED = "fail (precomputed)"
"""Cell of Table 3 where only the precomputed run had failures."""

CLIENT_MARK = " (client)"
"""Suffix of a cell of Table 3 where the load generators may have limited a
run."""

RAW_PLUS_MINMAX = "raw+minmax"
"""Member row that sums `raw` and `minmax`."""

PAPER_MULTIPLIERS = {
    RAW_PLUS_MINMAX: 5 / 3,
    "mean": 1 / 3,
    "var": 1 / 3,
    "count": 1 / 3,
}
"""Multiples of the raw float32 bytes that the paper states for TS-Zarr."""

MEMBER_ORDER = (
    "",
    "raw",
    "minmax",
    RAW_PLUS_MINMAX,
    "mean",
    "var",
    "count",
    "metadata",
)

KILOHERTZ = 10_000
"""Rates from this many hertz up print in kilohertz."""

LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


@dataclass(frozen=True)
class Table:
    """One table in both of its forms.

    Attributes:
        name: File stem, such as `table1`.
        frame: Values for the CSV file.
        headers: Column titles of the LaTeX table.
        rows: Cells of the LaTeX table, as text.
        alignment: One `l` or `r` per column.
        caption: Paragraph for captions.txt.
    """

    name: str
    frame: Any
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    alignment: str
    caption: str


def _escape(text: str) -> str:
    """Return text with the LaTeX special characters escaped."""
    return "".join(LATEX_ESCAPES.get(char, char) for char in text)


def latex(table: Table) -> str:
    """Return a booktabs `tabular` for a table."""
    lines = [
        rf"\begin{{tabular}}{{{table.alignment}}}",
        r"\toprule",
        " & ".join(_escape(h) for h in table.headers) + r" \\",
        r"\midrule",
        *(" & ".join(_escape(c) for c in row) + r" \\" for row in table.rows),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def write_table(table: Table, out_dir: Path) -> Output:
    """Write `<name>.csv` and `<name>.tex` under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{table.name}.csv"
    tex_path = out_dir / f"{table.name}.tex"
    table.frame.to_csv(csv_path, index=False)
    tex_path.write_text(latex(table))
    return Output((csv_path, tex_path), table.caption)


def _rate(rate_hz: int) -> str:
    """Return a sample rate as `512 Hz` or `30 kHz`."""
    if rate_hz >= KILOHERTZ:
        return f"{format_short(rate_hz / 1000)} kHz"
    return f"{rate_hz} Hz"


def _duration(seconds: int) -> str:
    """Return a duration as `24 h` or `90 s`."""
    return window_label(seconds)


def table1() -> Table:
    """Return Table 1, the recordings of the benchmark.

    Columns: id, modality, channels, rate, duration, samples per channel, raw
    float32 bytes, and levels.
    """
    specs = list(RECORDINGS.values())
    frame = pd.DataFrame(
        {
            "id": [s.id for s in specs],
            "modality": [s.modality for s in specs],
            "channels": [s.channels for s in specs],
            "rate_hz": [s.rate_hz for s in specs],
            "duration_s": [s.duration_s for s in specs],
            "samples_per_channel": [s.samples_per_channel for s in specs],
            "raw_bytes": [s.raw_bytes for s in specs],
            "levels": [s.levels for s in specs],
        }
    )
    rows = tuple(
        (
            s.id,
            s.modality,
            str(s.channels),
            _rate(s.rate_hz),
            _duration(s.duration_s),
            f"{s.samples_per_channel:,}",
            format_bytes(s.raw_bytes),
            str(s.levels),
        )
        for s in specs
    )
    return Table(
        name="table1",
        frame=frame,
        headers=(
            "id",
            "modality",
            "channels",
            "rate",
            "duration",
            "samples per channel",
            "raw float32",
            "levels",
        ),
        rows=rows,
        alignment="llrrrrrr",
        caption=(
            "Table 1. The synthetic recordings. Raw float32 is the size of "
            "every sample as float32 in decimal units."
        ),
    )


def _member_sort(frame: Any) -> Any:
    """Return frame sorted by recording, format, and member."""
    formats = [*FORMATS, "zarr-direct", "overhead"]
    recordings = recording_order(frame["recording"])

    def rank(values: list[str], order: Any) -> list[int]:
        return [order.index(v) if v in order else len(order) for v in values]

    frame = frame.assign(
        r=rank(list(frame["recording"]), recordings),
        f=rank(list(frame["format"]), formats),
        m=rank(list(frame["member"]), list(MEMBER_ORDER)),
    )
    return frame.sort_values(["r", "f", "m"]).drop(columns=["r", "f", "m"])


def _with_raw_plus_minmax(storage: Any) -> Any:
    """Return storage with one `raw+minmax` row after each pair of members."""
    extra = []
    for (recording, name), group in storage.groupby(["recording", "format"]):
        members = group.set_index("member")
        if {"raw", "minmax"} <= set(members.index):
            extra.append(
                {
                    "recording": recording,
                    "format": name,
                    "member": RAW_PLUS_MINMAX,
                    "bytes": members.loc[["raw", "minmax"], "bytes"].sum(),
                    "objects": members.loc[["raw", "minmax"], "objects"].sum(),
                }
            )
    if not extra:
        return storage
    return pd.concat([storage, pd.DataFrame(extra)], ignore_index=True)


MIN_MULTIPLIER = 0.001
"""Multipliers below this print as `<0.001`."""


def _multiplier(value: float) -> str:
    """Return a multiplier with three significant digits."""
    if value < MIN_MULTIPLIER:
        return f"<{MIN_MULTIPLIER}"
    return format_sig(value)


def table2(storage: Any) -> Table:
    """Return Table 2, the storage per recording and format.

    One row per recording, format, and TS-Zarr member of `storage.csv`, and one
    `raw+minmax` row per bundle that holds both members. The multiplier is
    bytes divided by the raw float32 bytes of the recording. `paper` holds the
    multiplier of the paper where it states one: 5/3 for `raw+minmax` and 1/3
    for each of `mean`, `var`, and `count`. A repeated row of the same
    recording, format, and member keeps its last copy.

    Args:
        storage: Data frame of `storage.csv`.

    Raises:
        NoDataError: `storage` has no rows.
        ValueError: A recording is not one that `tszbench.recordings` knows.
    """
    if storage.empty:
        raise NoDataError("storage.csv has no rows")
    unique = storage.drop_duplicates(
        ["recording", "format", "member"], keep="last"
    )
    full = _member_sort(_with_raw_plus_minmax(unique)).reset_index(drop=True)
    raw = {name: resolve(name).raw_bytes for name in set(full["recording"])}
    full["multiplier"] = full["bytes"] / full["recording"].map(raw)
    full["paper"] = full["member"].map(PAPER_MULTIPLIERS)
    frame = full[
        [
            "recording",
            "format",
            "member",
            "bytes",
            "objects",
            "multiplier",
            "paper",
        ]
    ]
    rows = tuple(
        (
            str(r.recording),
            FORMAT_LABELS.get(str(r.format), str(r.format)),
            str(r.member),
            format_bytes(r.bytes),
            f"{int(r.objects):,}",
            _multiplier(r.multiplier),
            "" if math.isnan(r.paper) else format_sig(r.paper),
        )
        for r in frame.itertuples()
    )
    return Table(
        name="table2",
        frame=frame,
        headers=(
            "recording",
            "format",
            "member",
            "bytes at rest",
            "objects",
            "multiplier",
            "paper",
        ),
        rows=rows,
        alignment="lllrrrr",
        caption=(
            "Table 2. Bytes at rest, objects, and multiplier per recording "
            "and format, from "
            f"{count_noun(len(unique), 'row')} of the storage table. The "
            "multiplier is bytes divided by the raw float32 bytes of the "
            "recording. The paper column gives the multiplier of the paper "
            "for the raw and minmax members together and for each further "
            "member. The raw-only row is a TS-Zarr bundle with no level."
        ),
    )


def table3(runs: Any, recording: str = "R3") -> Table:
    """Return Table 3, server seconds divided by precomputed seconds.

    One row per window and one column per user count. A cell is the median
    seconds per request of the server run divided by that of the precomputed
    run. A cell reads `not run` when either run was not run. This takes
    precedence over a failure in the other run. Otherwise a cell reads `fail`
    when the server run has a failure rate above 0, and `fail (precomputed)`
    when only the precomputed run does. A cell is empty when a method has no
    run at that window and user count. A cell of the LaTeX table ends with
    `CLIENT_MARK` when either run is client limited, and the frame holds a
    boolean column `users_<count>_client` for the same fact.

    Args:
        runs: Data frame of `resampling-runs.csv`.
        recording: Recording to tabulate.

    Raises:
        NoDataError: The recording has no the resampling benchmark run.
    """
    chosen = select_runs(runs)
    chosen = chosen[chosen["recording"] == recording]
    if chosen.empty:
        raise NoDataError(f"no the resampling benchmark runs of {recording}")
    windows = sorted(chosen["window_s"].unique())
    users = sorted(chosen["users"].unique())
    used = 0
    values: list[list[Any]] = []
    marks: list[list[bool]] = []
    for window in windows:
        row: list[Any] = []
        row_marks: list[bool] = []
        for count in users:
            cell = chosen[
                (chosen["window_s"] == window) & (chosen["users"] == count)
            ]
            server = cell[cell["method"] == "server"]
            precomputed = cell[cell["method"] == "precomputed"]
            if server.empty or precomputed.empty:
                row.append(float("nan"))
                row_marks.append(False)
                continue
            if bool(not_run(server).iloc[0]) or bool(
                not_run(precomputed).iloc[0]
            ):
                row.append(NOT_RUN)
                row_marks.append(False)
                continue
            row_marks.append(
                bool(client_limited(server).iloc[0])
                or bool(client_limited(precomputed).iloc[0])
            )
            used += int(
                server["requests"].iloc[0] + precomputed["requests"].iloc[0]
            )
            if server["failure_rate"].iloc[0] > 0:
                row.append(FAIL)
            elif precomputed["failure_rate"].iloc[0] > 0:
                row.append(FAIL_PRECOMPUTED)
            else:
                row.append(
                    float(
                        server["median_s"].iloc[0]
                        / precomputed["median_s"].iloc[0]
                    )
                )
        values.append(row)
        marks.append(row_marks)
    names = [f"users_{int(count)}" for count in users]
    frame = pd.DataFrame(values, columns=names, dtype=object)
    frame.insert(0, "window_s", windows)
    for position, name in enumerate(names):
        frame[f"{name}_client"] = [row[position] for row in marks]
    rows = tuple(
        (
            window_label(window),
            *(
                (
                    cell
                    if isinstance(cell, str)
                    else ""
                    if math.isnan(cell)
                    else format_sig(cell, 2)
                )
                + (CLIENT_MARK if marked else "")
                for cell, marked in zip(
                    values[index], marks[index], strict=True
                )
            ),
        )
        for index, window in enumerate(windows)
    )
    return Table(
        name="table3",
        frame=frame,
        headers=("window", *(f"{int(count):,} users" for count in users)),
        rows=rows,
        alignment="l" + "r" * len(users),
        caption=(
            f"Table 3. Median seconds per request of the server divided by "
            f"that of the precomputed read, on {recording}. Both methods read "
            "from S3. The medians cover "
            f"{count_noun(used, 'request')}, warm-up requests left out. A "
            f"cell reads {FAIL} where the server run had failures, and "
            f"{FAIL_PRECOMPUTED} where only the precomputed run had failures."
            + (
                f" A cell reads {NOT_RUN} where a run with fewer users had a "
                "median request that failed."
                if any(NOT_RUN in row for row in values)
                else ""
            )
            + (
                f" A cell ends with{CLIENT_MARK} where the sustained CPU "
                "use or receive rate of a load generator passed half its "
                "capacity in a run and no repeat with fewer users per "
                "generator replaced the run."
                if any(any(row) for row in marks)
                else ""
            )
        ),
    )


def table4(format_benchmark: Any) -> Table:
    """Return Table 4, TS-Zarr seconds against zarr-python seconds.

    One row per recording, store, read kind, and window that holds reads of
    both `tszarr` and `zarr-direct`. `zarr-direct` reads the raw arrays of the
    same bundle with zarr-python and no reader. The columns are
    `recording`, `store`, `read_kind`, `window_s`, `tszarr_s`,
    `zarr_direct_s`, and `ratio`. The seconds are medians. The ratio is
    `tszarr_s` divided by `zarr_direct_s`, the time that the reader adds.

    Args:
        format_benchmark: Data frame of `format-reads.csv`.

    Raises:
        NoDataError: No combination has reads of both formats.
    """
    key = ["recording", "store", "read_kind", "window_s"]
    medians = {}
    counts = 0
    for name in ("tszarr", "zarr-direct"):
        rows = format_benchmark[format_benchmark["format"] == name]
        counts += len(rows)
        medians[name] = rows.groupby(key)["seconds"].median()
    frame = pd.concat(
        {
            "tszarr_s": medians["tszarr"],
            "zarr_direct_s": medians["zarr-direct"],
        },
        axis=1,
        join="inner",
    ).reset_index()
    if frame.empty:
        raise NoDataError(
            "no reads of both tszarr and zarr-direct in format_benchmark"
        )
    frame["ratio"] = frame["tszarr_s"] / frame["zarr_direct_s"]
    recordings = recording_order(frame["recording"])
    stores = list(STORE_LABELS)
    kinds = list(READ_KIND_LABELS)
    frame = frame.assign(
        r=frame["recording"].map(recordings.index),
        s=frame["store"].map(
            lambda v: stores.index(v) if v in stores else len(stores)
        ),
        k=frame["read_kind"].map(
            lambda v: kinds.index(v) if v in kinds else len(kinds)
        ),
    )
    frame = frame.sort_values(["r", "s", "k", "window_s"])
    frame = frame.drop(columns=["r", "s", "k"]).reset_index(drop=True)
    rows = tuple(
        (
            str(r.recording),
            STORE_LABELS.get(str(r.store), str(r.store)),
            READ_KIND_LABELS.get(str(r.read_kind), str(r.read_kind)),
            window_label(r.window_s),
            format_sig(r.tszarr_s),
            format_sig(r.zarr_direct_s),
            format_sig(r.ratio),
        )
        for r in frame.itertuples()
    )
    drawn = format_benchmark[
        format_benchmark["format"].isin(("tszarr", "zarr-direct"))
    ].merge(frame[key], on=key)
    used = len(drawn)
    return Table(
        name="table4",
        frame=frame,
        headers=(
            "recording",
            "store",
            "read",
            "window",
            "TS-Zarr (s)",
            "zarr-direct (s)",
            "ratio",
        ),
        rows=rows,
        alignment="llllrrr",
        caption=(
            "Table 4. Median seconds per read of the TS-Zarr reader and of "
            "zarr-python on the raw arrays of the same bundle, and the ratio "
            "of the two, which is the time that the reader adds. The medians "
            f"cover {count_noun(used, 'read')}."
            + reads_sentence(drawn, format_benchmark.attrs.get("header"))
        ),
    )
