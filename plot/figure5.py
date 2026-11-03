"""Figure 5: median seconds and bytes per read against the channels read.

Reads `format-reads.csv` only.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from format_benchmark.positions import SUBSET_WINDOWS_S
from plot.data import (
    OVERHEAD,
    NoDataError,
    Output,
    count_noun,
    format_bytes,
    overhead_rows,
    reads_sentence,
    window_label,
)
from plot.figure2 import PANEL_HEIGHT, PANEL_WIDTH, overhead_floor
from plot.style import (
    FORMAT_LABELS,
    FORMAT_LINES,
    STORE_LABELS,
    new_figure,
    save_figure,
)
from tszbench.recordings import resolve
from tszbench.schemas import READ_FORMATS, READ_KINDS, STORES, subset_size


def channels_read(recording: str, read_kind: str) -> int:
    """Return the number of channels a read of read_kind covers.

    Raises:
        ValueError: recording is not a recording of `tszbench.recordings`.
    """
    if read_kind == "all":
        return resolve(recording).channels
    return subset_size(read_kind) or 1


def figure5_rows(format_benchmark: Any, recording: str) -> Any:
    """Return the reads of a recording that Figure 5 draws.

    The rows are the reads of a plotted format, of every read kind, at each
    window of `SUBSET_WINDOWS_S` that holds a subset read of the recording.

    Raises:
        NoDataError: The recording has no subset read at any of those windows.
    """
    rows = format_benchmark[
        (format_benchmark["recording"] == recording)
        & format_benchmark["format"].isin(READ_FORMATS)
        & format_benchmark["window_s"].isin(SUBSET_WINDOWS_S)
        & format_benchmark["read_kind"].isin(READ_KINDS)
    ]
    subset = [kind for kind in READ_KINDS if subset_size(kind) is not None]
    windows = set(rows[rows["read_kind"].isin(subset)]["window_s"])
    if not windows:
        labels = ", ".join(window_label(window) for window in SUBSET_WINDOWS_S)
        raise NoDataError(
            f"no subset reads of {recording} at {labels} in format_benchmark"
        )
    return rows[rows["window_s"].isin(windows)]


def figure5_series(format_benchmark: Any, recording: str) -> Any:
    """Return the medians per format, store, and read kind.

    The columns are `window_s`, `store`, `format`, `read_kind`, `channels`,
    `median_s`, `median_bytes`, and `reads`. `channels` is the number of
    channels the read covers. `zarr-direct` and `overhead` rows are not in the
    result.

    Raises:
        NoDataError: See `figure5_rows`.
    """
    rows = figure5_rows(format_benchmark, recording)
    series = rows.groupby(
        ["window_s", "store", "format", "read_kind"], as_index=False
    ).agg(
        median_s=("seconds", "median"),
        median_bytes=("bytes", "median"),
        reads=("seconds", "size"),
    )
    series["channels"] = series["read_kind"].map(
        lambda kind: channels_read(recording, str(kind))
    )
    return series.sort_values(
        ["window_s", "store", "format", "channels"]
    ).reset_index(drop=True)


def _draw_formats(axis: Any, panel: Any, column: str) -> None:
    """Draw one line per format of a panel, from the column of `panel`."""
    for name in READ_FORMATS:
        line = panel[panel["format"] == name].sort_values("channels")
        if line.empty:
            continue
        color, style, marker = FORMAT_LINES[name]
        axis.plot(
            line["channels"],
            line[column],
            color=color,
            linestyle=style,
            marker=marker,
            markersize=3.5,
            linewidth=1.2,
        )


def figure5(
    format_benchmark: Any,
    out_dir: Path,
    recording: str = "R3-1h",
    stem: str = "fig5",
) -> Output:
    """Write Figure 5 for one recording as `<stem>.pdf` and `<stem>.png`.

    One column per store and two rows per window, with a log x axis of the
    channels read. The first row of a window has a log y axis of median
    seconds. Each format is one line. The overhead is a horizontal dotted line
    at the median of the 1-byte reads of the store. The second row of a window
    has a log y axis of median bytes that the store served for a read.

    Args:
        format_benchmark: Data frame of `format-reads.csv`.
        out_dir: Directory to write into.
        recording: Recording to plot.
        stem: File name without extension.

    Returns:
        The two files and the caption.

    Raises:
        NoDataError: See `figure5_series`.
    """
    series = figure5_series(format_benchmark, recording)
    floor = overhead_floor(format_benchmark, recording)
    stores = [name for name in STORES if name in set(series["store"])]
    windows = sorted(int(window) for window in series["window_s"].unique())
    counts = sorted(int(count) for count in series["channels"].unique())
    rows = [
        (window, column, label)
        for window in windows
        for column, label in (
            ("median_s", "median seconds per read"),
            ("median_bytes", "median bytes per read"),
        )
    ]
    figure = new_figure(
        PANEL_WIDTH * len(stores) + 0.6, len(rows) * PANEL_HEIGHT + 1.2
    )
    axes = figure.subplots(len(rows), len(stores), sharex=True, squeeze=False)
    for row, (window, column, label) in enumerate(rows):
        for position, store in enumerate(stores):
            axis = axes[row][position]
            if axis is not axes[row][0]:
                axis.sharey(axes[row][0])
            panel = series[
                (series["window_s"] == window) & (series["store"] == store)
            ]
            _draw_formats(axis, panel, column)
            if column == "median_s" and store in floor:
                color, style, _ = FORMAT_LINES[OVERHEAD]
                axis.axhline(
                    floor[store], color=color, linestyle=style, linewidth=1.2
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xticks(counts)
            axis.set_xticklabels([str(count) for count in counts])
            axis.minorticks_off()
            axis.grid(linewidth=0.3, alpha=0.5)
            axis.set_title(f"{STORE_LABELS[store]}, {window_label(window)}")
            if column == "median_bytes":
                axis.yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _: format_bytes(value))
                )
            if position == 0:
                axis.set_ylabel(label)
    for axis in axes[-1]:
        axis.set_xlabel("channels read")
    legend_names = [
        name for name in READ_FORMATS if name in set(series["format"])
    ] + ([OVERHEAD] if floor else [])
    handles = [
        Line2D(
            [],
            [],
            color=FORMAT_LINES[name][0],
            linestyle=FORMAT_LINES[name][1],
            marker=FORMAT_LINES[name][2],
            markersize=3.5,
            label=FORMAT_LABELS[name],
        )
        for name in legend_names
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncols=min(len(stores), 3),
        frameon=False,
    )
    paths = save_figure(figure, out_dir, stem)
    drawn = pd.concat(
        [
            figure5_rows(format_benchmark, recording),
            overhead_rows(format_benchmark, recording),
        ]
    )
    return Output(
        paths,
        _caption(
            drawn,
            format_benchmark.attrs.get("header"),
            series,
            recording,
            stores,
            windows,
        ),
    )


def _caption(
    reads: Any,
    header: dict[str, str] | None,
    series: Any,
    recording: str,
    stores: list[str],
    windows: list[int],
) -> str:
    """Return the caption text of Figure 5 with its read counts."""
    total = int(series["reads"].sum())
    per_store = ", ".join(
        count_noun(int(series[series["store"] == store]["reads"].sum()), "read")
        + f" on {STORE_LABELS[store]}"
        for store in stores
    )
    return (
        "Figure 5. Median seconds per read and median bytes that the store "
        f"served per read of {recording} at the {_join(windows)} "
        f"{'window' if len(windows) == 1 else 'windows'} against the number "
        "of channels read, on log axes, for each format. Each window has a "
        "row of seconds above a row of bytes. A read covers one channel, a "
        "contiguous block of channels, or every channel. The overhead line is "
        "the median of the reads of a 1-byte object. "
        f"The medians cover {count_noun(total, 'read')}: {per_store}."
        + reads_sentence(reads, header)
    )


def _join(windows: list[int]) -> str:
    """Return the window labels joined as `10 s` or `10 s and 10 min`."""
    labels = [window_label(window) for window in windows]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]
