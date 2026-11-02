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

WINDOW_S = SUBSET_WINDOWS_S[0]
"""Window of Figure 5, in seconds. Only this window holds subset reads."""


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

    The rows are the reads of a plotted format at the window of Figure 5, of
    every read kind.

    Raises:
        NoDataError: The recording has no subset read at that window.
    """
    rows = format_benchmark[
        (format_benchmark["recording"] == recording)
        & format_benchmark["format"].isin(READ_FORMATS)
        & (format_benchmark["window_s"] == WINDOW_S)
        & format_benchmark["read_kind"].isin(READ_KINDS)
    ]
    subset = [kind for kind in READ_KINDS if subset_size(kind) is not None]
    if not rows["read_kind"].isin(subset).any():
        raise NoDataError(
            f"no subset reads of {recording} at {window_label(WINDOW_S)} "
            "in format_benchmark"
        )
    return rows


def figure5_series(format_benchmark: Any, recording: str) -> Any:
    """Return the medians per format, store, and read kind.

    The columns are `store`, `format`, `read_kind`, `channels`, `median_s`,
    `median_bytes`, and `reads`. `channels` is the number of channels the read
    covers. `zarr-direct` and `overhead` rows are not in the result.

    Raises:
        NoDataError: The recording has no subset read at the window of
            Figure 5.
    """
    rows = figure5_rows(format_benchmark, recording)
    series = rows.groupby(["store", "format", "read_kind"], as_index=False).agg(
        median_s=("seconds", "median"),
        median_bytes=("bytes", "median"),
        reads=("seconds", "size"),
    )
    series["channels"] = series["read_kind"].map(
        lambda kind: channels_read(recording, str(kind))
    )
    return series.sort_values(["store", "format", "channels"]).reset_index(
        drop=True
    )


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

    One panel per store, with a log x axis of the channels read. The upper row
    has a log y axis of median seconds. Each format is one line. The overhead
    is a horizontal dotted line at the median of the 1-byte reads of the
    store. The lower row has a log y axis of median bytes that the store
    served for a read.

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
    counts = sorted(int(count) for count in series["channels"].unique())
    figure = new_figure(PANEL_WIDTH * len(stores) + 0.6, 2 * PANEL_HEIGHT + 1.2)
    axes = figure.subplots(2, len(stores), sharex=True, squeeze=False)
    for row, (column, label) in enumerate(
        (
            ("median_s", "median seconds per read"),
            ("median_bytes", "median bytes per read"),
        )
    ):
        for position, store in enumerate(stores):
            axis = axes[row][position]
            if axis is not axes[row][0]:
                axis.sharey(axes[row][0])
            _draw_formats(axis, series[series["store"] == store], column)
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
            axis.set_title(STORE_LABELS[store])
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
        ),
    )


def _caption(
    reads: Any,
    header: dict[str, str] | None,
    series: Any,
    recording: str,
    stores: list[str],
) -> str:
    """Return the caption text of Figure 5 with its read counts."""
    total = int(series["reads"].sum())
    per_store = ", ".join(
        count_noun(int(series[series["store"] == store]["reads"].sum()), "read")
        + f" on {STORE_LABELS[store]}"
        for store in stores
    )
    return (
        "Figure 5. Median seconds per read (upper panels) and median bytes "
        f"that the store served per read (lower panels) of {recording} at the "
        f"{window_label(WINDOW_S)} window against the number of channels "
        "read, on log axes, for each format. A read covers one channel, a "
        "contiguous block of channels, or every channel. The overhead line is "
        "the median of the reads of a 1-byte object. "
        f"The medians cover {count_noun(total, 'read')}: {per_store}."
        + reads_sentence(reads, header)
    )
