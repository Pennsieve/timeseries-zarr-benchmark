"""Figure 2: median seconds and bytes per read against window length.

Reads `format-reads.csv` only.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

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
from plot.style import (
    FORMAT_LABELS,
    FORMAT_LINES,
    READ_KIND_LABELS,
    STORE_LABELS,
    new_figure,
    save_figure,
)
from tszbench.schemas import READ_FORMATS, READ_KINDS, STORES

PANEL_WIDTH = 3.2
PANEL_HEIGHT = 2.6


def figure2_series(format_benchmark: Any, recording: str) -> Any:
    """Return the medians per format, store, read kind, and window.

    The columns are `store`, `read_kind`, `format`, `window_s`, `median_s`,
    `median_bytes`, and `reads`. `zarr-direct` and `overhead` rows are not in the result.

    Raises:
        NoDataError: The recording has no read of a plotted format.
    """
    rows = format_benchmark[
        (format_benchmark["recording"] == recording)
        & format_benchmark["format"].isin(READ_FORMATS)
    ]
    if rows.empty:
        raise NoDataError(f"no reads of {recording} in format_benchmark")
    return rows.groupby(
        ["store", "read_kind", "format", "window_s"], as_index=False
    ).agg(
        median_s=("seconds", "median"),
        median_bytes=("bytes", "median"),
        reads=("seconds", "size"),
    )


def overhead_floor(format_benchmark: Any, recording: str) -> dict[str, float]:
    """Return the median seconds of the 1-byte reads of each store.

    The overhead is the per-request floor of the store, so one value per store
    covers every window and read kind.
    """
    rows = overhead_rows(format_benchmark, recording)
    medians = rows.groupby("store")["seconds"].median()
    return {str(store): float(value) for store, value in medians.items()}


def _draw_formats(axis: Any, panel: Any, column: str) -> None:
    """Draw one line per format of a panel, from the column of `panel`."""
    for name in READ_FORMATS:
        line = panel[panel["format"] == name].sort_values("window_s")
        if line.empty:
            continue
        color, style, marker = FORMAT_LINES[name]
        axis.plot(
            line["window_s"],
            line[column],
            color=color,
            linestyle=style,
            marker=marker,
            markersize=3.5,
            linewidth=1.2,
        )


def figure2(
    format_benchmark: Any,
    out_dir: Path,
    recording: str = "R3-1h",
    stem: str = "fig2",
) -> Output:
    """Write Figure 2 for one recording as `<stem>.pdf` and `<stem>.png`.

    One panel per store and read kind, with a log x axis of window length. The
    upper rows have a log y axis of median seconds. Each format is one line.
    The overhead is a horizontal dotted line at the median of the 1-byte reads
    of the store. The lower rows, one per read kind, have a log y axis of
    median bytes that the store served for a read.

    Args:
        format_benchmark: Data frame of `format-reads.csv`.
        out_dir: Directory to write into.
        recording: Recording to plot.
        stem: File name without extension.

    Returns:
        The two files and the caption.

    Raises:
        NoDataError: See `figure2_series`.
    """
    series = figure2_series(format_benchmark, recording)
    floor = overhead_floor(format_benchmark, recording)
    stores = [name for name in STORES if name in set(series["store"])]
    kinds = [name for name in READ_KINDS if name in set(series["read_kind"])]
    windows = sorted(series["window_s"].unique())
    figure = new_figure(
        PANEL_WIDTH * len(stores) + 0.6, 2 * PANEL_HEIGHT * len(kinds) + 1.2
    )
    axes = figure.subplots(
        2 * len(kinds), len(stores), sharex=True, squeeze=False
    )
    for block, (column, label) in enumerate(
        (
            ("median_s", "median seconds per read"),
            ("median_bytes", "median bytes per read"),
        )
    ):
        reference = axes[block * len(kinds)][0]
        for row, kind in enumerate(kinds):
            for position, store in enumerate(stores):
                axis = axes[block * len(kinds) + row][position]
                if axis is not reference:
                    axis.sharey(reference)
                panel = series[
                    (series["store"] == store) & (series["read_kind"] == kind)
                ]
                _draw_formats(axis, panel, column)
                if column == "median_s" and store in floor:
                    color, style, _ = FORMAT_LINES[OVERHEAD]
                    axis.axhline(
                        floor[store],
                        color=color,
                        linestyle=style,
                        linewidth=1.2,
                    )
                axis.set_xscale("log")
                axis.set_yscale("log")
                axis.set_xticks(windows)
                axis.set_xticklabels([window_label(w) for w in windows])
                axis.minorticks_off()
                axis.grid(linewidth=0.3, alpha=0.5)
                axis.set_title(
                    f"{STORE_LABELS[store]}, {READ_KIND_LABELS[kind]}"
                )
                if column == "median_bytes":
                    axis.yaxis.set_major_formatter(
                        FuncFormatter(lambda value, _: format_bytes(value))
                    )
                if position == 0:
                    axis.set_ylabel(label)
    for axis in axes[-1]:
        axis.set_xlabel("window length")
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
    return Output(
        paths,
        _caption(
            _drawn_reads(format_benchmark, recording),
            format_benchmark.attrs.get("header"),
            series,
            floor,
            recording,
            stores,
        ),
    )


def _drawn_reads(format_benchmark: Any, recording: str) -> Any:
    """Return the reads of a recording panel, overhead reads included."""
    return pd.concat(
        [
            format_benchmark[
                (format_benchmark["recording"] == recording)
                & format_benchmark["format"].isin(READ_FORMATS)
            ],
            overhead_rows(format_benchmark, recording),
        ]
    )


def _caption(
    reads: Any,
    header: dict[str, str] | None,
    series: Any,
    floor: dict[str, float],
    recording: str,
    stores: list[str],
) -> str:
    """Return the caption text of Figure 2 with its read counts."""
    total = int(series["reads"].sum())
    per_store = ", ".join(
        count_noun(int(series[series["store"] == store]["reads"].sum()), "read")
        + f" on {STORE_LABELS[store]}"
        for store in stores
    )
    return (
        "Figure 2. Median seconds per read (upper panels) and median bytes "
        f"that the store served per read (lower panels) of {recording} against "
        "window length, on log axes, for each format. The overhead line is the "
        "median of the reads of a 1-byte object. "
        f"The medians cover {count_noun(total, 'read')}: {per_store}."
        + reads_sentence(reads, header)
    )
