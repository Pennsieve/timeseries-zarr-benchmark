"""Drawing and captions shared by Figures 1, 2, and 5 of the format benchmark.

Each figure has one row per store. A row holds one channel panel per window of
`format_benchmark.positions.SUBSET_WINDOWS_S` at which the recording has subset
reads, then one window panel for the one-channel reads and one for the
all-channel reads. A line is the median of one format.
"""

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from format_benchmark.positions import SUBSET_WINDOWS_S
from plot.data import (
    OVERHEAD,
    NoDataError,
    count_noun,
    format_bytes,
    format_short,
    join_words,
    overhead_rows,
    reads_sentence,
    window_label,
)
from plot.style import (
    FORMAT_LABELS,
    FORMAT_LINES,
    LABEL_OFFSET_POINTS,
    LINE_WIDTH,
    STORE_LABELS,
    TSZARR_LINE_WIDTH,
    label_lines,
    new_figure,
)
from tszbench.recordings import resolve
from tszbench.schemas import (
    READ_FORMATS,
    READ_KINDS,
    STORES,
    SUBSET_KINDS,
    subset_size,
)

PANEL_WIDTH = 2.3
"""Width of one panel, in inches."""

ROW_HEIGHT = 2.4
"""Height of one row of panels, in inches."""

LABEL_ROOM = 2.3
"""Width left of the first panel and right of the last panel, in inches."""

AXIS_ROOM = 0.8
"""Height of the x tick labels and the x label, in inches."""

NARROW_DECADES = 2.5
"""A y axis that spans fewer decades than this gets ticks at 1, 2, and 5."""

SAMPLE_POINTS = 18.0
"""Length of the line sample left of a direct label, in points."""

SAMPLE_GAP_POINTS = 4.0
"""Distance from a line sample to its label, in points."""

SAMPLE_GID = "line-sample"
"""Group id of the line samples, which tells them from the data lines."""

MARKER_SIZE = 3.5
BAND_ALPHA = 0.15
ONE_BYTE_LABEL = "1-byte read"
SECONDS_LABEL = "median seconds per read"
BYTES_LABEL = "median bytes per read"

WINDOW_KINDS = ("one", "all")
"""Read kinds that have a window panel."""

PLOT_ORDER = (*(name for name in READ_FORMATS if name != "tszarr"), "tszarr")
"""Drawing order of the format lines. TS-Zarr is last and draws on top."""


@dataclass(frozen=True)
class Panel:
    """One column of panels.

    Attributes:
        title: Title of the panel in the first row.
        x_label: Label of the x axis of the panel in the last row.
        x_column: Column of the series on the x axis, `channels` or `window_s`.
        ticks: Positions of the x ticks.
        read_kind: Read kind that a window panel draws, or None.
        window_s: Window in seconds that a channel panel draws, or None.
    """

    title: str
    x_label: str
    x_column: str
    ticks: tuple[float, ...]
    read_kind: str | None = None
    window_s: float | None = None

    def select(self, series: Any, store: str) -> Any:
        """Return the rows of a store that the panel draws."""
        rows = series[series["store"] == store]
        if self.window_s is not None:
            return rows[rows["window_s"] == self.window_s]
        return rows[rows["read_kind"] == self.read_kind]

    def tick_label(self, value: float) -> str:
        """Return the label of an x tick."""
        if self.x_column == "channels":
            return str(int(round(value)))
        return window_label(value)


@dataclass(frozen=True)
class Drawing:
    """A drawn figure with the data behind it.

    Attributes:
        figure: The figure.
        series: The medians that the figure draws, as from `format_series`.
        reads: The reads that the medians cover. Seconds figures add the
            1-byte reads.
        stores: Stores of the rows of the figure, in row order.
        panels: Panels of the figure, in column order.
    """

    figure: Any
    series: Any
    reads: Any
    stores: tuple[str, ...]
    panels: tuple[Panel, ...]


def channels_read(recording: str, read_kind: str) -> int:
    """Return the number of channels a read of read_kind covers.

    Raises:
        ValueError: recording is not a recording of `tszbench.recordings`.
    """
    if read_kind == "one":
        return 1
    if read_kind == "all":
        return resolve(recording).channels
    return subset_size(read_kind) or 1


def _drawn_rows(
    format_benchmark: Any, recording: str, stores: tuple[str, ...]
) -> Any:
    """Return the reads of a plotted format that the figures of a recording draw.

    A subset read counts only at a window of `SUBSET_WINDOWS_S`.
    """
    subset = format_benchmark["read_kind"].isin(SUBSET_KINDS)
    return format_benchmark[
        (format_benchmark["recording"] == recording)
        & format_benchmark["store"].isin(stores)
        & format_benchmark["format"].isin(READ_FORMATS)
        & format_benchmark["read_kind"].isin(READ_KINDS)
        & (~subset | format_benchmark["window_s"].isin(SUBSET_WINDOWS_S))
    ]


def format_series(
    format_benchmark: Any, recording: str, stores: tuple[str, ...] = STORES
) -> Any:
    """Return the medians and quartiles per store, read kind, format, and window.

    The columns are `store`, `read_kind`, `format`, `window_s`, `channels`,
    `median_s`, `p25_s`, `p75_s`, `median_bytes`, and `reads`. `channels` is
    the number of channels the read kind covers. `zarr-direct` and `overhead`
    rows are not in the result.

    Args:
        format_benchmark: Data frame of `format-reads.csv`.
        recording: Recording to select.
        stores: Stores to select.

    Raises:
        NoDataError: The recording has no read of a plotted format on stores.
    """
    rows = _drawn_rows(format_benchmark, recording, stores)
    if rows.empty:
        where = join_words([STORE_LABELS[store] for store in stores])
        raise NoDataError(
            f"no reads of {recording} on {where} in format_benchmark"
        )
    series = rows.groupby(
        ["store", "read_kind", "format", "window_s"], as_index=False
    ).agg(
        median_s=("seconds", "median"),
        p25_s=("seconds", lambda values: values.quantile(0.25)),
        p75_s=("seconds", lambda values: values.quantile(0.75)),
        median_bytes=("bytes", "median"),
        reads=("seconds", "size"),
    )
    series["channels"] = series["read_kind"].map(
        lambda kind: channels_read(recording, str(kind))
    )
    series["store"] = pd.Categorical(
        series["store"], categories=STORES, ordered=True
    )
    series["read_kind"] = pd.Categorical(
        series["read_kind"], categories=READ_KINDS, ordered=True
    )
    series = series.sort_values(
        ["store", "read_kind", "format", "window_s"]
    ).reset_index(drop=True)
    series["store"] = series["store"].astype(str)
    series["read_kind"] = series["read_kind"].astype(str)
    return series[
        [
            "store",
            "read_kind",
            "format",
            "window_s",
            "channels",
            "median_s",
            "p25_s",
            "p75_s",
            "median_bytes",
            "reads",
        ]
    ]


def overhead_floor(format_benchmark: Any, recording: str) -> dict[str, float]:
    """Return the median seconds of the 1-byte reads of each store.

    The overhead is the per-request floor of the store. One value per store
    covers every window and read kind.
    """
    rows = overhead_rows(format_benchmark, recording)
    medians = rows.groupby("store")["seconds"].median()
    return {str(store): float(value) for store, value in medians.items()}


def panels_of(series: Any) -> tuple[Panel, ...]:
    """Return the panels of a series, in column order.

    A channel panel exists for each window of `SUBSET_WINDOWS_S` that has a
    subset read. A window panel exists for each of `WINDOW_KINDS` that has a
    read.
    """
    panels: list[Panel] = []
    subset = series[series["read_kind"].isin(SUBSET_KINDS)]
    for window in SUBSET_WINDOWS_S:
        if window not in set(subset["window_s"]):
            continue
        counts = sorted(
            int(count)
            for count in series[series["window_s"] == window][
                "channels"
            ].unique()
        )
        panels.append(
            Panel(
                title=f"{window_label(window)} window",
                x_label="channels read",
                x_column="channels",
                ticks=tuple(float(count) for count in counts),
                window_s=window,
            )
        )
    titles = {"one": "one channel", "all": "all channels"}
    for kind in WINDOW_KINDS:
        rows = series[series["read_kind"] == kind]
        if rows.empty:
            continue
        windows = sorted(float(window) for window in rows["window_s"].unique())
        panels.append(
            Panel(
                title=titles[kind],
                x_label="window length",
                x_column="window_s",
                ticks=tuple(windows),
                read_kind=kind,
            )
        )
    return tuple(panels)


def _draw_formats(
    axis: Any, rows: Any, panel: Panel, value: str, seconds: bool
) -> list[tuple[Line2D, str]]:
    """Draw one line per format and return each line with its label."""
    lines = []
    for name in PLOT_ORDER:
        line = rows[rows["format"] == name].sort_values(panel.x_column)
        if line.empty:
            continue
        color, style, marker = FORMAT_LINES[name]
        on_top = name == "tszarr"
        if seconds:
            axis.fill_between(
                line[panel.x_column],
                line["p25_s"],
                line["p75_s"],
                color=color,
                alpha=BAND_ALPHA,
                linewidth=0,
                zorder=3 if on_top else 2,
            )
        (drawn,) = axis.plot(
            line[panel.x_column],
            line[value],
            color=color,
            linestyle=style,
            marker=marker,
            markersize=MARKER_SIZE,
            linewidth=TSZARR_LINE_WIDTH if on_top else LINE_WIDTH,
            zorder=5 if on_top else 4,
        )
        lines.append((drawn, FORMAT_LABELS[name]))
    return lines


def _style_axis(axis: Any, panel: Panel, seconds: bool) -> None:
    """Set the scales, ticks, and grid of one panel."""
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(panel.ticks)
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: panel.tick_label(value))
    )
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.minorticks_off()
    axis.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: (
                format_short(value) if seconds else format_bytes(value)
            )
        )
    )
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.grid(linewidth=0.3, alpha=0.5)


def draw(
    format_benchmark: Any,
    recording: str,
    stores: tuple[str, ...],
    *,
    seconds: bool,
) -> Drawing:
    """Draw the panels of a recording for the stores, one row per store.

    A seconds figure draws the median seconds per read with a band from the
    25th to the 75th percentile, and one dotted line at the median seconds of
    the 1-byte reads of each store. A bytes figure draws the median bytes per
    read. Every row shares its y axis and every column shares its x axis. The
    rightmost panel of a row labels each line.

    Args:
        format_benchmark: Data frame of `format-reads.csv`.
        recording: Recording to draw.
        stores: Stores to draw, in row order of `tszbench.schemas.STORES`.
        seconds: Whether to draw seconds. Otherwise the figure draws bytes.

    Raises:
        NoDataError: See `format_series`.
    """
    series = format_series(format_benchmark, recording, stores)
    drawn_stores = tuple(
        store for store in STORES if store in set(series["store"])
    )
    panels = panels_of(series)
    floor = overhead_floor(format_benchmark, recording) if seconds else {}
    value = "median_s" if seconds else "median_bytes"
    figure = new_figure(
        PANEL_WIDTH * len(panels) + LABEL_ROOM,
        ROW_HEIGHT * len(drawn_stores) + AXIS_ROOM,
    )
    axes = figure.subplots(
        len(drawn_stores),
        len(panels),
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    labeled: list[list[tuple[Line2D, str]]] = []
    for row, store in enumerate(drawn_stores):
        for column, panel in enumerate(panels):
            axis = axes[row][column]
            lines = _draw_formats(
                axis, panel.select(series, store), panel, value, seconds
            )
            if seconds and store in floor:
                color, style, _ = FORMAT_LINES[OVERHEAD]
                one_byte = axis.axhline(
                    floor[store],
                    color=color,
                    linestyle=style,
                    linewidth=LINE_WIDTH,
                    zorder=1,
                )
                lines.append((one_byte, ONE_BYTE_LABEL))
            _style_axis(axis, panel, seconds)
            if row == 0:
                axis.set_title(panel.title)
            if row == len(drawn_stores) - 1:
                axis.set_xlabel(panel.x_label)
            if column == len(panels) - 1:
                labeled.append(lines)
        axes[row][0].set_ylabel(_y_label(store, drawn_stores, seconds))
    for row, lines in enumerate(labeled):
        low, high = axes[row][0].get_ylim()
        if math.log10(high / low) < NARROW_DECADES:
            axes[row][0].yaxis.set_major_locator(
                LogLocator(base=10, subs=(1.0, 2.0, 5.0))
            )
        _label_with_samples(axes[row][-1], lines)
    reads = _drawn_rows(format_benchmark, recording, drawn_stores)
    if seconds:
        floor_reads = overhead_rows(format_benchmark, recording)
        reads = pd.concat(
            [reads, floor_reads[floor_reads["store"].isin(drawn_stores)]]
        )
    return Drawing(figure, series, reads, drawn_stores, panels)


def _label_with_samples(axis: Any, lines: list[tuple[Line2D, str]]) -> None:
    """Label the lines of an axis, each with a sample of its line.

    The sample has the color, line style, marker, and width of the line. It
    sits in the right margin of the axis, between the axis and the label, at
    the height of the label.
    """
    notes = label_lines(
        axis, [line for line, _ in lines], [text for _, text in lines]
    )
    shift = SAMPLE_POINTS + SAMPLE_GAP_POINTS
    for note in notes:
        x_points, y_points = note.xyann
        note.xyann = (x_points + shift, y_points)
    figure = axis.get_figure()
    figure.draw_without_rendering()
    points = figure.dpi / 72 / axis.get_window_extent().width
    start = LABEL_OFFSET_POINTS * points
    stop = (LABEL_OFFSET_POINTS + SAMPLE_POINTS) * points
    for (line, _), note in zip(lines, notes, strict=True):
        height = note.xy[1]
        sample = Line2D(
            [1 + start, 1 + (start + stop) / 2, 1 + stop],
            [height] * 3,
            transform=axis.transAxes,
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            markevery=[1],
            clip_on=False,
            gid=SAMPLE_GID,
        )
        axis.add_line(sample)


def _y_label(store: str, stores: tuple[str, ...], seconds: bool) -> str:
    """Return the y label of the first panel of a row."""
    text = SECONDS_LABEL if seconds else BYTES_LABEL
    if len(stores) == 1:
        return text
    return f"{STORE_LABELS[store]}\n{text}"


def _panel_sentence(panels: tuple[Panel, ...]) -> str:
    """Return the caption sentences that say what each panel varies."""
    windows = [
        window_label(panel.window_s)
        for panel in panels
        if panel.window_s is not None
    ]
    parts = []
    if len(windows) == 1:
        parts.append(
            f"The channel panel shows the {windows[0]} window and varies the "
            "channels read."
        )
    elif windows:
        parts.append(
            "The channel panels show the window lengths "
            f"{join_words(windows)} and vary the channels read."
        )
    if any(panel.read_kind is not None for panel in panels):
        parts.append(
            "The window panels show one channel and all channels and vary "
            "the window length."
        )
    return " ".join(parts)


def _read_counts(drawing: Drawing) -> str:
    """Return the sentence that counts the reads that the medians cover."""
    parts = [
        count_noun(
            int(
                drawing.series[drawing.series["store"] == store]["reads"].sum()
            ),
            "read",
        )
        + f" on {STORE_LABELS[store]}"
        for store in drawing.stores
    ]
    return f" The medians cover {join_words(parts)}."


def seconds_caption(
    number: int,
    drawing: Drawing,
    recording: str,
    header: dict[str, str] | None,
) -> str:
    """Return the caption of a seconds figure with its read counts."""
    stores = join_words([STORE_LABELS[store] for store in drawing.stores])
    return (
        f"Figure {number}. Median seconds per read of {recording} on {stores}, "
        f"on log axes. {_panel_sentence(drawing.panels)} "
        "A band spans the 25th to 75th percentile of the reads. "
        "The dotted line is the median seconds of a 1-byte read on the same "
        "store." + _read_counts(drawing) + reads_sentence(drawing.reads, header)
    )


def bytes_caption(
    number: int,
    drawing: Drawing,
    recording: str,
    header: dict[str, str] | None,
) -> str:
    """Return the caption of a bytes figure with its read counts."""
    store = STORE_LABELS[drawing.stores[0]]
    return (
        f"Figure {number}. Median bytes that {store} served per read of "
        f"{recording}, on log axes. The bytes do not depend on the store. "
        f"{_panel_sentence(drawing.panels)}"
        + _read_counts(drawing)
        + reads_sentence(drawing.reads, header)
    )
