"""Figure 4: the resampling benchmark seconds per request against raw bytes under the window.

Reads `resampling-runs.csv` only. The figure draws the 1-user runs, one line
per method and recording, and a reference line for the time to move the raw
bytes over the link of a load generator.
"""

from pathlib import Path
from typing import Any

from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from plot.data import (
    NoDataError,
    Output,
    client_limited,
    client_limited_sentence,
    count_noun,
    failed_request_sentence,
    format_bytes,
    format_short,
    join_words,
    recording_order,
    select_runs,
    without_not_run,
)
from plot.style import (
    CLIENT_LIMITED_MARKER,
    FAILED_MARKER,
    LINE_WIDTH,
    METHOD_COLORS,
    METHOD_LABELS,
    RECORDING_MARKERS,
    new_figure,
    save_figure,
)
from resampling_benchmark.load.generator import LINK_BYTES_PER_S
from tszbench.minmax import COLUMNS
from tszbench.schemas import METHODS

USER_COUNTS = (1,)

FIGURE_WIDTH = 6.6
FIGURE_HEIGHT = 4.2

Y_MARGIN = 1.5
"""Factor between the extreme medians and the ends of the y axis."""

REFERENCE_COLOR = "#777777"
"""Color of the reference line and of the gray legend markers."""


def figure4_series(runs: Any) -> Any:
    """Return the runs at 1 user that Figure 4 draws.

    One row per method, recording, and window, chosen by
    `plot.data.select_runs` and sorted by method, recording, and raw bytes. A
    run that was not run is left out.

    Raises:
        NoDataError: No run has 1 user.
    """
    chosen = without_not_run(select_runs(runs))
    chosen = chosen[chosen["users"].isin(USER_COUNTS)]
    if chosen.empty:
        raise NoDataError("no resampling benchmark runs at 1 user")
    return chosen.sort_values(["method", "recording", "raw_bytes"]).reset_index(
        drop=True
    )


def _link_label() -> str:
    """Return the legend label of the reference line."""
    return f"raw bytes at {format_short(LINK_BYTES_PER_S * 8 / 1e9)} Gbps"


def figure4(runs: Any, out_dir: Path, stem: str = "fig4") -> Output:
    """Write Figure 4 as `<stem>.pdf` and `<stem>.png`.

    Each recording and method is one solid line through the windows of the
    recording in raw-byte order, at the raw bytes under the window on a log x
    axis and the median seconds per request on a log y axis. The line has the
    color of its method, and its marker shape names the recording. No line
    joins two recordings. A dotted gray line is the seconds to move the raw
    bytes over the link of a load generator. A run with a failure rate above 0
    carries a distinct marker. A run that a load generator link may have
    limited has an open ring around its marker.

    Args:
        runs: Data frame of `resampling-runs.csv`.
        out_dir: Directory to write into.
        stem: File name without extension.

    Returns:
        The two files and the caption.

    Raises:
        NoDataError: See `figure4_series`.
    """
    series = figure4_series(runs)
    recordings = recording_order(series["recording"])
    markers = {
        name: RECORDING_MARKERS[index % len(RECORDING_MARKERS)]
        for index, name in enumerate(recordings)
    }
    figure = new_figure(FIGURE_WIDTH, FIGURE_HEIGHT)
    axis = figure.subplots()
    for method in METHODS:
        color = METHOD_COLORS[method]
        for name in recordings:
            rows = series[
                (series["method"] == method) & (series["recording"] == name)
            ]
            if rows.empty:
                continue
            axis.plot(
                rows["raw_bytes"],
                rows["median_s"],
                color=color,
                linestyle="-",
                linewidth=LINE_WIDTH,
                marker=markers[name],
                markersize=4.5,
            )
        failed = series[
            (series["method"] == method) & (series["failure_rate"] > 0)
        ]
        axis.plot(
            failed["raw_bytes"],
            failed["median_s"],
            color=color,
            linestyle="none",
            marker=FAILED_MARKER,
            markersize=8,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.8,
        )
        limited = series[(series["method"] == method) & client_limited(series)]
        axis.plot(
            limited["raw_bytes"],
            limited["median_s"],
            linestyle="none",
            marker=CLIENT_LIMITED_MARKER,
            markersize=11,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.8,
        )
    low = float(series["raw_bytes"].min())
    high = float(series["raw_bytes"].max())
    axis.plot(
        [low, high],
        [low / LINK_BYTES_PER_S, high / LINK_BYTES_PER_S],
        color=REFERENCE_COLOR,
        linestyle=":",
        linewidth=LINE_WIDTH,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_ylim(
        float(series["median_s"].min()) / Y_MARGIN,
        float(series["median_s"].max()) * Y_MARGIN,
    )
    axis.xaxis.set_major_locator(LogLocator(base=100.0))
    axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.0,)))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: format_bytes(v)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: format_short(v)))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("raw float32 bytes under the window")
    axis.set_ylabel("median seconds per request")
    axis.grid(linewidth=0.3, alpha=0.5)
    handles = [
        Line2D(
            [],
            [],
            color=METHOD_COLORS[m],
            linewidth=LINE_WIDTH,
            label=METHOD_LABELS[m],
        )
        for m in METHODS
        if m in set(series["method"])
    ]
    handles += [
        Line2D(
            [],
            [],
            color=REFERENCE_COLOR,
            linestyle="none",
            marker=markers[name],
            label=name,
        )
        for name in recordings
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=REFERENCE_COLOR,
            linestyle=":",
            linewidth=LINE_WIDTH,
            label=_link_label(),
        )
    )
    handles.append(
        Line2D(
            [],
            [],
            color="grey",
            linestyle="none",
            marker=FAILED_MARKER,
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=8,
            label="run with failures",
        )
    )
    if client_limited(series).any():
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=CLIENT_LIMITED_MARKER,
                markersize=11,
                markerfacecolor="none",
                markeredgecolor="black",
                label="client limited",
            )
        )
    figure.legend(handles=handles, frameon=False, loc="outside right center")
    paths = save_figure(figure, out_dir, stem)
    return Output(paths, _caption(runs, series, recordings))


def _caption(runs: Any, series: Any, recordings: list[str]) -> str:
    """Return the caption text of Figure 4 with its request count."""
    requests = count_noun(int(series["requests"].sum()), "request")
    return (
        "Figure 4. Median seconds per request at 1 user against the raw "
        "float32 bytes of every channel under the window, for "
        + join_words(recordings)
        + ", on log axes. Each line is one recording and method, with one "
        "point per window. A request returns the minimum and maximum of "
        f"every channel over the window in {COLUMNS:,} pixel columns. "
        "The dotted line is the seconds to move the raw bytes at "
        f"{format_short(LINK_BYTES_PER_S * 8 / 1e9)} Gbps. A cross marks a "
        "run with a failure rate above 0. Both methods read from S3. The "
        f"runs hold {requests} at 1 user, warm-up requests left out."
        f"{failed_request_sentence(runs, percentiles=False)}"
        f"{client_limited_sentence(series)}"
    )
