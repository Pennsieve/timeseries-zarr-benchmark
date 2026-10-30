"""Figure 3: the resampling benchmark seconds per request and requests per second.

Reads `resampling-runs.csv` only.
"""

from pathlib import Path
from typing import Any

from matplotlib.lines import Line2D

from plot.data import (
    NoDataError,
    Output,
    count_noun,
    failed_request_sentence,
    link_limited,
    link_limited_sentence,
    select_runs,
    window_label,
    without_not_run,
)
from plot.style import (
    FAILED_MARKER,
    LINK_LIMITED_MARKER,
    METHOD_COLORS,
    METHOD_LABELS,
    new_figure,
    save_figure,
)
from tszbench.schemas import METHODS

PANEL_WIDTH = 2.6
PANEL_HEIGHT = 2.2


def figure3_series(runs: Any, recording: str) -> Any:
    """Return the runs of one recording that Figure 3 draws.

    One row per method, window, and user count, chosen by
    `plot.data.select_runs` and sorted by window, method, and users. A run
    that was not run is left out.

    Raises:
        NoDataError: The recording has no the resampling benchmark run.
    """
    chosen = without_not_run(select_runs(runs))
    chosen = chosen[chosen["recording"] == recording]
    if chosen.empty:
        raise NoDataError(f"no the resampling benchmark runs of {recording}")
    return chosen.sort_values(["window_s", "method", "users"]).reset_index(
        drop=True
    )


def _draw_method(axes: Any, rows: Any, method: str) -> None:
    """Draw the seconds band, the median line, and requests per second."""
    color = METHOD_COLORS[method]
    seconds, rate = axes
    seconds.fill_between(
        rows["users"],
        rows["p5_s"],
        rows["p95_s"],
        color=color,
        alpha=0.2,
        linewidth=0,
    )
    seconds.plot(rows["users"], rows["median_s"], color=color, linewidth=1.2)
    rate.plot(
        rows["users"], rows["requests_per_second"], color=color, linewidth=1.2
    )
    failed = rows["failure_rate"] > 0
    heights = (
        (seconds, rows["median_s"]),
        (rate, rows["requests_per_second"]),
    )
    for axis, height in heights:
        axis.plot(
            rows.loc[~failed, "users"],
            height[~failed],
            color=color,
            linestyle="none",
            marker="o",
            markersize=3.5,
        )
        axis.plot(
            rows.loc[failed, "users"],
            height[failed],
            color=color,
            linestyle="none",
            marker=FAILED_MARKER,
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.6,
        )
    limited = link_limited(rows)
    for axis, height in heights:
        axis.plot(
            rows.loc[limited, "users"],
            height[limited],
            linestyle="none",
            marker=LINK_LIMITED_MARKER,
            markersize=10,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.8,
        )


def figure3(
    runs: Any, out_dir: Path, recording: str = "R3", stem: str = "fig3"
) -> Output:
    """Write Figure 3 for one recording as `<stem>.pdf` and `<stem>.png`.

    One column per window. The top row has a log x axis of simultaneous users
    and a log y axis of seconds per request: the median of the run as a line
    and the 5th to 95th percentile as a band. The bottom row shows requests
    per second. The server and the precomputed read are one line each. A run
    with a failure rate above 0 carries a distinct marker at its median. A
    run that a load generator link may have limited has an open ring around
    its point.

    Args:
        runs: Data frame of `resampling-runs.csv`.
        out_dir: Directory to write into.
        recording: Recording to plot.
        stem: File name without extension.

    Returns:
        The two files and the caption.

    Raises:
        NoDataError: See `figure3_series`.
    """
    series = figure3_series(runs, recording)
    windows = sorted(series["window_s"].unique())
    users = sorted(series["users"].unique())
    figure = new_figure(
        PANEL_WIDTH * len(windows) + 0.8, 2 * PANEL_HEIGHT + 1.0
    )
    axes = figure.subplots(2, len(windows), sharex=True, squeeze=False)
    for column, window in enumerate(windows):
        panel = series[series["window_s"] == window]
        for method in METHODS:
            rows = panel[panel["method"] == method]
            if len(rows):
                _draw_method((axes[0][column], axes[1][column]), rows, method)
        axes[0][column].set_yscale("log")
        axes[0][column].set_title(window_label(window))
        axes[1][column].set_xlabel("simultaneous users")
        axes[1][column].set_ylim(bottom=0)
        for axis in axes[:, column]:
            axis.set_xscale("log")
            axis.set_xticks(users)
            axis.set_xticklabels([f"{int(u):,}" for u in users])
            axis.minorticks_off()
            axis.grid(linewidth=0.3, alpha=0.5)
    axes[0][0].set_ylabel("seconds per request")
    axes[1][0].set_ylabel("requests per second")
    handles = [
        Line2D([], [], color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        for m in METHODS
        if m in set(series["method"])
    ]
    handles.append(
        Line2D(
            [],
            [],
            color="white",
            marker=FAILED_MARKER,
            markersize=7,
            markeredgecolor="black",
            markerfacecolor="grey",
            label="run with failures",
        )
    )
    if link_limited(series).any():
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=LINK_LIMITED_MARKER,
                markersize=10,
                markerfacecolor="none",
                markeredgecolor="black",
                label="generator link limited",
            )
        )
    figure.legend(
        handles=handles, loc="outside lower center", ncols=3, frameon=False
    )
    paths = save_figure(figure, out_dir, stem)
    return Output(paths, _caption(runs, series, recording))


def _caption(runs: Any, series: Any, recording: str) -> str:
    """Return the caption text of Figure 3 with its request counts."""
    parts = ", ".join(
        count_noun(
            int(series[series["method"] == method]["requests"].sum()), "request"
        )
        + f" by the {METHOD_LABELS[method]} method"
        for method in METHODS
        if method in set(series["method"])
    )
    return (
        f"Figure 3. Seconds per request of {recording}, as the median and the "
        "5th to 95th percentile band of each run, and requests per second of "
        "each run, against simultaneous users. A cross marks a run with a "
        "failure rate above 0. Both methods read from S3. The runs hold "
        f"{parts}, warm-up requests left out."
        f"{failed_request_sentence(runs, percentiles=True)}"
        f"{link_limited_sentence(series)}"
    )
