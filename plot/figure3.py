"""Figure 3: the resampling benchmark seconds per request and requests per second.

One column per window and two rows, both on log axes against simultaneous
users. The upper row shows seconds per request with the timeout. The lower row
shows requests per second with a dotted reference line per method: the rate
if every request took the median seconds of the 1-user run. Reads
`resampling-runs.csv` only.
"""

from pathlib import Path
from typing import Any

from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from plot.data import (
    NoDataError,
    Output,
    client_limited,
    client_limited_sentence,
    count_noun,
    failed_request_seconds,
    failed_request_sentence,
    format_short,
    s3_rate_sentence,
    select_runs,
    window_label,
    without_not_run,
)
from plot.style import (
    CLIENT_LIMITED_MARKER,
    FAILED_MARKER,
    METHOD_COLORS,
    METHOD_LABELS,
    new_figure,
    save_figure,
)
from tszbench.minmax import COLUMNS
from tszbench.schemas import METHODS

PANEL_WIDTH = 2.6
PANEL_HEIGHT = 2.2
TIMEOUT_HEADROOM = 2.0
"""Factor above the timeout line that the upper axes reach."""
GUIDE_COLOR = "#777777"
"""Gray of the timeout line, the no-queueing legend entry, and the timeout text."""


def figure3_series(runs: Any, recording: str) -> Any:
    """Return the runs of one recording that Figure 3 draws.

    One row per method, window, and user count, chosen by
    `plot.data.select_runs` and sorted by window, method, and users. A run
    that was not run is left out.

    Raises:
        NoDataError: The recording has no resampling benchmark run.
    """
    chosen = without_not_run(select_runs(runs))
    chosen = chosen[chosen["recording"] == recording]
    if chosen.empty:
        raise NoDataError(f"no resampling benchmark runs of {recording}")
    return chosen.sort_values(["window_s", "method", "users"]).reset_index(
        drop=True
    )


def _draw_method(axes: Any, rows: Any, method: str) -> None:
    """Draw one method in one window.

    The upper axis gets the percentile band and the median line. The lower axis
    gets the requests per second and, when the method has a 1-user run, a
    dotted reference line in the method color. Failed and client-limited runs
    get their markers on both axes.
    """
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
    single = rows[rows["users"] == 1]
    if len(single):
        rate.plot(
            rows["users"],
            rows["users"] / single["median_s"].iloc[0],
            color=color,
            alpha=0.5,
            linestyle=":",
            linewidth=1.2,
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
    limited = client_limited(rows)
    for axis, height in heights:
        axis.plot(
            rows.loc[limited, "users"],
            height[limited],
            linestyle="none",
            marker=CLIENT_LIMITED_MARKER,
            markersize=10,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.8,
        )


def figure3(
    runs: Any, out_dir: Path, recording: str = "R3", stem: str = "fig3"
) -> Output:
    """Write Figure 3 for one recording as `<stem>.pdf` and `<stem>.png`.

    One column per window, with a log x axis of simultaneous users and plain
    number tick labels on every log axis. The top row has a log y axis of
    seconds per request: the median of the run as a line and the 5th to 95th
    percentile as a band, and a dotted gray line at the timeout. The bottom row
    has a log y axis of requests per second, with a dotted reference line per
    method at the users divided by the median seconds of its 1-user run. The
    server and the precomputed read are one line each. A run with a failure
    rate above 0 carries a distinct marker at its median. A client-limited run
    has an open ring around its point.

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
    timeout = failed_request_seconds(runs)
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
        axes[0][column].axhline(
            timeout, color=GUIDE_COLOR, linestyle=":", linewidth=1.0
        )
        if column == 0:
            axes[0][column].text(
                0.03,
                timeout,
                "timeout",
                transform=axes[0][column].get_yaxis_transform(),
                color=GUIDE_COLOR,
                fontsize=7,
                ha="left",
                va="bottom",
            )
        axes[0][column].set_title(window_label(window))
        axes[1][column].set_xlabel("simultaneous users")
        for axis in axes[:, column]:
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: format_short(value))
            )
            axis.set_xticks(users)
            axis.set_xticklabels([f"{int(u):,}" for u in users])
            axis.minorticks_off()
            axis.grid(linewidth=0.3, alpha=0.5)
        low, high = axes[0][column].get_ylim()
        axes[0][column].set_ylim(low, max(high, TIMEOUT_HEADROOM * timeout))
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
            color=GUIDE_COLOR,
            linestyle=":",
            linewidth=1.2,
            label="no queueing",
        )
    )
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
    if client_limited(series).any():
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=CLIENT_LIMITED_MARKER,
                markersize=10,
                markerfacecolor="none",
                markeredgecolor="black",
                label="client limited",
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
        f"Figure 3. A request asks for the minimum and maximum of every "
        f"channel of {recording} over one window, reduced to {COLUMNS:,} pixel "
        "columns, and returns them as one array. The upper panels show the "
        "seconds per request, as the median and the 5th to 95th percentile "
        "band of each run, against simultaneous users. The lower panels show "
        "the requests that the run completed per second. The dotted lines in "
        "the lower panels give the requests per second if every request took "
        "the median seconds of the 1-user run. The dotted gray line in the "
        "upper panels is the timeout. A cross marks a run with a failure rate "
        "above 0. Both methods read from S3. The runs hold "
        f"{parts}, warm-up requests left out."
        f"{failed_request_sentence(runs, percentiles=True)}"
        f"{client_limited_sentence(series)}"
        f"{s3_rate_sentence(series)}"
    )
