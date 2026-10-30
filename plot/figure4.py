"""Figure 4: the resampling benchmark seconds per request against raw bytes under the window.

Reads `resampling-runs.csv` only.
"""

from pathlib import Path
from typing import Any

from matplotlib.lines import Line2D
from matplotlib.typing import LineStyleType

from plot.data import (
    NoDataError,
    Output,
    count_noun,
    failed_request_sentence,
    join_words,
    link_limited,
    link_limited_sentence,
    recording_order,
    select_runs,
    without_not_run,
)
from plot.style import (
    FAILED_MARKER,
    LINK_LIMITED_MARKER,
    METHOD_COLORS,
    METHOD_LABELS,
    RECORDING_MARKERS,
    new_figure,
    save_figure,
)
from tszbench.schemas import METHODS

USER_COUNTS = (1, 100)
USER_LINES: dict[int, LineStyleType] = {1: "-", 100: "--"}
"""Line style of each drawn user count."""

FIGURE_WIDTH = 6.6
FIGURE_HEIGHT = 4.2


def figure4_series(runs: Any) -> Any:
    """Return the runs at 1 and 100 users that Figure 4 draws.

    One row per method, recording, window, and user count, chosen by
    `plot.data.select_runs` and sorted by raw bytes. A run that was not run is
    left out. A recording with only one of the two user counts has points at
    that count only.

    Raises:
        NoDataError: No run has 1 or 100 users.
    """
    chosen = without_not_run(select_runs(runs))
    chosen = chosen[chosen["users"].isin(USER_COUNTS)]
    if chosen.empty:
        raise NoDataError("no the resampling benchmark runs at 1 or 100 users")
    return chosen.sort_values(["method", "users", "raw_bytes"]).reset_index(
        drop=True
    )


def figure4(runs: Any, out_dir: Path, stem: str = "fig4") -> Output:
    """Write Figure 4 as `<stem>.pdf` and `<stem>.png`.

    Every recording and window is one point at its raw bytes under the window
    on a log x axis, and at its median seconds per request on a log y axis.
    Each method has a line of one color, solid at 1 user and dashed at 100
    users. The marker shape names the recording. A run with a failure rate
    above 0 carries a distinct marker. A run that a load generator link may
    have limited has an open ring around its marker.

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
        for users in USER_COUNTS:
            rows = series[
                (series["method"] == method) & (series["users"] == users)
            ]
            if rows.empty:
                continue
            color = METHOD_COLORS[method]
            axis.plot(
                rows["raw_bytes"],
                rows["median_s"],
                color=color,
                linestyle=USER_LINES[users],
                linewidth=1.2,
            )
            for name in recordings:
                points = rows[rows["recording"] == name]
                axis.plot(
                    points["raw_bytes"],
                    points["median_s"],
                    color=color,
                    linestyle="none",
                    marker=markers[name],
                    markersize=4.5,
                )
            failed = rows[rows["failure_rate"] > 0]
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
            limited = rows[link_limited(rows)]
            axis.plot(
                limited["raw_bytes"],
                limited["median_s"],
                linestyle="none",
                marker=LINK_LIMITED_MARKER,
                markersize=11,
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=0.8,
            )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("raw bytes under the window")
    axis.set_ylabel("median seconds per request")
    axis.grid(linewidth=0.3, alpha=0.5)
    handles = [
        Line2D([], [], color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        for m in METHODS
        if m in set(series["method"])
    ]
    handles += [
        Line2D(
            [],
            [],
            color="black",
            linestyle=USER_LINES[u],
            label=f"{u} user" if u == 1 else f"{u} users",
        )
        for u in USER_COUNTS
        if u in set(series["users"])
    ]
    handles += [
        Line2D(
            [],
            [],
            color="grey",
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
            color="grey",
            linestyle="none",
            marker=FAILED_MARKER,
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=8,
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
                markersize=11,
                markerfacecolor="none",
                markeredgecolor="black",
                label="generator link limited",
            )
        )
    figure.legend(handles=handles, frameon=False, loc="outside right center")
    paths = save_figure(figure, out_dir, stem)
    return Output(paths, _caption(runs, series, recordings))


def _caption(runs: Any, series: Any, recordings: list[str]) -> str:
    """Return the caption text of Figure 4 with its request counts."""
    parts = ", ".join(
        count_noun(
            int(series[series["users"] == users]["requests"].sum()), "request"
        )
        + f" at {count_noun(users, 'user')}"
        for users in USER_COUNTS
        if users in set(series["users"])
    )
    return (
        "Figure 4. Median seconds per request against the raw float32 bytes "
        "under the window, for "
        + join_words(recordings)
        + ". Each point is one run, at 1 user (solid) or 100 users (dashed). "
        "A cross marks a run with a failure rate above 0. Both methods read "
        f"from S3. The runs hold {parts}, warm-up requests left out."
        f"{failed_request_sentence(runs, percentiles=False)}"
        f"{link_limited_sentence(series)}"
    )
