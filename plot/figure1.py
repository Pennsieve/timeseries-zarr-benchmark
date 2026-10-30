"""Figure 1: seconds per read at the 10 s all-channel window as raincloud plots.

Reads `format-reads.csv` only.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from plot.data import (
    COMBINATION,
    OVERHEAD,
    NoDataError,
    Output,
    count_noun,
    join_words,
    overhead_rows,
    reads_sentence,
    recording_order,
    window_label,
)
from plot.style import (
    FORMAT_LABELS,
    STORE_COLORS,
    STORE_LABELS,
    new_figure,
    save_figure,
)
from tszbench.schemas import READ_FORMATS, STORES

WINDOW_S = 10
READ_KIND = "all"
JITTER_SEED = 1
SLOT_HEIGHT = 0.28
"""Vertical space of one store inside the row of a format."""
CLOUD_BASE = 0.03
CLOUD_HEIGHT = 0.11
BOX_HEIGHT = 0.045
RAIN_OFFSET = -0.07
RAIN_SPREAD = 0.035
DENSITY_POINTS = 200
PANEL_WIDTH = 3.6
ROW_HEIGHT = 0.95


def figure1_rows(format_benchmark: Any) -> Any:
    """Return the reads that Figure 1 draws, as a data frame.

    The columns are `COMBINATION` and `seconds`.

    A format row holds the reads at the 10 s window over all channels. The
    overhead rows of each recording panel follow `plot.data.overhead_rows`,
    restricted to the 10 s all-channel reads when any exist.

    Raises:
        NoDataError: No format has a read at the 10 s all-channel window.
    """
    selected = format_benchmark[
        format_benchmark["format"].isin(READ_FORMATS)
        & (format_benchmark["window_s"] == WINDOW_S)
        & (format_benchmark["read_kind"] == READ_KIND)
    ]
    if selected.empty:
        raise NoDataError(
            f"no reads at {WINDOW_S} s over {READ_KIND} channels in format_benchmark"
        )
    parts = [selected]
    for recording in recording_order(selected["recording"]):
        floor = overhead_rows(format_benchmark, recording)
        matching = floor[
            (floor["window_s"] == WINDOW_S) & (floor["read_kind"] == READ_KIND)
        ]
        floor = matching if len(matching) else floor
        parts.append(floor.assign(recording=recording))
    frame = pd.concat(parts)
    return frame[[*COMBINATION, "seconds"]].reset_index(drop=True)


def _cloud(axis: Any, logs: Any, base: float, color: str) -> None:
    """Draw the half violin of log10 seconds with its base at `base`."""
    if len(logs) < 2 or np.ptp(logs) == 0:
        return
    grid = np.linspace(logs.min(), logs.max(), DENSITY_POINTS)
    density = gaussian_kde(logs)(grid)
    density = CLOUD_HEIGHT * density / density.max()
    axis.fill_between(
        10**grid,
        base,
        base + density,
        color=color,
        alpha=0.65,
        linewidth=0,
    )


def _box(axis: Any, values: Any, center: float, color: str) -> None:
    """Draw the quartile box, median, and whiskers of a slot."""
    axis.boxplot(
        values,
        positions=[center],
        widths=BOX_HEIGHT,
        orientation="horizontal",
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
        boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 0.8},
        whiskerprops={"color": color, "linewidth": 0.8},
        capprops={"color": color, "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1.2},
    )


def _rain(axis: Any, values: Any, center: float, color: str, rng: Any) -> None:
    """Draw one jittered point per read below the box."""
    jitter = rng.uniform(-RAIN_SPREAD, RAIN_SPREAD, len(values))
    axis.scatter(
        values,
        center + RAIN_OFFSET + jitter,
        s=3,
        color=color,
        alpha=0.5,
        linewidths=0,
        rasterized=True,
    )


def figure1(format_benchmark: Any, out_dir: Path, stem: str = "fig1") -> Output:
    """Write Figure 1 as `<stem>.pdf` and `<stem>.png`.

    One panel per recording, one row per format in `READ_FORMATS` order and then
    the overhead row, one color per store, on a log axis of seconds. Each store
    in a row draws a half violin of a Gaussian kernel density estimate of
    log10 seconds, a box of the quartiles with the median, and one jittered
    point per read.

    Args:
        format_benchmark: Data frame of `format-reads.csv`.
        out_dir: Directory to write into.
        stem: File name without extension.

    Returns:
        The two files and the caption.

    Raises:
        NoDataError: See `figure1_rows`.
    """
    rows = figure1_rows(format_benchmark)
    recordings = recording_order(rows["recording"])
    labels = [
        name
        for name in (*READ_FORMATS, OVERHEAD)
        if name in set(rows["format"])
    ]
    stores = [name for name in STORES if name in set(rows["store"])]
    figure = new_figure(
        PANEL_WIDTH * len(recordings) + 1.5, ROW_HEIGHT * len(labels) + 1.6
    )
    axes = figure.subplots(
        1, len(recordings), sharex=True, sharey=True, squeeze=False
    )[0]
    rng = np.random.default_rng(JITTER_SEED)
    top = len(labels) - 1
    for axis, recording in zip(axes, recordings, strict=True):
        panel = rows[rows["recording"] == recording]
        for row, label in enumerate(labels):
            for position, store in enumerate(stores):
                values = panel[
                    (panel["format"] == label) & (panel["store"] == store)
                ]["seconds"].to_numpy(dtype=float)
                values = values[values > 0]
                if not len(values):
                    continue
                offset = (position - (len(stores) - 1) / 2) * SLOT_HEIGHT
                center = top - row - offset
                color = STORE_COLORS[store]
                _cloud(axis, np.log10(values), center + CLOUD_BASE, color)
                _box(axis, values, center, color)
                _rain(axis, values, center, color, rng)
        axis.set_xscale("log")
        axis.set_title(recording)
        axis.set_xlabel("seconds per read")
        axis.set_yticks([top - row for row in range(len(labels))])
        axis.set_yticklabels([FORMAT_LABELS[name] for name in labels])
        axis.set_ylim(-0.6, top + 0.6)
        axis.grid(axis="x", linewidth=0.3, alpha=0.5)
    figure.legend(
        handles=[
            Patch(color=STORE_COLORS[store], label=STORE_LABELS[store])
            for store in stores
        ],
        loc="outside lower center",
        ncols=len(stores),
        frameon=False,
    )
    paths = save_figure(figure, out_dir, stem)
    return Output(
        paths,
        _caption(
            format_benchmark.attrs.get("header"), rows, recordings, stores
        ),
    )


def _caption(
    header: dict[str, str] | None,
    rows: Any,
    recordings: list[str],
    stores: list[str],
) -> str:
    """Return the caption text of Figure 1 with its read counts."""
    parts = []
    for recording in recordings:
        panel = rows[rows["recording"] == recording]
        counts = ", ".join(
            count_noun(int((panel["store"] == store).sum()), "read")
            + f" on {STORE_LABELS[store]}"
            for store in stores
        )
        parts.append(f"{recording}: {counts}.")
    return (
        f"Figure 1. Seconds per read of the {window_label(WINDOW_S)} window "
        "over all channels, one point per read, on a log axis. Each box marks "
        "the median and the quartiles, and each half violin is a kernel "
        "density estimate of the logarithm of the seconds. The overhead row "
        "reads a 1-byte object. Stores are "
        + join_words([STORE_LABELS[store] for store in stores])
        + ". Reads drawn, overhead included. "
        + " ".join(parts)
        + reads_sentence(rows, header)
    )
