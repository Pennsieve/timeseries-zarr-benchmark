"""Colors, labels, and file writing shared by every figure.

Figures draw on `matplotlib.figure.Figure` objects with the Agg canvas. No
pyplot state is involved.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.typing import LineStyleType

from tszbench.schemas import SUBSET_CHANNELS, SUBSET_KINDS

METHOD_COLORS = {"server": "#D55E00", "precomputed": "#0072B2"}
"""One color per resampling benchmark method in every figure. The precomputed
read is a TS-Zarr read and has the TS-Zarr blue."""

STORE_LABELS = {"local": "local disk", "http": "HTTP", "s3": "S3"}
METHOD_LABELS = {"server": "server", "precomputed": "precomputed"}

FORMAT_LABELS = {
    "edf": "EDF",
    "nwb-hdf5-time": "NWB HDF5, time chunks",
    "nwb-hdf5-channel": "NWB HDF5, channel chunks",
    "nwb-zarr-time": "NWB Zarr, time chunks",
    "nwb-zarr-channel": "NWB Zarr, channel chunks",
    "tszarr": "TS-Zarr",
    "tszarr-raw": "TS-Zarr, raw only",
    "overhead": "overhead (1 byte)",
}

FORMAT_LINES: dict[str, tuple[str, LineStyleType, str]] = {
    "edf": ("#000000", "-", "o"),
    "nwb-hdf5-time": ("#D55E00", "-", "s"),
    "nwb-hdf5-channel": ("#D55E00", "--", "D"),
    "nwb-zarr-time": ("#56B4E9", "-", "^"),
    "nwb-zarr-channel": ("#56B4E9", "--", "v"),
    "tszarr": ("#0072B2", "-", "P"),
    "overhead": ("#777777", ":", "x"),
}
"""Color, line style, and marker of each format line. The Zarr formats are
blue, EDF is black, and NWB HDF5 is vermillion, from the Wong palette. Time
chunks are solid and channel chunks are dashed."""

LINE_WIDTH = 1.2
"""Width of a data line, in points."""

TSZARR_LINE_WIDTH = 2.0
"""Width of the TS-Zarr line, in points."""

RECORDING_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
"""Markers of the recordings in Figure 4, in recording order."""

READ_KIND_LABELS = {
    "one": "one channel",
    **{
        kind: f"{count} channels"
        for kind, count in zip(SUBSET_KINDS, SUBSET_CHANNELS, strict=True)
    },
    "all": "all channels",
}
"""Label of each read kind, in the order of `tszbench.schemas.READ_KINDS`."""

FAILED_MARKER = "X"
"""Marker of a resampling benchmark run whose failure rate is above 0."""

CLIENT_LIMITED_MARKER = "o"
"""Marker of a resampling benchmark run that the load generators may have limited. It
is drawn as a large open ring around the point of the run."""

RASTER_DPI = 200

LABEL_FONT_SIZE = 8.0
"""Font size of a direct line label, in points."""

LABEL_GAP = 1.15
"""Least distance between the centers of two direct labels, in label heights."""

LABEL_OFFSET_POINTS = 4.0
"""Distance from the right edge of the axis to a direct label, in points."""


def new_figure(width: float, height: float) -> Any:
    """Return an empty figure of the given size in inches on the Agg canvas."""
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def save_figure(figure: Any, out_dir: Path, stem: str) -> tuple[Path, Path]:
    """Write `<stem>.pdf` and `<stem>.png` under out_dir and return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    figure.savefig(pdf)
    figure.savefig(png, dpi=RASTER_DPI)
    return pdf, png


def spread_labels(
    positions: Sequence[float], gap: float, low: float, high: float
) -> list[float]:
    """Return label centers near positions, at least gap apart.

    Labels closer than gap form a group. A group is centered on the mean of
    the positions of its labels and keeps their order. Every center lies in
    [low, high] when the labels fit in that range.

    Args:
        positions: Wanted center of each label.
        gap: Least distance between two centers.
        low: Lowest center.
        high: Highest center.

    Returns:
        One center per position, in the order of positions.
    """
    order = sorted(range(len(positions)), key=lambda index: positions[index])
    groups = [[index] for index in order]
    centers = [float(positions[index]) for index in order]

    def half(group: list[int]) -> float:
        return (len(group) - 1) * gap / 2

    merged = True
    while merged:
        merged = False
        for k, group in enumerate(groups):
            if 2 * half(group) > high - low:
                centers[k] = (low + high) / 2
            else:
                centers[k] = min(
                    max(centers[k], low + half(group)), high - half(group)
                )
        for k in range(len(groups) - 1):
            top = centers[k] + half(groups[k])
            bottom = centers[k + 1] - half(groups[k + 1])
            if bottom - top < gap:
                group = groups[k] + groups[k + 1]
                groups[k : k + 2] = [group]
                centers[k : k + 2] = [
                    sum(float(positions[index]) for index in group) / len(group)
                ]
                merged = True
                break
    result = [0.0] * len(positions)
    for group, center in zip(groups, centers, strict=True):
        start = center - half(group)
        for step, index in enumerate(group):
            result[index] = start + step * gap
    return result


def label_lines(
    axis: Any,
    lines: Sequence[Any],
    labels: Sequence[str],
    font_size: float = LABEL_FONT_SIZE,
) -> list[Any]:
    """Write a label for each line right of the axis, in the line color.

    A label sits at the height of the last finite point of its line and moves
    apart from its neighbors by `LABEL_GAP` label heights. Call it after the
    axis limits are final. A horizontal line from `axhline` works as a line.

    Args:
        axis: Axis that holds the lines.
        lines: `Line2D` objects of the axis.
        labels: One label per line.
        font_size: Font size of the labels, in points.

    Returns:
        The `Annotation` of each label, in the order of lines.
    """
    figure = axis.get_figure()
    figure.draw_without_rendering()
    height = axis.get_window_extent().height
    gap = LABEL_GAP * font_size * figure.dpi / 72 / height
    to_axes = axis.transScale + axis.transLimits
    right = axis.get_xlim()[1]
    wanted = []
    for line in lines:
        heights = np.asarray(line.get_ydata(), dtype=float)
        last = heights[np.isfinite(heights)][-1]
        wanted.append(float(to_axes.transform((right, last))[1]))
    centers = spread_labels(wanted, gap, 0.0, 1.0)
    return [
        axis.annotate(
            label,
            xy=(1.0, center),
            xycoords="axes fraction",
            xytext=(LABEL_OFFSET_POINTS, 0.0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=line.get_color(),
            fontsize=font_size,
            annotation_clip=False,
        )
        for line, label, center in zip(lines, labels, centers, strict=True)
    ]
