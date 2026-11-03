"""Colors, labels, and file writing shared by every figure.

Figures draw on `matplotlib.figure.Figure` objects with the Agg canvas. No
pyplot state is involved.
"""

from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.typing import LineStyleType

from tszbench.schemas import SUBSET_CHANNELS, SUBSET_KINDS

STORE_COLORS = {"local": "#0072B2", "http": "#E69F00", "s3": "#009E73"}
"""One color per store in every figure. The Wong colorblind-safe palette."""

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
