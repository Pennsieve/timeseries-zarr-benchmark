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

METHOD_COLORS = {"server": "#D55E00", "precomputed": "#CC79A7"}
"""One color per the resampling benchmark method in every figure."""

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
    "edf": ("#332288", "-", "o"),
    "nwb-hdf5-time": ("#882255", "-", "s"),
    "nwb-hdf5-channel": ("#882255", "--", "D"),
    "nwb-zarr-time": ("#AA9A33", "-", "^"),
    "nwb-zarr-channel": ("#AA9A33", "--", "v"),
    "tszarr": ("#117733", "-", "P"),
    "overhead": ("#555555", ":", "x"),
}
"""Color, line style, and marker of each format line in Figure 2."""

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
