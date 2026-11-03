"""Figure 2: median seconds per read on every store against the channels and the window."""

from pathlib import Path
from typing import Any

from plot.data import Output
from plot.formats import draw, seconds_caption
from plot.style import save_figure
from tszbench.schemas import STORES


def figure2(
    format_benchmark: Any,
    out_dir: Path,
    recording: str = "R3-1h",
    stem: str = "fig2",
) -> Output:
    """Write Figure 2 for one recording as `<stem>.pdf` and `<stem>.png`.

    The layout of Figure 1 with one row per store.

    Args:
        format_benchmark: Data frame of `format-reads.csv`.
        out_dir: Directory to write into.
        recording: Recording to plot.
        stem: File name without extension.

    Returns:
        The two files and the caption.

    Raises:
        NoDataError: The recording has no read of a plotted format.
    """
    drawing = draw(format_benchmark, recording, STORES, seconds=True)
    paths = save_figure(drawing.figure, out_dir, stem)
    header = format_benchmark.attrs.get("header")
    return Output(paths, seconds_caption(2, drawing, recording, header))
