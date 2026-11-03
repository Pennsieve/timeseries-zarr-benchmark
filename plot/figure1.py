"""Figure 1: median seconds per read on S3 against the channels read and the window."""

from pathlib import Path
from typing import Any

from plot.data import Output
from plot.formats import draw, seconds_caption
from plot.style import save_figure

STORES_DRAWN = ("s3",)


def figure1(
    format_benchmark: Any,
    out_dir: Path,
    recording: str = "R3-1h",
    stem: str = "fig1",
) -> Output:
    """Write Figure 1 for one recording as `<stem>.pdf` and `<stem>.png`.

    One row for S3 with log axes. The channel panels draw the median seconds
    per read against the channels read, one panel per window with subset reads.
    The window panels draw it against the window length, one for the
    one-channel reads and one for the all-channel reads. Each format is one
    line with a band from the 25th to the 75th percentile. A dotted line marks
    the median 1-byte read.

    Args:
        format_benchmark: Data frame of `format-reads.csv`.
        out_dir: Directory to write into.
        recording: Recording to plot.
        stem: File name without extension.

    Returns:
        The two files and the caption.

    Raises:
        NoDataError: The recording has no read of a plotted format on S3.
    """
    drawing = draw(format_benchmark, recording, STORES_DRAWN, seconds=True)
    paths = save_figure(drawing.figure, out_dir, stem)
    header = format_benchmark.attrs.get("header")
    return Output(paths, seconds_caption(1, drawing, recording, header))
