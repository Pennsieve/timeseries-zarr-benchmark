"""Figure 5: median bytes per read against the channels read and the window."""

from pathlib import Path
from typing import Any

from plot.data import Output
from plot.formats import bytes_caption, draw
from plot.style import save_figure

STORES_DRAWN = ("s3",)


def figure5(
    format_benchmark: Any,
    out_dir: Path,
    recording: str = "R3-1h",
    stem: str = "fig5",
) -> Output:
    """Write Figure 5 for one recording as `<stem>.pdf` and `<stem>.png`.

    The layout of Figure 1 with the median bytes that S3 served per read on
    the y axis, and no band and no 1-byte line. The bytes of a read do not
    depend on the store.

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
    drawing = draw(format_benchmark, recording, STORES_DRAWN, seconds=False)
    paths = save_figure(drawing.figure, out_dir, stem)
    header = format_benchmark.attrs.get("header")
    return Output(paths, bytes_caption(5, drawing, recording, header))
