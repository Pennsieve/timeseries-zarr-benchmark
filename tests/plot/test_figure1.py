import pandas as pd
import pytest

from plot.data import NoDataError
from plot.figure1 import figure1
from tests.plot.conftest import trim_reads

LONG_WINDOW_S = 600


@pytest.fixture
def two_windows(format_benchmark):
    """Return the reads with a copy of the R3-1h reads at 10 s moved to 10 min."""
    short = format_benchmark[
        (format_benchmark["recording"] == "R3-1h")
        & (format_benchmark["window_s"] == 10)
    ].copy()
    short["window_s"] = LONG_WINDOW_S
    frame = pd.concat([format_benchmark, short], ignore_index=True)
    frame.attrs = dict(format_benchmark.attrs)
    return frame


def test_figure1_writes_pdf_and_png(format_benchmark, tmp_path):
    output = figure1(format_benchmark, tmp_path, "R3-1h", "fig1-test")
    assert [p.name for p in output.paths] == ["fig1-test.pdf", "fig1-test.png"]
    assert all(p.stat().st_size > 1000 for p in output.paths)
    assert output.paths[0].read_bytes().startswith(b"%PDF")
    assert output.paths[1].read_bytes().startswith(b"\x89PNG")


def test_figure1_caption_states_the_statistic_and_the_encodings(
    format_benchmark, tmp_path
):
    caption = figure1(format_benchmark, tmp_path).caption
    assert caption.startswith(
        "Figure 1. Median seconds per read of R3-1h on S3"
    )
    assert "on log axes" in caption
    assert (
        "The channel panel shows the 10 s window and varies the channels read."
    ) in caption
    assert "The window panels show one channel and all channels" in caption
    assert "A band spans the 25th to 75th percentile of the reads." in caption
    assert "The dotted line is the median seconds of a 1-byte read" in caption
    assert caption.isascii()


def test_figure1_caption_counts_the_reads_on_s3_only(
    format_benchmark, tmp_path
):
    caption = figure1(format_benchmark, tmp_path).caption
    # 6 formats x 9 combinations x 21 reads
    assert "The medians cover 1,134 reads on S3." in caption
    assert "local disk" not in caption
    assert "Each combination holds 21 reads." in caption


def test_figure1_caption_states_the_read_budget(format_benchmark, tmp_path):
    caption = figure1(format_benchmark, tmp_path).caption
    assert (
        "A combination stops after the read that brings the sum of its read "
        "seconds to at least the read budget of 1 h, once it has made at "
        "least three reads."
    ) in caption


def test_figure1_caption_leaves_out_the_budget_without_header_keys(
    format_benchmark, tmp_path
):
    format_benchmark.attrs.clear()
    caption = figure1(format_benchmark, tmp_path).caption
    assert "Each combination holds 21 reads." in caption
    assert "budget" not in caption


def test_figure1_caption_omits_the_channel_panels_without_subset_reads(
    format_benchmark, tmp_path
):
    caption = figure1(format_benchmark, tmp_path, "R1-1h").caption
    assert "channel panels" not in caption
    assert "of R1-1h" in caption


def test_figure1_caption_counts_only_the_reads_on_s3(
    format_benchmark, tmp_path
):
    trimmed = trim_reads(format_benchmark, 5)
    assert trimmed.iloc[0]["store"] == "local"
    caption = figure1(trimmed, tmp_path).caption
    assert "Each combination holds 21 reads." in caption


def test_figure1_raises_for_an_absent_recording(format_benchmark, tmp_path):
    with pytest.raises(NoDataError):
        figure1(format_benchmark, tmp_path, "R4")


def test_figure1_raises_for_a_recording_without_reads_on_s3(
    format_benchmark, tmp_path
):
    without_s3 = format_benchmark[format_benchmark["store"] != "s3"]
    with pytest.raises(NoDataError):
        figure1(without_s3, tmp_path)


def test_figure1_caption_says_the_channel_panels_for_two_windows(
    two_windows, tmp_path
):
    caption = figure1(two_windows, tmp_path).caption
    assert (
        "The channel panels show the window lengths 10 s and 10 min and vary "
        "the channels read."
    ) in caption
    assert "The channel panel shows" not in caption
