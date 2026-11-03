import pandas as pd
import pytest

from plot import figure5 as module
from plot.data import NoDataError
from plot.figure5 import channels_read, figure5, figure5_series
from tests.plot.conftest import MEDIAN_FACTOR

LONG_WINDOW_S = 600
"""Second subset window. The shared fixture stops at 1 min."""


@pytest.fixture
def two_windows(format_benchmark):
    """Return the reads with a copy of the R3-1h reads at 10 s moved to 10 min."""
    short = format_benchmark[
        (format_benchmark["recording"] == "R3-1h")
        & (format_benchmark["window_s"] == 10)
    ].copy()
    short["window_s"] = LONG_WINDOW_S
    short["seconds"] = short["seconds"] * 60
    frame = pd.concat([format_benchmark, short], ignore_index=True)
    frame.attrs = dict(format_benchmark.attrs)
    return frame


def test_channels_read_counts_the_channels_of_each_read_kind() -> None:
    assert channels_read("R3-1h", "one") == 1
    assert channels_read("R3-1h", "channels16") == 16
    assert channels_read("R3-1h", "all") == 200


def test_series_medians_match_the_synthetic_reads(format_benchmark, seconds):
    series = figure5_series(format_benchmark, "R3-1h")
    row = series[
        (series["store"] == "http")
        & (series["read_kind"] == "channels16")
        & (series["format"] == "nwb-zarr-time")
    ]
    assert len(row) == 1
    assert row["channels"].iloc[0] == 16
    assert row["median_s"].iloc[0] == pytest.approx(
        seconds("nwb-zarr-time", "http", "channels16", 10) * MEDIAN_FACTOR
    )
    assert row["reads"].iloc[0] == 21


def test_series_holds_one_row_per_store_format_and_read_kind(format_benchmark):
    series = figure5_series(format_benchmark, "R3-1h")
    assert len(series) == 3 * 6 * 5
    assert set(series["channels"]) == {1, 4, 16, 64, 200}
    assert set(series["format"]).isdisjoint({"zarr-direct", "overhead"})


def test_series_keeps_only_the_windows_of_the_subset_reads(format_benchmark):
    series = figure5_series(format_benchmark, "R3-1h")
    one = series[(series["store"] == "s3") & (series["read_kind"] == "one")]
    assert one["reads"].tolist() == [21] * 6
    assert set(series["window_s"]) == {10}


def test_series_holds_every_subset_window_with_subset_reads(two_windows):
    series = figure5_series(two_windows, "R3-1h")
    assert len(series) == 2 * 3 * 6 * 5
    assert set(series["window_s"]) == {10, LONG_WINDOW_S}


def test_series_drops_a_subset_window_without_subset_reads(two_windows):
    one_and_all = two_windows[
        (two_windows["window_s"] != LONG_WINDOW_S)
        | two_windows["read_kind"].isin(["one", "all"])
    ]
    series = figure5_series(one_and_all, "R3-1h")
    assert set(series["window_s"]) == {10}


def test_series_raises_for_a_recording_without_subset_reads(format_benchmark):
    with pytest.raises(NoDataError, match="no subset reads of R1-1h"):
        figure5_series(format_benchmark, "R1-1h")


def test_series_raises_for_an_absent_recording(format_benchmark):
    with pytest.raises(NoDataError):
        figure5_series(format_benchmark, "R4")


def test_figure5_draws_one_panel_per_store_in_two_rows_per_window(
    format_benchmark, tmp_path, capture
):
    saved = capture(module)
    output = figure5(format_benchmark, tmp_path, "R3-1h", "fig5-test")
    assert [p.name for p in output.paths] == ["fig5-test.pdf", "fig5-test.png"]
    assert all(p.stat().st_size > 1000 for p in output.paths)
    axes = saved[0].axes
    assert len(axes) == 6
    assert all(
        a.get_xscale() == "log" and a.get_yscale() == "log" for a in axes
    )
    assert len(axes[0].get_lines()) == 7
    assert len(axes[3].get_lines()) == 6


def test_figure5_labels_the_ticks_with_the_channel_counts(
    format_benchmark, tmp_path, capture
):
    saved = capture(module)
    figure5(format_benchmark, tmp_path)
    saved[0].canvas.draw()
    axis = saved[0].axes[3]
    assert axis.get_xticks().tolist() == [1, 4, 16, 64, 200]
    assert [t.get_text() for t in axis.get_xticklabels()] == [
        "1",
        "4",
        "16",
        "64",
        "200",
    ]
    assert saved[0].axes[3].get_xlabel() == "channels read"


def test_figure5_caption_counts_reads(format_benchmark, tmp_path):
    caption = figure5(format_benchmark, tmp_path, "R3-1h").caption
    assert caption.startswith("Figure 5. ")
    assert "of R3-1h at the 10 s window" in caption
    assert "median bytes that the store served per read" in caption
    assert "1,890 reads" in caption
    assert "630 reads on S3" in caption
    assert "Each combination holds 21 reads." in caption
    assert "the read budget of 1 h" in caption
    assert caption.isascii()


def test_figure5_draws_two_rows_for_each_window(two_windows, tmp_path, capture):
    saved = capture(module)
    figure5(two_windows, tmp_path, "R3-1h", "fig5-test")
    axes = saved[0].axes
    assert len(axes) == 12
    assert axes[0].get_title() == "local disk, 10 s"
    assert axes[6].get_title() == "local disk, 10 min"
    assert len(axes[6].get_lines()) == 7
    assert len(axes[9].get_lines()) == 6
    assert axes[9].get_xlabel() == "channels read"
    assert axes[3].get_xlabel() == ""


def test_figure5_caption_names_both_windows(two_windows, tmp_path):
    caption = figure5(two_windows, tmp_path, "R3-1h").caption
    assert "of R3-1h at the 10 s and 10 min windows" in caption
    assert "3,780 reads" in caption
    assert "1,260 reads on S3" in caption
    assert caption.isascii()
