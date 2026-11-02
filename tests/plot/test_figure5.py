import pytest

from plot import figure5 as module
from plot.data import NoDataError
from plot.figure5 import channels_read, figure5, figure5_series
from tests.plot.conftest import MEDIAN_FACTOR


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


def test_series_keeps_only_the_window_of_the_subset_reads(format_benchmark):
    series = figure5_series(format_benchmark, "R3-1h")
    one = series[(series["store"] == "s3") & (series["read_kind"] == "one")]
    assert one["reads"].tolist() == [21] * 6


def test_series_raises_for_a_recording_without_subset_reads(format_benchmark):
    with pytest.raises(NoDataError, match="no subset reads of R1-1h"):
        figure5_series(format_benchmark, "R1-1h")


def test_series_raises_for_an_absent_recording(format_benchmark):
    with pytest.raises(NoDataError):
        figure5_series(format_benchmark, "R4")


def test_figure5_draws_one_panel_per_store_in_two_rows(
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
