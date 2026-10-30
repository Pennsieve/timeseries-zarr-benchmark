import pytest

from plot import figure2 as module
from plot.data import NoDataError
from plot.figure2 import figure2, figure2_series, overhead_floor
from tests.plot.conftest import MEDIAN_FACTOR, base_bytes, trim_reads


def test_series_medians_match_the_synthetic_reads(format_benchmark, seconds):
    series = figure2_series(format_benchmark, "R3-1h")
    row = series[
        (series["store"] == "http")
        & (series["read_kind"] == "all")
        & (series["format"] == "nwb-zarr-time")
        & (series["window_s"] == 10)
    ]
    assert len(row) == 1
    assert row["median_s"].iloc[0] == pytest.approx(
        seconds("nwb-zarr-time", "http", "all", 10) * MEDIAN_FACTOR
    )
    assert row["reads"].iloc[0] == 21


def test_series_leaves_out_direct_and_overhead_reads(format_benchmark):
    series = figure2_series(format_benchmark, "R3-1h")
    assert set(series["format"]) == {
        "edf",
        "nwb-hdf5-time",
        "nwb-hdf5-channel",
        "nwb-zarr-time",
        "nwb-zarr-channel",
        "tszarr",
    }
    assert len(series) == 3 * 2 * 6 * 3


def test_series_raises_for_an_absent_recording(format_benchmark):
    with pytest.raises(NoDataError):
        figure2_series(format_benchmark, "R4")


def test_overhead_floor_is_the_median_per_store(format_benchmark):
    floor = overhead_floor(format_benchmark, "R3-1h")
    assert floor["local"] == pytest.approx(0.005 * MEDIAN_FACTOR)
    assert floor["s3"] == pytest.approx(0.015 * MEDIAN_FACTOR)


def test_figure2_draws_one_panel_per_store_and_read_kind(
    format_benchmark, tmp_path, capture
):
    saved = capture(module)
    output = figure2(format_benchmark, tmp_path, "R3-1h", "fig2-test")
    assert [p.name for p in output.paths] == ["fig2-test.pdf", "fig2-test.png"]
    assert all(p.stat().st_size > 1000 for p in output.paths)
    axes = saved[0].axes
    assert len(axes) == 12
    assert all(
        a.get_xscale() == "log" and a.get_yscale() == "log" for a in axes
    )
    assert len(axes[0].get_lines()) == 7


def test_series_holds_the_median_bytes_per_read(format_benchmark):
    series = figure2_series(format_benchmark, "R3-1h")
    row = series[
        (series["store"] == "s3")
        & (series["read_kind"] == "one")
        & (series["format"] == "tszarr")
        & (series["window_s"] == 60)
    ]
    assert row["median_bytes"].iloc[0] == base_bytes("tszarr", 60)


def test_figure2_draws_the_bytes_panels_below_the_seconds_panels(
    format_benchmark, tmp_path, capture
):
    saved = capture(module)
    figure2(format_benchmark, tmp_path)
    axes = saved[0].axes
    seconds, bytes_panels = axes[:6], axes[6:]
    assert {a.get_ylabel() for a in seconds if a.get_ylabel()} == {
        "median seconds per read"
    }
    assert {a.get_ylabel() for a in bytes_panels if a.get_ylabel()} == {
        "median bytes per read"
    }
    # The overhead line is a seconds line only.
    assert len(bytes_panels[0].get_lines()) == 6
    heights = [
        height
        for line in bytes_panels[0].get_lines()
        for height in line.get_ydata()
    ]
    assert max(heights) == base_bytes("tszarr", 60)


def test_figure2_caption_counts_reads(format_benchmark, tmp_path):
    caption = figure2(format_benchmark, tmp_path, "R3-1h").caption
    assert "of R3" in caption
    assert "median bytes that the store served per read" in caption
    assert "2,268 reads" in caption
    assert "756 reads on S3" in caption
    assert caption.isascii()


def test_figure2_caption_gives_the_fewest_and_most_reads_of_a_combination(
    format_benchmark, tmp_path
):
    caption = figure2(
        trim_reads(format_benchmark, 5), tmp_path, "R3-1h"
    ).caption
    assert "Each combination holds between five and 21 reads." in caption
    assert "the read budget of 1 h, once it has made at least three reads." in (
        caption
    )


def test_figure2_caption_counts_only_the_reads_of_the_plotted_recording(
    format_benchmark, tmp_path
):
    trimmed = trim_reads(format_benchmark, 5)
    assert "Each combination holds 21 reads." in (
        figure2(trimmed, tmp_path, "R1-1h").caption
    )
    assert "Each combination holds between five and 21 reads." in (
        figure2(trimmed, tmp_path, "R3-1h").caption
    )
