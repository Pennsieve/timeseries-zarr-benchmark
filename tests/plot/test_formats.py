import pandas as pd
import pytest

from plot import figure1 as figure1_module
from plot import figure2 as figure2_module
from plot import figure5 as figure5_module
from plot.data import NoDataError
from plot.figure1 import figure1
from plot.figure2 import figure2
from plot.figure5 import figure5
from plot.formats import (
    SAMPLE_GID,
    channels_read,
    format_series,
    overhead_floor,
    panels_of,
)
from plot.style import (
    FORMAT_LABELS,
    FORMAT_LINES,
    LINE_WIDTH,
    TSZARR_LINE_WIDTH,
)
from tests.plot.conftest import MEDIAN_FACTOR, base_bytes

LONG_WINDOW_S = 600
"""Second subset window. The shared fixture stops at 1 min."""

COLUMNS = [
    "store",
    "read_kind",
    "format",
    "window_s",
    "channels",
    "median_s",
    "p25_s",
    "p75_s",
    "median_bytes",
    "reads",
]


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


def data_lines(axis):
    """Return the lines of an axis without the samples next to the labels."""
    return [line for line in axis.get_lines() if line.get_gid() != SAMPLE_GID]


def samples_of(axis):
    """Return the line samples of an axis."""
    return [line for line in axis.get_lines() if line.get_gid() == SAMPLE_GID]


def titles(axes):
    return [axis.get_title() for axis in axes]


def test_channels_read_counts_the_channels_of_each_read_kind() -> None:
    assert channels_read("R3-1h", "one") == 1
    assert channels_read("R3-1h", "channels16") == 16
    assert channels_read("R3-1h", "all") == 200


def test_series_has_one_row_per_store_read_kind_format_and_window(
    format_benchmark,
):
    series = format_series(format_benchmark, "R3-1h")
    assert list(series.columns) == COLUMNS
    # 3 stores x 6 formats x (one and all at 3 windows, 3 subsets at 10 s)
    assert len(series) == 3 * 6 * 9
    assert not series.duplicated(
        ["store", "read_kind", "format", "window_s"]
    ).any()
    assert set(series["format"]).isdisjoint({"zarr-direct", "overhead"})


def test_series_holds_the_channels_of_each_read_kind(format_benchmark):
    series = format_series(format_benchmark, "R3-1h")
    channels = series.groupby("read_kind")["channels"].unique()
    assert {kind: list(values) for kind, values in channels.items()} == {
        "one": [1],
        "channels4": [4],
        "channels16": [16],
        "channels64": [64],
        "all": [200],
    }


def test_series_holds_the_median_and_quartiles_of_the_seconds(
    format_benchmark, seconds
):
    series = format_series(format_benchmark, "R3-1h")
    row = series[
        (series["store"] == "http")
        & (series["read_kind"] == "channels16")
        & (series["format"] == "nwb-zarr-time")
    ]
    base = seconds("nwb-zarr-time", "http", "channels16", 10)
    assert len(row) == 1
    assert row["median_s"].iloc[0] == pytest.approx(base * MEDIAN_FACTOR)
    assert row["p25_s"].iloc[0] == pytest.approx(base * 1.05)
    assert row["p75_s"].iloc[0] == pytest.approx(base * 1.15)
    assert row["reads"].iloc[0] == 21


def test_series_holds_the_median_bytes_per_read(format_benchmark):
    series = format_series(format_benchmark, "R3-1h")
    row = series[
        (series["store"] == "s3")
        & (series["read_kind"] == "one")
        & (series["format"] == "tszarr")
        & (series["window_s"] == 60)
    ]
    assert row["median_bytes"].iloc[0] == base_bytes("tszarr", 60)


def test_series_keeps_only_the_stores_asked_for(format_benchmark):
    series = format_series(format_benchmark, "R3-1h", ("s3",))
    assert set(series["store"]) == {"s3"}


def test_series_raises_for_an_absent_recording(format_benchmark):
    with pytest.raises(NoDataError, match="no reads of R4"):
        format_series(format_benchmark, "R4")


def test_series_raises_for_a_store_without_reads(format_benchmark):
    without_s3 = format_benchmark[format_benchmark["store"] != "s3"]
    with pytest.raises(NoDataError, match="on S3"):
        format_series(without_s3, "R3-1h", ("s3",))


def test_overhead_floor_is_the_median_per_store(format_benchmark):
    floor = overhead_floor(format_benchmark, "R3-1h")
    assert floor["local"] == pytest.approx(0.005 * MEDIAN_FACTOR)
    assert floor["s3"] == pytest.approx(0.015 * MEDIAN_FACTOR)


def test_panels_of_a_recording_with_subset_reads_at_one_window(
    format_benchmark,
):
    panels = panels_of(format_series(format_benchmark, "R3-1h"))
    assert titles_of(panels) == ["10 s window", "one channel", "all channels"]
    assert panels[0].ticks == (1, 4, 16, 64, 200)
    assert panels[1].ticks == (1, 10, 60)
    assert [panel.x_label for panel in panels] == [
        "channels read",
        "window length",
        "window length",
    ]


def test_panels_of_a_recording_without_subset_reads(format_benchmark):
    panels = panels_of(format_series(format_benchmark, "R1-1h"))
    assert titles_of(panels) == ["one channel", "all channels"]


def test_panels_of_a_recording_with_two_subset_windows(two_windows):
    panels = panels_of(format_series(two_windows, "R3-1h"))
    assert titles_of(panels) == [
        "10 s window",
        "10 min window",
        "one channel",
        "all channels",
    ]
    assert panels[2].ticks == (1, 10, 60, LONG_WINDOW_S)


def titles_of(panels):
    return [panel.title for panel in panels]


def test_panel_tick_labels_are_counts_or_window_lengths(format_benchmark):
    panels = panels_of(format_series(format_benchmark, "R3-1h"))
    assert [panels[0].tick_label(value) for value in panels[0].ticks] == [
        "1",
        "4",
        "16",
        "64",
        "200",
    ]
    assert [panels[1].tick_label(value) for value in panels[1].ticks] == [
        "1 s",
        "10 s",
        "1 min",
    ]


def test_figure1_has_one_row_of_three_panels(
    format_benchmark, tmp_path, capture
):
    saved = capture(figure1_module)
    figure1(format_benchmark, tmp_path)
    axes = saved[0].axes
    assert titles(axes) == ["10 s window", "one channel", "all channels"]
    assert axes[0].get_xlabel() == "channels read"
    assert axes[1].get_xlabel() == "window length"
    assert axes[0].get_ylabel() == "median seconds per read"
    assert all(
        axis.get_xscale() == "log" and axis.get_yscale() == "log"
        for axis in axes
    )


def test_figure1_has_four_panels_for_two_subset_windows(
    two_windows, tmp_path, capture
):
    saved = capture(figure1_module)
    figure1(two_windows, tmp_path)
    assert titles(saved[0].axes) == [
        "10 s window",
        "10 min window",
        "one channel",
        "all channels",
    ]


def test_figure1_has_two_panels_without_subset_reads(
    format_benchmark, tmp_path, capture
):
    saved = capture(figure1_module)
    figure1(format_benchmark, tmp_path, "R1-1h")
    assert titles(saved[0].axes) == ["one channel", "all channels"]


def test_figure2_has_one_row_per_store(format_benchmark, tmp_path, capture):
    saved = capture(figure2_module)
    figure2(format_benchmark, tmp_path)
    axes = saved[0].axes
    assert len(axes) == 9
    assert titles(axes[:3]) == ["10 s window", "one channel", "all channels"]
    assert titles(axes[3:]) == [""] * 6
    assert [axis.get_ylabel() for axis in axes[::3]] == [
        "local disk\nmedian seconds per read",
        "HTTP\nmedian seconds per read",
        "S3\nmedian seconds per read",
    ]
    assert [axis.get_xlabel() for axis in axes[6:]] == [
        "channels read",
        "window length",
        "window length",
    ]


def test_figure5_has_one_row_of_bytes_panels(
    format_benchmark, tmp_path, capture
):
    saved = capture(figure5_module)
    figure5(format_benchmark, tmp_path)
    axes = saved[0].axes
    assert titles(axes) == ["10 s window", "one channel", "all channels"]
    assert axes[0].get_ylabel() == "median bytes per read"


def draw_figures(format_benchmark, tmp_path, capture):
    one = capture(figure1_module)
    two = capture(figure2_module)
    five = capture(figure5_module)
    figure1(format_benchmark, tmp_path)
    figure2(format_benchmark, tmp_path)
    figure5(format_benchmark, tmp_path)
    return one[0], two[0], five[0]


def rows_of(figure, columns=3):
    axes = figure.axes
    return [
        axes[start : start + columns] for start in range(0, len(axes), columns)
    ]


def test_every_panel_of_a_row_shares_the_y_axis(
    format_benchmark, tmp_path, capture
):
    _, figure, _ = draw_figures(format_benchmark, tmp_path, capture)
    rows = rows_of(figure)
    shared = figure.axes[0].get_shared_y_axes()
    for row in rows:
        assert all(shared.joined(row[0], axis) for axis in row)
    assert not shared.joined(rows[0][0], rows[1][0])


def test_every_panel_of_a_column_shares_the_x_axis(
    format_benchmark, tmp_path, capture
):
    _, figure, _ = draw_figures(format_benchmark, tmp_path, capture)
    rows = rows_of(figure)
    shared = figure.axes[0].get_shared_x_axes()
    for column in range(3):
        assert all(shared.joined(rows[0][column], row[column]) for row in rows)


def test_the_x_ticks_sit_at_the_counts_and_windows_without_minor_ticks(
    format_benchmark, tmp_path, capture
):
    figure, _, _ = draw_figures(format_benchmark, tmp_path, capture)
    figure.canvas.draw()
    axes = figure.axes
    assert axes[0].get_xticks().tolist() == [1, 4, 16, 64, 200]
    assert [t.get_text() for t in axes[0].get_xticklabels()] == [
        "1",
        "4",
        "16",
        "64",
        "200",
    ]
    assert axes[1].get_xticks().tolist() == [1, 10, 60]
    assert [t.get_text() for t in axes[1].get_xticklabels()] == [
        "1 s",
        "10 s",
        "1 min",
    ]
    assert len(axes[0].xaxis.get_minorticklocs()) == 0


def test_the_seconds_figures_draw_one_line_per_format_and_a_1_byte_line(
    format_benchmark, tmp_path, capture
):
    figure, _, _ = draw_figures(format_benchmark, tmp_path, capture)
    lines = data_lines(figure.axes[0])
    assert len(lines) == 7
    assert lines[-1].get_linestyle() == ":"
    assert lines[-1].get_color() == "#777777"
    assert lines[-1].get_ydata()[0] == pytest.approx(0.015 * MEDIAN_FACTOR)


def test_the_tszarr_line_is_wider_and_drawn_last(
    format_benchmark, tmp_path, capture
):
    figure, _, _ = draw_figures(format_benchmark, tmp_path, capture)
    widths = [line.get_linewidth() for line in data_lines(figure.axes[0])[:6]]
    assert widths == [LINE_WIDTH] * 5 + [TSZARR_LINE_WIDTH]
    assert data_lines(figure.axes[0])[5].get_color() == "#0072B2"
    zorders = {line.get_zorder() for line in data_lines(figure.axes[0])[:5]}
    assert max(zorders) < data_lines(figure.axes[0])[5].get_zorder()


def test_the_seconds_figures_draw_a_quartile_band_for_each_format(
    format_benchmark, tmp_path, capture
):
    figure, _, _ = draw_figures(format_benchmark, tmp_path, capture)
    bands = figure.axes[0].collections
    assert len(bands) == 6
    assert all(band.get_alpha() == pytest.approx(0.15) for band in bands)
    assert all(band.get_linewidth()[0] == 0 for band in bands)
    heights = bands[0].get_paths()[0].vertices[:, 1]
    assert heights.min() < heights.max()


def test_the_bytes_figure_draws_no_band_and_no_1_byte_line(
    format_benchmark, tmp_path, capture
):
    _, _, figure = draw_figures(format_benchmark, tmp_path, capture)
    assert all(len(axis.collections) == 0 for axis in figure.axes)
    assert all(len(data_lines(axis)) == 6 for axis in figure.axes)
    assert "1-byte read" not in [
        text.get_text() for text in figure.axes[-1].texts
    ]


def test_the_bytes_figure_draws_the_median_bytes(
    format_benchmark, tmp_path, capture
):
    _, _, figure = draw_figures(format_benchmark, tmp_path, capture)
    heights = [
        height
        for line in data_lines(figure.axes[1])
        for height in line.get_ydata()
    ]
    assert max(heights) == base_bytes("tszarr", 60)


def test_the_y_ticks_are_plain_numbers_for_seconds_and_bytes_for_bytes(
    format_benchmark, tmp_path, capture
):
    seconds_figure, _, bytes_figure = draw_figures(
        format_benchmark, tmp_path, capture
    )
    seconds_format = seconds_figure.axes[0].yaxis.get_major_formatter()
    assert [seconds_format(value) for value in (0.01, 0.1, 1, 10, 100)] == [
        "0.01",
        "0.1",
        "1",
        "10",
        "100",
    ]
    bytes_format = bytes_figure.axes[0].yaxis.get_major_formatter()
    assert [bytes_format(value) for value in (1000, 1e6, 1e9)] == [
        "1 kB",
        "1 MB",
        "1 GB",
    ]


def test_the_last_panel_of_each_row_labels_every_line(
    format_benchmark, tmp_path, capture
):
    one, two, five = draw_figures(format_benchmark, tmp_path, capture)
    format_labels = [
        "EDF",
        "NWB HDF5, time chunks",
        "NWB HDF5, channel chunks",
        "NWB Zarr, time chunks",
        "NWB Zarr, channel chunks",
        "TS-Zarr",
    ]
    for figure in (one, two):
        for row in rows_of(figure):
            labels = [text.get_text() for text in row[-1].texts]
            assert sorted(labels) == sorted([*format_labels, "1-byte read"])
            assert all(not axis.texts for axis in row[:-1])
    for row in rows_of(five):
        labels = [text.get_text() for text in row[-1].texts]
        assert sorted(labels) == sorted(format_labels)


def test_the_labels_of_a_row_do_not_overlap(
    format_benchmark, tmp_path, capture
):
    for figure in draw_figures(format_benchmark, tmp_path, capture):
        figure.canvas.draw()
        for row in rows_of(figure):
            extents = [text.get_window_extent() for text in row[-1].texts]
            for index, extent in enumerate(extents):
                for other in extents[index + 1 :]:
                    assert not extent.overlaps(other)


def test_the_tick_labels_of_neighboring_panels_do_not_touch(
    two_windows, tmp_path, capture
):
    saved = capture(figure2_module)
    figure2(two_windows, tmp_path)
    figure = saved[0]
    figure.canvas.draw()
    bottom = rows_of(figure, columns=4)[-1]
    assert len(bottom) == 4
    for left, right in zip(bottom, bottom[1:], strict=False):
        last = [t for t in left.get_xticklabels() if t.get_text()][-1]
        first = [t for t in right.get_xticklabels() if t.get_text()][0]
        gap = first.get_window_extent().x0 - last.get_window_extent().x1
        assert gap > 0.18 * figure.dpi


def test_the_labels_match_the_colors_of_their_lines(
    format_benchmark, tmp_path, capture
):
    figure, _, _ = draw_figures(format_benchmark, tmp_path, capture)
    axis = figure.axes[-1]
    colors = {text.get_text(): text.get_color() for text in axis.texts}
    assert colors["TS-Zarr"] == "#0072B2"
    assert colors["EDF"] == "#000000"
    assert colors["1-byte read"] == "#777777"


def test_each_format_label_has_a_sample_of_its_line_style_and_marker(
    format_benchmark, tmp_path, capture
):
    figures = draw_figures(format_benchmark, tmp_path, capture)
    label_of = {FORMAT_LABELS[name]: name for name in FORMAT_LINES}
    for figure in figures:
        figure.canvas.draw()
        for row in rows_of(figure):
            axis = row[-1]
            samples = samples_of(axis)
            assert len(samples) == len(axis.texts)
            for text in axis.texts:
                (sample,) = [
                    line
                    for line in samples
                    if line.get_ydata()[0] == pytest.approx(text.xy[1])
                ]
                name = label_of.get(text.get_text())
                if name is None:
                    assert text.get_text() == "1-byte read"
                    name = "overhead"
                color, style, marker = FORMAT_LINES[name]
                assert sample.get_color() == color
                assert sample.get_linestyle() == style
                if name != "overhead":
                    assert sample.get_marker() == marker


def test_the_samples_sit_between_the_axis_and_their_labels(
    format_benchmark, tmp_path, capture
):
    for figure in draw_figures(format_benchmark, tmp_path, capture):
        figure.canvas.draw()
        for row in rows_of(figure):
            axis = row[-1]
            edge = axis.get_window_extent().x1
            labels = [text.get_window_extent() for text in axis.texts]
            for sample in samples_of(axis):
                extent = sample.get_window_extent()
                assert extent.x0 >= edge
                assert all(extent.x1 < label.x0 for label in labels)
