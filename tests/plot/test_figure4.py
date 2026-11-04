import pytest

from plot import figure4 as module
from plot.data import NoDataError
from plot.figure4 import figure4, figure4_series
from plot.style import METHOD_COLORS
from resampling_benchmark.load.generator import (
    LINK_BYTES_PER_S,
    REQUEST_TIMEOUT_S,
)
from tests.plot.conftest import (
    FAILED_REQUEST_SECONDS,
    mark_client_limited,
    mark_not_run,
    ring_points,
)


def data_lines(axis):
    """Return the solid lines of a method and recording, in draw order."""
    return [line for line in axis.get_lines() if line.get_linestyle() == "-"]


def mark_failed(runs, method, recording, window, users):
    changed = runs.copy()
    mask = (
        (changed["method"] == method)
        & (changed["recording"] == recording)
        & (changed["window_s"] == window)
        & (changed["users"] == users)
    )
    changed.loc[mask, ["median_s", "p5_s", "p95_s"]] = REQUEST_TIMEOUT_S
    changed.loc[mask, "failure_rate"] = 1.0
    return changed


def test_series_keeps_only_1_user_runs_of_every_recording(runs):
    series = figure4_series(runs)
    assert set(series["users"]) == {1}
    assert set(series["recording"]) == {"R1", "R3"}
    assert len(series) == 2 * 2 * 2


def test_series_is_sorted_by_raw_bytes_within_a_line(runs):
    series = figure4_series(runs)
    for _, line in series.groupby(["method", "recording"]):
        assert list(line["raw_bytes"]) == sorted(line["raw_bytes"])


def test_series_leaves_out_a_run_that_was_not_run(runs):
    changed = mark_not_run(runs, "server", "R3", 600, 1)
    series = figure4_series(changed)
    assert len(series) == 2 * 2 * 2 - 1
    assert not series["not_run"].eq("true").any()


def test_series_raises_without_matching_runs(runs):
    with pytest.raises(NoDataError):
        figure4_series(runs[runs["users"] == 10])


def test_figure4_draws_one_solid_line_per_method_and_recording(
    runs, tmp_path, capture
):
    saved = capture(module)
    output = figure4(runs, tmp_path)
    assert all(p.stat().st_size > 1000 for p in output.paths)
    (axis,) = saved[0].axes
    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"
    lines = data_lines(axis)
    assert len(lines) == 2 * 2
    for line in lines:
        assert len(line.get_xdata()) == 2
        assert list(line.get_xdata()) == sorted(line.get_xdata())


def test_figure4_colors_a_line_by_method_and_marks_it_by_recording(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure4(runs, tmp_path)
    (axis,) = saved[0].axes
    lines = data_lines(axis)
    assert {line.get_color() for line in lines} == set(METHOD_COLORS.values())
    assert [line.get_marker() for line in lines] == ["o", "s", "o", "s"]


def test_figure4_draws_a_dotted_reference_line_of_the_link_time(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure4(runs, tmp_path)
    (axis,) = saved[0].axes
    (line,) = [x for x in axis.get_lines() if x.get_linestyle() == ":"]
    xs = list(line.get_xdata())
    ys = list(line.get_ydata())
    assert xs[0] < xs[-1]
    for x, y in zip(xs, ys, strict=True):
        assert y == pytest.approx(x / LINK_BYTES_PER_S)


def test_figure4_labels_the_reference_line_in_the_legend(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure4(runs, tmp_path)
    labels = [text.get_text() for text in saved[0].legends[0].get_texts()]
    assert "raw bytes at 12.5 Gbps" in labels


def test_figure4_labels_the_axes_with_units(runs, tmp_path, capture):
    saved = capture(module)
    figure4(runs, tmp_path)
    (axis,) = saved[0].axes
    assert axis.get_xlabel() == "raw float32 bytes under the window"
    assert axis.get_ylabel() == "median seconds per request"


def test_figure4_caption_counts_requests(runs, tmp_path):
    caption = figure4(runs, tmp_path).caption
    assert caption.startswith("Figure 4. Median seconds per request at 1 user")
    assert "for R1 and R3" in caption
    assert "The runs hold 720 requests at 1 user" in caption
    assert "100 users" not in caption
    assert caption.isascii()


def test_figure4_caption_names_the_columns_and_the_link_rate(runs, tmp_path):
    caption = figure4(runs, tmp_path).caption
    assert "in 2,000 pixel columns" in caption
    assert "move the raw bytes at 12.5 Gbps" in caption


def test_figure4_marks_a_failed_1_user_run_at_its_median(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure4(mark_failed(runs, "server", "R3", 600, 1), tmp_path)
    (axis,) = saved[0].axes
    heights = [
        height
        for line in axis.get_lines()
        if line.get_marker() == "X"
        for height in line.get_ydata()
    ]
    assert heights == [REQUEST_TIMEOUT_S]


def test_figure4_draws_no_cross_without_a_failed_1_user_run(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure4(runs, tmp_path)
    (axis,) = saved[0].axes
    assert not [
        line
        for line in axis.get_lines()
        if line.get_marker() == "X" and len(line.get_xdata())
    ]


def test_figure4_caption_gives_the_seconds_of_a_failed_request_from_the_header(
    runs, tmp_path
):
    caption = figure4(runs, tmp_path).caption
    assert (
        "The medians count each failed request as at least "
        f"{FAILED_REQUEST_SECONDS} s."
    ) in caption
    assert "every measured request failed" not in caption


def test_figure4_caption_falls_back_to_the_request_timeout(runs, tmp_path):
    runs.attrs.clear()
    caption = figure4(runs, tmp_path).caption
    assert "count each failed request as at least 180 s." in caption


def test_figure4_rings_a_client_limited_run(runs, tmp_path, capture):
    saved = capture(module)
    figure4(mark_client_limited(runs, "server", "R3", 600, 1), tmp_path)
    assert len(ring_points(saved[0].axes)) == 1


def test_figure4_labels_the_ring_client_limited_in_the_legend(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure4(mark_client_limited(runs, "server", "R3", 600, 1, "cpu"), tmp_path)
    labels = [text.get_text() for text in saved[0].legends[0].get_texts()]
    assert "client limited" in labels
    assert not any("link" in label for label in labels)


def test_figure4_draws_no_ring_without_a_client_limited_run(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure4(runs, tmp_path)
    assert ring_points(saved[0].axes) == []


def test_figure4_caption_counts_the_client_limited_runs(runs, tmp_path):
    changed = mark_client_limited(runs, "server", "R3", 600, 1)
    caption = figure4(changed, tmp_path).caption
    assert "A ring marks one client-limited run" in caption
    assert (
        "or the run put more users on a load generator than the cap of 100 "
        "for the server method or 5 for the precomputed method"
    ) in caption
    assert "S3 served" not in caption


def test_figure4_draws_what_exists_when_a_recording_has_no_1_user_run(
    runs, tmp_path, capture
):
    saved = capture(module)
    changed = runs[~((runs["recording"] == "R1") & (runs["users"] == 1))]
    output = figure4(changed, tmp_path)
    assert all(p.stat().st_size > 1000 for p in output.paths)
    (axis,) = saved[0].axes
    assert axis.get_lines()
    assert "for R3," in output.caption


def test_figure4_lists_and_captions_the_cross_only_with_a_failed_run(
    runs, tmp_path, capture
):
    saved = capture(module)
    clean = figure4(runs, tmp_path).caption
    failed = figure4(mark_failed(runs, "server", "R3", 600, 1), tmp_path)
    labels = [
        [text.get_text() for text in figure.legends[0].get_texts()]
        for figure in saved
    ]
    assert "run with failures" not in labels[0]
    assert "run with failures" in labels[1]
    assert "A cross marks" not in clean
    assert "A cross marks a run with a failure rate above 0." in failed.caption
