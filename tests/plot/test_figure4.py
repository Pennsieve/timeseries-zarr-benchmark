import pytest

from plot import figure4 as module
from plot.data import NoDataError
from plot.figure4 import figure4, figure4_series
from resampling_benchmark.load.generator import REQUEST_TIMEOUT_S
from tests.plot.conftest import (
    FAILED_REQUEST_SECONDS,
    mark_link_limited,
    ring_points,
)


def test_series_keeps_users_1_and_100_of_every_recording(runs):
    series = figure4_series(runs)
    assert set(series["users"]) == {1, 100}
    assert set(series["recording"]) == {"R1", "R3"}
    assert len(series) == 2 * 2 * 2 * 2


def test_series_is_sorted_by_raw_bytes_within_a_line(runs):
    series = figure4_series(runs)
    line = series[(series["method"] == "server") & (series["users"] == 1)]
    assert list(line["raw_bytes"]) == sorted(line["raw_bytes"])


def test_series_raises_without_matching_runs(runs):
    with pytest.raises(NoDataError):
        figure4_series(runs[runs["users"] == 10])


def test_figure4_draws_one_line_per_method_and_user_count(
    runs, tmp_path, capture
):
    saved = capture(module)
    output = figure4(runs, tmp_path)
    assert all(p.stat().st_size > 1000 for p in output.paths)
    (axis,) = saved[0].axes
    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"
    styles = {
        line.get_linestyle()
        for line in axis.get_lines()
        if line.get_linestyle() != "None"
    }
    assert styles == {"-", "--"}


def test_figure4_caption_counts_requests(runs, tmp_path):
    caption = figure4(runs, tmp_path).caption
    assert "for R1 and R3" in caption
    assert "720 requests at one user, 675 requests at 100 users" in caption
    assert caption.isascii()


def test_figure4_marks_a_run_in_which_every_request_failed_at_its_median(
    runs, tmp_path, capture
):
    saved = capture(module)
    changed = runs.copy()
    mask = (
        (changed["method"] == "server")
        & (changed["recording"] == "R3")
        & (changed["window_s"] == 600)
        & (changed["users"] == 100)
    )
    changed.loc[mask, ["median_s", "p5_s", "p95_s"]] = REQUEST_TIMEOUT_S
    changed.loc[mask, "failure_rate"] = 1.0
    figure4(changed, tmp_path)
    (axis,) = saved[0].axes
    heights = [
        height
        for line in axis.get_lines()
        if line.get_marker() == "X"
        for height in line.get_ydata()
    ]
    assert REQUEST_TIMEOUT_S in heights


def test_figure4_marks_every_run_with_failures_at_its_median(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure4(runs, tmp_path)
    (axis,) = saved[0].axes
    heights = sorted(
        height
        for line in axis.get_lines()
        if line.get_marker() == "X"
        for height in line.get_ydata()
    )
    # The fixture server runs with failures at 1 or 100 users: R3 at 600 s.
    assert heights == [600 * 100 / 100]


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


def test_figure4_rings_a_link_limited_run(runs, tmp_path, capture):
    saved = capture(module)
    figure4(mark_link_limited(runs, "server", "R3", 600, 100), tmp_path)
    assert len(ring_points(saved[0].axes)) == 1


def test_figure4_draws_no_ring_without_a_link_limited_run(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure4(runs, tmp_path)
    assert ring_points(saved[0].axes) == []


def test_figure4_caption_counts_the_link_limited_runs(runs, tmp_path):
    changed = mark_link_limited(runs, "server", "R3", 600, 100)
    assert "A ring marks one run" in figure4(changed, tmp_path).caption
