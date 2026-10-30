import pytest

from plot import figure3 as module
from plot.data import NoDataError
from plot.figure3 import figure3, figure3_series
from resampling_benchmark.load.generator import REQUEST_TIMEOUT_S
from tests.plot.conftest import (
    FAILED_REQUEST_SECONDS,
    mark_link_limited,
    ring_points,
)


def test_series_has_one_run_per_method_window_and_users(runs):
    series = figure3_series(runs, "R3")
    assert len(series) == 2 * 2 * 4
    assert list(series["window_s"])[:8] == [10] * 8


def test_series_uses_the_repeated_run(runs):
    series = figure3_series(runs, "R3")
    row = series[
        (series["method"] == "server")
        & (series["window_s"] == 10)
        & (series["users"] == 100)
    ]
    assert row["requests_per_second"].iloc[0] == 5.0


def test_series_raises_for_an_absent_recording(runs):
    with pytest.raises(NoDataError):
        figure3_series(runs, "R4")


def test_figure3_marks_runs_with_failures(runs, tmp_path, capture):
    saved = capture(module)
    output = figure3(runs, tmp_path)
    assert all(p.stat().st_size > 1000 for p in output.paths)
    axes = saved[0].axes
    assert len(axes) == 4
    failed = [
        len(line.get_xdata())
        for axis in axes
        for line in axis.get_lines()
        if line.get_marker() == "X"
    ]
    # Server failures: 1 user count at 10 s, 2 at 10 min, in two rows each.
    assert sum(failed) == 2 * (1 + 2)


def test_figure3_draws_the_failure_marker_at_the_median(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure3(runs, tmp_path)
    # The server median is window * users / 100 in the fixture.
    expected = {10 * 1000 / 100, 600 * 100 / 100, 600 * 1000 / 100}
    for column in (0, 1):
        seconds_axis = saved[0].axes[column]
        heights = {
            height
            for line in seconds_axis.get_lines()
            if line.get_marker() == "X"
            for height in line.get_ydata()
        }
        assert heights <= expected
    marked = {
        height
        for line in saved[0].axes[1].get_lines()
        if line.get_marker() == "X"
        for height in line.get_ydata()
    }
    assert marked == {600 * 100 / 100, 600 * 1000 / 100}


def test_figure3_draws_a_percentile_band_per_method(runs, tmp_path, capture):
    saved = capture(module)
    figure3(runs, tmp_path)
    top = saved[0].axes[0]
    assert top.get_yscale() == "log"
    assert len(top.collections) == 2


def test_figure3_caption_counts_requests(runs, tmp_path):
    caption = figure3(runs, tmp_path).caption
    # Server: 8 runs of 90 requests, minus the replaced run of 90, plus 45.
    assert "675 requests by the server method" in caption
    assert "720 requests by the precomputed method" in caption
    assert caption.isascii()


def _all_failed(runs, method, window, users):
    changed = runs.copy()
    mask = (
        (changed["method"] == method)
        & (changed["recording"] == "R3")
        & (changed["window_s"] == window)
        & (changed["users"] == users)
        & (changed["repeated"] == "false")
    )
    changed.loc[mask, ["median_s", "p5_s", "p95_s"]] = REQUEST_TIMEOUT_S
    changed.loc[mask, "failure_rate"] = 1.0
    return changed


def test_figure3_marks_a_run_in_which_every_request_failed_at_its_median(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure3(_all_failed(runs, "server", 600, 1000), tmp_path)
    seconds_axis = saved[0].axes[1]
    heights = [
        height
        for line in seconds_axis.get_lines()
        if line.get_marker() == "X"
        for height in line.get_ydata()
    ]
    assert REQUEST_TIMEOUT_S in heights


def test_figure3_caption_gives_the_seconds_of_a_failed_request_from_the_header(
    runs, tmp_path
):
    caption = figure3(runs, tmp_path).caption
    assert (
        f"The medians and percentiles count each failed request as at "
        f"least {FAILED_REQUEST_SECONDS} s."
    ) in caption
    assert "every measured request failed" not in caption


def test_figure3_caption_falls_back_to_the_request_timeout(runs, tmp_path):
    runs.attrs.clear()
    caption = figure3(runs, tmp_path).caption
    assert "count each failed request as at least 180 s." in caption


def test_figure3_rings_a_link_limited_run_on_both_rows(runs, tmp_path, capture):
    saved = capture(module)
    figure3(mark_link_limited(runs, "precomputed", "R3", 600, 1000), tmp_path)
    assert len(ring_points(saved[0].axes)) == 2


def test_figure3_draws_no_ring_without_a_link_limited_run(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure3(runs, tmp_path)
    assert ring_points(saved[0].axes) == []


def test_figure3_caption_counts_the_link_limited_runs(runs, tmp_path):
    changed = mark_link_limited(runs, "precomputed", "R3", 600, 1000)
    caption = figure3(changed, tmp_path).caption
    assert "A ring marks one run in which a load generator" in caption
    assert "ring" not in figure3(runs, tmp_path).caption
