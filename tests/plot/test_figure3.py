import pytest

from plot import figure3 as module
from plot.data import NoDataError
from plot.figure3 import figure3, figure3_series
from resampling_benchmark.load.generator import REQUEST_TIMEOUT_S
from tests.plot.conftest import (
    FAILED_REQUEST_SECONDS,
    mark_client_limited,
    mark_not_run,
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


def test_series_leaves_out_a_run_that_was_not_run(runs):
    changed = mark_not_run(runs, "server", "R3", 600, 1000)
    series = figure3_series(changed, "R3")
    assert len(series) == 2 * 2 * 4 - 1
    left = series[
        (series["method"] == "server")
        & (series["window_s"] == 600)
        & (series["users"] == 1000)
    ]
    assert left.empty


def test_figure3_draws_without_the_runs_that_were_not_run(
    runs, tmp_path, capture
):
    saved = capture(module)
    changed = mark_not_run(runs, "server", "R3", 600, 1000)
    changed = mark_not_run(changed, "precomputed", "R3", 600, 1000)
    output = figure3(changed, tmp_path)
    assert all(p.stat().st_size > 1000 for p in output.paths)
    assert len(saved[0].axes) == 4


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


def test_figure3_rings_a_client_limited_run_on_both_rows(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure3(mark_client_limited(runs, "precomputed", "R3", 600, 1000), tmp_path)
    assert len(ring_points(saved[0].axes)) == 2


def test_figure3_labels_the_ring_client_limited_in_the_legend(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure3(
        mark_client_limited(runs, "precomputed", "R3", 600, 1000, "cpu"),
        tmp_path,
    )
    legend = saved[0].legends[0]
    labels = [text.get_text() for text in legend.get_texts()]
    assert "client limited" in labels
    assert not any("link" in label for label in labels)
    saved.clear()
    figure3(runs, tmp_path)
    labels = [text.get_text() for text in saved[0].legends[0].get_texts()]
    assert "client limited" not in labels


def test_figure3_draws_no_ring_without_a_client_limited_run(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure3(runs, tmp_path)
    assert ring_points(saved[0].axes) == []


def test_figure3_caption_counts_the_client_limited_runs(runs, tmp_path):
    changed = mark_client_limited(runs, "precomputed", "R3", 600, 1000)
    caption = figure3(changed, tmp_path).caption
    assert "A ring marks one client-limited run, in which" in caption
    assert "ring" not in figure3(runs, tmp_path).caption


def test_figure3_caption_states_the_cap_of_users_per_generator(runs, tmp_path):
    changed = mark_client_limited(runs, "precomputed", "R3", 600, 1000, "cpu")
    caption = figure3(changed, tmp_path).caption
    assert (
        "passed half its capacity, or the run put more users on a load "
        "generator than the cap of 100 for the server method or 5 for the "
        "precomputed method, and no repeat with fewer users per generator "
        "replaced the run."
    ) in caption


def test_figure3_caption_states_the_s3_rates_of_the_client_limited_runs(
    runs, tmp_path
):
    changed = mark_client_limited(runs, "precomputed", "R3", 600, 1000)
    mask = (
        (changed["method"] == "precomputed")
        & (changed["window_s"] == 600)
        & (changed["users"] == 1000)
        & (changed["repeated"] == "false")
    )
    changed.loc[mask, "s3_requests_per_second"] = 107_000.0
    changed.loc[mask, "s3_bytes_per_second"] = 3.7e9
    caption = figure3(changed, tmp_path).caption
    assert (
        "S3 served the client-limited precomputed runs of 1,000 users at "
        "107,000 requests per second and 3.7 GB per second (10 min window)."
    ) in caption
    assert "S3 served" not in figure3(runs, tmp_path).caption


def test_figure3_states_no_s3_rate_for_a_client_limited_server_run(
    runs, tmp_path
):
    changed = mark_client_limited(runs, "server", "R3", 600, 1000)
    assert "S3 served" not in figure3(changed, tmp_path).caption


def _dotted(axis, color):
    return [
        line
        for line in axis.get_lines()
        if line.get_linestyle() == ":" and line.get_color() == color
    ]


def test_figure3_draws_both_rows_on_log_y_axes(runs, tmp_path, capture):
    saved = capture(module)
    figure3(runs, tmp_path)
    assert [axis.get_yscale() for axis in saved[0].axes] == ["log"] * 4


def test_figure3_labels_log_ticks_as_plain_numbers(runs, tmp_path, capture):
    saved = capture(module)
    figure3(runs, tmp_path)
    axis = saved[0].axes[2]
    formatter = axis.yaxis.get_major_formatter()
    assert [formatter(value) for value in (0.01, 0.1, 1, 10, 1000)] == [
        "0.01",
        "0.1",
        "1",
        "10",
        "1,000",
    ]


def test_figure3_draws_the_timeout_in_every_upper_axis(runs, tmp_path, capture):
    saved = capture(module)
    figure3(runs, tmp_path)
    for axis in saved[0].axes[:2]:
        lines = _dotted(axis, module.GUIDE_COLOR)
        assert len(lines) == 1
        assert set(lines[0].get_ydata()) == {FAILED_REQUEST_SECONDS}


def test_figure3_writes_the_timeout_text_in_the_first_column_only(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure3(runs, tmp_path)
    texts = [[t.get_text() for t in axis.texts] for axis in saved[0].axes]
    assert texts == [["timeout"], [], [], []]


def test_figure3_draws_the_timeout_at_the_request_timeout_without_a_header(
    runs, tmp_path, capture
):
    saved = capture(module)
    runs.attrs.clear()
    figure3(runs, tmp_path)
    line = _dotted(saved[0].axes[0], module.GUIDE_COLOR)[0]
    assert set(line.get_ydata()) == {REQUEST_TIMEOUT_S}


def test_figure3_draws_the_reference_rate_from_the_one_user_run(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure3(runs, tmp_path)
    rate_10s = saved[0].axes[2]
    server = _dotted(rate_10s, module.METHOD_COLORS["server"])
    assert len(server) == 1
    assert list(server[0].get_xdata()) == [1, 10, 100, 1000]
    # The server median at 1 user is window / 100, so 10 s gives 0.1 s.
    assert list(server[0].get_ydata()) == pytest.approx(
        [users / 0.1 for users in (1, 10, 100, 1000)]
    )
    assert server[0].get_alpha() == 0.5


def test_figure3_draws_one_reference_line_per_method_in_the_method_color(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure3(runs, tmp_path)
    for method, color in module.METHOD_COLORS.items():
        assert len(_dotted(saved[0].axes[3], color)) == 1, method


def test_figure3_draws_no_reference_line_without_a_one_user_run(
    runs, tmp_path, capture
):
    saved = capture(module)
    changed = runs[~((runs["method"] == "server") & (runs["users"] == 1))]
    figure3(changed.reset_index(drop=True), tmp_path)
    assert _dotted(saved[0].axes[2], module.METHOD_COLORS["server"]) == []
    assert len(_dotted(saved[0].axes[2], module.METHOD_COLORS["precomputed"]))


def test_figure3_labels_the_reference_lines_no_queueing_in_the_legend(
    runs, tmp_path, capture
):
    saved = capture(module)
    figure3(runs, tmp_path)
    labels = [text.get_text() for text in saved[0].legends[0].get_texts()]
    assert labels.count("no queueing") == 1


def test_figure3_caption_defines_a_request(runs, tmp_path):
    caption = figure3(runs, tmp_path).caption
    assert caption.startswith("Figure 3. A request asks for the minimum and ")
    assert "reduced to 2,000 pixel columns" in caption
    assert "returns them as one array" in caption
    assert "the median seconds of the 1-user run" in caption
    assert "dotted gray line in the upper panels is the timeout" in caption
