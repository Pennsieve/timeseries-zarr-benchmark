import numpy as np
import pytest

from plot.style import LABEL_GAP, label_lines, new_figure, spread_labels


def test_spread_keeps_labels_that_do_not_overlap():
    assert spread_labels([0.1, 0.5, 0.9], 0.1, 0.0, 1.0) == [0.1, 0.5, 0.9]


def test_spread_centers_two_labels_at_one_position_on_it():
    centers = spread_labels([0.5, 0.5], 0.1, 0.0, 1.0)
    assert sorted(centers) == pytest.approx([0.45, 0.55])


def test_spread_keeps_the_order_of_the_positions():
    centers = spread_labels([0.52, 0.5, 0.51], 0.1, 0.0, 1.0)
    assert centers[1] < centers[2] < centers[0]
    assert np.diff(sorted(centers)).min() == pytest.approx(0.1)


def test_spread_moves_a_group_inside_the_bounds():
    centers = spread_labels([0.99, 1.0, 1.0], 0.1, 0.0, 1.0)
    assert max(centers) == pytest.approx(1.0)
    assert np.diff(sorted(centers)).min() == pytest.approx(0.1)


def test_spread_merges_a_group_that_a_bound_pushes_into_a_neighbor():
    centers = spread_labels([0.82, 1.0, 1.0], 0.1, 0.0, 1.0)
    assert np.diff(sorted(centers)).min() >= 0.1 - 1e-9
    assert max(centers) <= 1.0 + 1e-9


def test_spread_centers_labels_that_do_not_fit():
    centers = spread_labels([0.5] * 5, 0.5, 0.0, 1.0)
    assert np.mean(centers) == pytest.approx(0.5)


def test_label_lines_writes_one_label_per_line_in_its_color():
    figure = new_figure(3, 2)
    axis = figure.subplots()
    low = axis.plot([1, 2], [1, 2], color="#000000")[0]
    high = axis.plot([1, 2], [1, 8], color="#0072B2")[0]
    texts = label_lines(axis, [low, high], ["low", "high"])
    assert [t.get_text() for t in texts] == ["low", "high"]
    assert texts[1].get_color() == "#0072B2"
    assert texts[0].xy[1] < texts[1].xy[1]


def test_label_lines_separates_labels_of_lines_that_end_together():
    figure = new_figure(3, 2)
    axis = figure.subplots()
    axis.set_yscale("log")
    lines = [
        axis.plot([1, 2], [1, end])[0] for end in (10.0, 10.1, 10.2, np.nan)
    ]
    lines[3].set_ydata([3.0, np.nan])
    floor = axis.axhline(1.0)
    axis.set_ylim(0.5, 20)
    texts = label_lines(axis, [*lines, floor], ["a", "b", "c", "d", "e"])
    figure.draw_without_rendering()
    boxes = sorted(
        (t.get_window_extent() for t in texts[:3]), key=lambda box: box.y0
    )
    for below, above in zip(boxes, boxes[1:], strict=False):
        assert above.y0 >= below.y1 - 0.5
    assert texts[4].xy[1] == pytest.approx(
        (np.log10(1.0) - np.log10(0.5)) / (np.log10(20) - np.log10(0.5))
    )
    assert LABEL_GAP > 1


def test_label_lines_places_a_line_ending_in_nan_at_its_last_finite_point():
    figure = new_figure(3, 2)
    axis = figure.subplots()
    line = axis.plot([1, 2, 3], [0.0, 5.0, np.nan])[0]
    axis.set_ylim(0, 10)
    (text,) = label_lines(axis, [line], ["a"])
    assert text.xy[1] == pytest.approx(0.5)
