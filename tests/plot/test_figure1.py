import pytest

from plot import figure1 as module
from plot.data import NoDataError
from plot.figure1 import figure1, figure1_rows
from tests.plot.conftest import trim_reads
from tszbench.schemas import FORMATS


def test_rows_hold_only_the_10_s_all_channel_reads(format_benchmark):
    rows = figure1_rows(format_benchmark)
    format_rows = rows[rows["format"].isin(FORMATS)]
    # 2 recordings x 3 stores x 6 formats x 21 reads
    assert len(format_rows) == 2 * 3 * 6 * 21
    assert "zarr-direct" not in set(rows["format"])


def test_rows_copy_overhead_into_a_recording_without_it(format_benchmark):
    rows = figure1_rows(format_benchmark)
    for recording in ("R1-1h", "R3-1h"):
        overhead = rows[
            (rows["recording"] == recording) & (rows["format"] == "overhead")
        ]
        assert len(overhead) == 3 * 21


def test_rows_raise_without_matching_reads(format_benchmark):
    with pytest.raises(NoDataError):
        figure1_rows(format_benchmark[format_benchmark["window_s"] == 60])


def test_figure1_writes_pdf_and_png(format_benchmark, tmp_path):
    output = figure1(format_benchmark, tmp_path)
    assert [p.name for p in output.paths] == ["fig1.pdf", "fig1.png"]
    assert all(p.stat().st_size > 1000 for p in output.paths)
    assert output.paths[0].read_bytes().startswith(b"%PDF")
    assert output.paths[1].read_bytes().startswith(b"\x89PNG")


def test_figure1_has_one_panel_per_recording_and_one_row_per_format(
    format_benchmark, tmp_path, capture
):
    saved = capture(module)
    figure1(format_benchmark, tmp_path)
    axes = saved[0].axes
    assert [axis.get_title() for axis in axes] == ["R1-1h", "R3-1h"]
    by_position = dict(
        zip(
            axes[0].get_yticks(),
            [t.get_text() for t in axes[0].get_yticklabels()],
            strict=True,
        )
    )
    assert len(by_position) == 7
    assert by_position[max(by_position)] == "EDF"
    assert by_position[min(by_position)] == "overhead (1 byte)"
    assert axes[0].get_xscale() == "log"


def test_figure1_caption_counts_reads_per_store(format_benchmark, tmp_path):
    caption = figure1(format_benchmark, tmp_path).caption
    assert caption.startswith("Figure 1.")
    assert (
        "R3-1h: 147 reads on local disk, 147 reads on HTTP, 147 reads on S3."
        in caption
    )
    assert caption.isascii()


def test_figure1_caption_states_the_reads_per_combination_and_the_budget(
    format_benchmark, tmp_path
):
    caption = figure1(format_benchmark, tmp_path).caption
    assert (
        " Each combination holds 21 reads. A combination stops after the read "
        "that brings the sum of its read seconds to at least the read budget "
        "of 1 h, once it has made at least three reads."
    ) in caption


def test_figure1_caption_leaves_out_the_budget_without_header_keys(
    format_benchmark, tmp_path
):
    format_benchmark.attrs.clear()
    caption = figure1(format_benchmark, tmp_path).caption
    assert "Each combination holds 21 reads." in caption
    assert "budget" not in caption


def test_figure1_caption_counts_only_the_reads_that_it_draws(
    format_benchmark, tmp_path
):
    trimmed = trim_reads(format_benchmark, 5)
    first = trimmed.iloc[0]
    assert first["window_s"] != 10
    caption = figure1(trimmed, tmp_path).caption
    assert "Each combination holds 21 reads." in caption
