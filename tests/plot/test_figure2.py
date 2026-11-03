import pytest

from plot.data import NoDataError
from plot.figure2 import figure2
from tests.plot.conftest import trim_reads


def test_figure2_writes_pdf_and_png(format_benchmark, tmp_path):
    output = figure2(format_benchmark, tmp_path, "R3-1h", "fig2-test")
    assert [p.name for p in output.paths] == ["fig2-test.pdf", "fig2-test.png"]
    assert all(p.stat().st_size > 1000 for p in output.paths)
    assert output.paths[0].read_bytes().startswith(b"%PDF")
    assert output.paths[1].read_bytes().startswith(b"\x89PNG")


def test_figure2_caption_names_every_store(format_benchmark, tmp_path):
    caption = figure2(format_benchmark, tmp_path).caption
    assert caption.startswith(
        "Figure 2. Median seconds per read of R3-1h on local disk, HTTP, and S3"
    )
    assert "A band spans the 25th to 75th percentile of the reads." in caption
    assert caption.isascii()


def test_figure2_caption_counts_the_reads_per_store(format_benchmark, tmp_path):
    caption = figure2(format_benchmark, tmp_path).caption
    assert (
        "The medians cover 1,134 reads on local disk, 1,134 reads on HTTP, "
        "and 1,134 reads on S3."
    ) in caption
    assert "Each combination holds 21 reads." in caption
    assert "the read budget of 1 h" in caption


def test_figure2_caption_gives_the_fewest_and_most_reads_of_a_combination(
    format_benchmark, tmp_path
):
    caption = figure2(trim_reads(format_benchmark, 5), tmp_path).caption
    assert "Each combination holds between five and 21 reads." in caption
    assert "1,118 reads on local disk" in caption


def test_figure2_caption_counts_only_the_reads_of_the_plotted_recording(
    format_benchmark, tmp_path
):
    trimmed = trim_reads(format_benchmark, 5)
    assert "Each combination holds 21 reads." in (
        figure2(trimmed, tmp_path, "R1-1h").caption
    )


def test_figure2_raises_for_an_absent_recording(format_benchmark, tmp_path):
    with pytest.raises(NoDataError):
        figure2(format_benchmark, tmp_path, "R4")
