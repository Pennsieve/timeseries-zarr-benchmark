import pytest

from plot.data import NoDataError
from plot.figure5 import figure5


def test_figure5_writes_pdf_and_png(format_benchmark, tmp_path):
    output = figure5(format_benchmark, tmp_path, "R3-1h", "fig5-test")
    assert [p.name for p in output.paths] == ["fig5-test.pdf", "fig5-test.png"]
    assert all(p.stat().st_size > 1000 for p in output.paths)
    assert output.paths[0].read_bytes().startswith(b"%PDF")
    assert output.paths[1].read_bytes().startswith(b"\x89PNG")


def test_figure5_caption_says_that_s3_served_the_bytes(
    format_benchmark, tmp_path
):
    caption = figure5(format_benchmark, tmp_path).caption
    assert caption.startswith(
        "Figure 5. Median bytes that S3 served per read of R3-1h, on log axes."
    )
    assert "The bytes do not depend on the store." in caption
    assert "band" not in caption
    assert "1-byte" not in caption
    assert caption.isascii()


def test_figure5_caption_counts_the_reads_on_s3(format_benchmark, tmp_path):
    caption = figure5(format_benchmark, tmp_path).caption
    assert "The medians cover 1,134 reads on S3." in caption
    assert "Each combination holds 21 reads." in caption
    assert "the read budget of 1 h" in caption


def test_figure5_raises_for_an_absent_recording(format_benchmark, tmp_path):
    with pytest.raises(NoDataError):
        figure5(format_benchmark, tmp_path, "R4")


def test_figure5_raises_for_a_recording_without_reads_on_s3(
    format_benchmark, tmp_path
):
    without_s3 = format_benchmark[format_benchmark["store"] != "s3"]
    with pytest.raises(NoDataError):
        figure5(without_s3, tmp_path)
