import pytest

from plot.data import NoDataError
from plot.methods import (
    budget_paragraph,
    check_paragraph,
    generation_paragraph,
    open_paragraph,
    upload_paragraph,
    versions_paragraph,
    write_methods,
)
from tests.plot.conftest import REPEATS, STOPPED_READS, write_check, write_open


def test_upload_paragraph_gives_the_median_rate_of_each_format(results_dir):
    text = upload_paragraph(results_dir)
    # The rates are 50, 100, and 150 MB/s for EDF and 20, 40, and 60 MB/s for
    # TS-Zarr.
    assert text == (
        "The median upload rate of `aws s3 cp` at default settings is "
        "100 MB/s for EDF and 40 MB/s for TS-Zarr."
    )


def test_open_paragraph_gives_median_seconds_and_requests_per_file_format(
    results_dir,
):
    text = open_paragraph(results_dir)
    assert text == (
        "The open of TS-Zarr takes a median of 0.2 s and 2 requests on "
        "local disk, 0.4 s and 2 requests on HTTP, and 0.6 s and 2 requests "
        "on S3."
    )


def test_budget_paragraph_counts_the_combinations_that_stopped(results_dir):
    assert budget_paragraph(results_dir) == (
        "Of 12 combinations, three stopped at the read budget of 1 h, each "
        "after at least seven reads."
    )


def test_budget_paragraph_says_so_when_no_combination_stopped(results_dir):
    write_open(results_dir / "format-opens.csv", stopped=0)
    assert budget_paragraph(results_dir) == (
        "No combination reached the read budget of 1 h."
    )


def test_budget_paragraph_uses_the_fewest_reads_of_the_stopped_combinations(
    results_dir,
):
    path = results_dir / "format-opens.csv"
    lines = path.read_text().splitlines(keepends=True)
    first = next(i for i, line in enumerate(lines) if "tszarr,one" in line)
    lines[first] = lines[first].replace(f",{STOPPED_READS},", ",4,")
    path.write_text("".join(lines))
    assert "each after at least four reads." in budget_paragraph(results_dir)


def test_budget_paragraph_writes_a_minute_budget_in_minutes(results_dir):
    path = results_dir / "format-opens.csv"
    path.write_text(
        path.read_text().replace("read_budget_s: 3600", "read_budget_s: 1800")
    )
    assert "read budget of 30 min," in budget_paragraph(results_dir)


def test_budget_paragraph_leaves_out_the_budget_without_the_header_key(
    results_dir,
):
    write_open(results_dir / "format-opens.csv", header=False)
    assert "stopped at the read budget, each after" in budget_paragraph(
        results_dir
    )
    write_open(results_dir / "format-opens.csv", stopped=0, header=False)
    assert budget_paragraph(results_dir) == (
        "No combination reached the read budget."
    )


def test_budget_paragraph_uses_the_singular_for_one_stopped_combination(
    results_dir,
):
    path = results_dir / "format-opens.csv"
    lines = path.read_text().splitlines(keepends=True)
    kept = 0
    for index, line in enumerate(lines):
        if f",{STOPPED_READS}," in line:
            kept += 1
            if kept > 1:
                lines[index] = line.replace(
                    f",{STOPPED_READS},", f",{REPEATS},"
                )
    path.write_text("".join(lines))
    assert budget_paragraph(results_dir) == (
        "Of 12 combinations, one stopped at the read budget of 1 h, after "
        "seven reads."
    )


def test_budget_paragraph_raises_when_the_file_is_absent(tmp_path):
    with pytest.raises(NoDataError):
        budget_paragraph(tmp_path)


def test_generation_paragraph_follows_the_recording_order(results_dir):
    text = generation_paragraph(results_dir)
    assert text == (
        "Generating R3-1h takes 10 s for EDF and 20 s for TS-Zarr. "
        "Generating R3 takes 240 s for EDF and 480 s for TS-Zarr."
    )


def test_check_paragraph_counts_equal_windows(results_dir):
    assert check_paragraph(results_dir) == (
        "The server and the precomputed read return equal arrays in three "
        "of three windows, on R3."
    )
    write_check(results_dir / "resampling-check.csv", ("true", "false", "true"))
    assert "in two of three windows" in check_paragraph(results_dir)


def test_versions_paragraph_reads_the_csv_headers(results_dir):
    text = versions_paragraph(results_dir)
    assert text.startswith(
        "The CSV headers record timeseries-zarr 0.1.0a7 and tsdownsample "
        "0.1.5.1."
    )
    assert "pynwb" not in text
    assert "c6i.2xlarge for format-opens.csv" in text
    assert "c6i.xlarge for resampling-check.csv" in text
    assert "generation-R3-1h.csv" in text


def test_versions_paragraph_lists_every_version_the_files_disagree_on(
    results_dir,
):
    path = results_dir / "format-opens.csv"
    path.write_text(
        path.read_text().replace(
            "version.tsdownsample: 0.1.5.1", "version.tsdownsample: 0.1.5.0"
        )
    )
    assert "tsdownsample (0.1.5.0, 0.1.5.1)" in versions_paragraph(results_dir)


def test_write_methods_writes_one_paragraph_per_statement(
    results_dir, tmp_path
):
    messages: list[str] = []
    output = write_methods(results_dir, tmp_path / "out", messages.append)
    assert [p.name for p in output.paths] == ["methods.txt"]
    assert output.caption == ""
    text = output.paths[0].read_text()
    assert len(text.strip().split("\n\n")) == 6
    assert text.isascii()
    assert messages == []


def test_write_methods_skips_a_paragraph_whose_csv_is_absent(
    results_dir, tmp_path
):
    (results_dir / "format-opens.csv").unlink()
    (results_dir / "resampling-check.csv").unlink()
    messages: list[str] = []
    output = write_methods(results_dir, tmp_path, messages.append)
    assert len(output.paths[0].read_text().strip().split("\n\n")) == 3
    assert messages == [
        "Methods open sentence: format-opens.csv not found",
        "Methods read budget sentence: format-opens.csv not found",
        "Methods equality check sentence: resampling-check.csv not found",
    ]


def test_write_methods_raises_without_any_data(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    messages: list[str] = []
    with pytest.raises(NoDataError):
        write_methods(empty, tmp_path / "out", messages.append)
    assert len(messages) == 6
    assert not (tmp_path / "out" / "methods.txt").exists()
