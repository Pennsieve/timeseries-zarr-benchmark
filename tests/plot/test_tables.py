import pandas as pd
import pytest

from plot.data import NoDataError
from plot.tables import (
    NOT_RUN,
    latex,
    table1,
    table2,
    table3,
    table4,
    write_table,
)
from tests.plot.conftest import (
    mark_client_limited,
    mark_not_run,
    trim_reads,
)
from tszbench.recordings import RECORDINGS


def test_table1_lists_every_recording():
    table = table1()
    assert list(table.frame["id"]) == ["R1", "R2", "R3", "R4"]
    r3 = table.rows[2]
    assert r3[:2] == ("R3", "intracranial EEG")
    assert r3[2:] == ("200", "2000 Hz", "24 h", "172,800,000", "138 GB", "8")
    assert table.rows[3][3] == "30 kHz"
    assert table.rows[3][4] == "10 min"
    assert table.frame["raw_bytes"].iloc[2] == RECORDINGS["R3"].raw_bytes


def test_table2_multiplier_is_bytes_over_raw_float32_bytes(storage):
    frame = table2(storage).frame.set_index(["format", "member"])
    raw = RECORDINGS["R3"].raw_bytes
    assert frame.loc[("edf", ""), "multiplier"] == pytest.approx(0.5)
    assert frame.loc[("nwb-hdf5-time", ""), "multiplier"] == pytest.approx(1.0)
    assert frame.loc[("tszarr", "raw"), "bytes"] == raw
    assert frame.loc[("tszarr", "mean"), "multiplier"] == pytest.approx(0.25)


def test_table2_adds_a_raw_plus_minmax_row_with_the_paper_multiplier(storage):
    frame = table2(storage).frame.set_index(["format", "member"])
    row = frame.loc[("tszarr", "raw+minmax")]
    assert row["bytes"] == 1.5 * RECORDINGS["R3"].raw_bytes
    assert row["objects"] == 150
    assert row["multiplier"] == pytest.approx(1.5)
    assert row["paper"] == pytest.approx(5 / 3)


def test_table2_paper_multiplier_is_one_third_for_further_members(storage):
    frame = table2(storage).frame.set_index(["format", "member"])
    for member in ("mean", "var", "count"):
        assert frame.loc[("tszarr", member), "paper"] == pytest.approx(1 / 3)
    assert pd.isna(frame.loc[("edf", ""), "paper"])
    assert pd.isna(frame.loc[("tszarr", "raw"), "paper"])


def test_table2_orders_formats_then_members(storage):
    frame = table2(storage).frame
    assert list(frame["format"]) == (
        ["edf", "nwb-hdf5-time"] + ["tszarr"] * 8 + ["tszarr-raw"]
    )
    assert list(frame["member"])[2:] == [
        "",
        "raw",
        "minmax",
        "raw+minmax",
        "mean",
        "var",
        "count",
        "metadata",
        "",
    ]


def test_table2_shows_the_raw_only_bundle_with_its_label(storage):
    table = table2(storage)
    row = next(r for r in table.rows if r[1] == "TS-Zarr, raw only")
    assert row[0] == "R3"
    assert row[2] == ""


def test_table2_keeps_the_last_of_a_repeated_row(storage):
    doubled = pd.concat([storage, storage.iloc[[0]].assign(bytes=1.0)])
    frame = table2(doubled).frame.set_index(["format", "member"])
    assert frame.loc[("edf", ""), "bytes"] == 1.0


def test_table2_raises_without_rows(storage):
    with pytest.raises(NoDataError):
        table2(storage.iloc[0:0])


def test_table3_divides_server_seconds_by_precomputed_seconds(runs):
    frame = table3(runs).frame.set_index("window_s")
    # Server median is window * users / 100. Precomputed is
    # 0.5 * sqrt(window) * users ** 0.25.
    assert frame.loc[10, "users_1"] == pytest.approx(0.1 / (0.5 * 10**0.5))
    assert frame.loc[10, "users_10"] == pytest.approx(
        1.0 / (0.5 * 10**0.5 * 10**0.25)
    )
    assert frame.loc[600, "users_1"] == pytest.approx(6.0 / (0.5 * 600**0.5))


def test_table3_uses_the_repeated_run(runs):
    frame = table3(runs).frame.set_index("window_s")
    assert frame.loc[10, "users_100"] == pytest.approx(
        7.0 / (0.5 * 10**0.5 * 100**0.25)
    )


def test_table3_marks_failed_runs_with_fail(runs):
    table = table3(runs)
    frame = table.frame.set_index("window_s")
    assert frame.loc[10, "users_1000"] == "fail"
    assert frame.loc[600, "users_100"] == "fail"
    assert frame.loc[600, "users_1000"] == "fail"
    assert table.rows[0][-1] == "fail"
    assert table.rows[1][3] == "fail"


def test_table3_marks_a_failed_precomputed_run_apart_from_a_failed_server(
    runs,
):
    changed = runs.copy()
    mask = (
        (changed["method"] == "precomputed")
        & (changed["recording"] == "R3")
        & (changed["window_s"] == 10)
        & (changed["users"] == 1)
    )
    changed.loc[mask, "failure_rate"] = 0.5
    table = table3(changed)
    assert table.frame["users_1"].iloc[0] == "fail (precomputed)"
    assert table.rows[0][1] == "fail (precomputed)"
    assert table.frame["users_1000"].iloc[0] == "fail"
    assert "fail (precomputed) where only the precomputed run" in table.caption


def test_table3_reads_fail_where_both_runs_failed(runs):
    changed = runs.copy()
    mask = (
        (changed["recording"] == "R3")
        & (changed["window_s"] == 10)
        & (changed["users"] == 1)
    )
    changed.loc[mask, "failure_rate"] = 0.5
    assert table3(changed).frame["users_1"].iloc[0] == "fail"


def test_table3_reads_not_run_where_a_run_was_not_run(runs):
    changed = mark_not_run(runs, "server", "R3", 600, 1000)
    changed = mark_not_run(changed, "precomputed", "R3", 10, 100)
    table = table3(changed)
    assert table.frame["users_1000"].iloc[1] == NOT_RUN
    assert table.rows[1][-1] == "not run"
    assert table.frame["users_100"].iloc[0] == NOT_RUN
    assert "not run where a run with fewer users" in table.caption
    assert "not run" not in table3(runs).caption


def test_table3_reads_not_run_before_fail(runs):
    # The server run at 10 s and 1000 users has failures.
    changed = mark_not_run(runs, "precomputed", "R3", 10, 1000)
    assert table3(changed).frame["users_1000"].iloc[0] == NOT_RUN


def test_table3_counts_no_requests_of_a_run_that_was_not_run(runs):
    changed = mark_not_run(runs, "server", "R3", 600, 1000)
    assert "1,395 requests" in table3(runs).caption
    assert "1,215 requests" in table3(changed).caption


def test_table3_marks_a_client_limited_run_in_the_cell_and_the_frame(runs):
    changed = mark_client_limited(runs, "precomputed", "R3", 600, 10, "cpu")
    table = table3(changed)
    assert table.rows[1][2].endswith(" (client)")
    assert not table.rows[1][1].endswith("(client)")
    frame = table.frame.set_index("window_s")
    assert bool(frame.loc[600, "users_10_client"]) is True
    assert bool(frame.loc[600, "users_1_client"]) is False
    assert "A cell ends with (client) where, in a run, " in table.caption
    assert (
        "or the run put more users on a load generator than the cap of 100 "
        "for the server method or 12 for the precomputed method"
    ) in table.caption


def test_table3_has_no_client_mark_without_a_client_limited_run(runs):
    table = table3(runs)
    assert not any(
        cell.endswith("(client)") for row in table.rows for cell in row
    )
    assert not table.frame.filter(like="_client").to_numpy().any()
    assert "(client)" not in table.caption


def test_table3_leaves_a_missing_method_empty(runs):
    only_server = runs[
        ~(
            (runs["method"] == "precomputed")
            & (runs["window_s"] == 600)
            & (runs["users"] == 10)
        )
    ]
    table = table3(only_server)
    assert table.rows[1][2] == ""


def test_table3_header_and_caption_count_requests(runs):
    table = table3(runs)
    assert table.headers == (
        "window",
        "1 users",
        "10 users",
        "100 users",
        "1,000 users",
    )
    assert "on R3" in table.caption
    assert "1,395 requests" in table.caption
    assert "failed request" not in table.caption


def test_table3_raises_for_an_absent_recording(runs):
    with pytest.raises(NoDataError):
        table3(runs, "R4")


def test_write_table_writes_csv_and_tex(storage, tmp_path):
    output = write_table(table2(storage), tmp_path)
    assert [p.name for p in output.paths] == ["table2.csv", "table2.tex"]
    csv = pd.read_csv(output.paths[0])
    assert len(csv) == 11
    tex = output.paths[1].read_text()
    assert tex.startswith("\\begin{tabular}{lllrrrr}\n\\toprule\n")
    assert "\\midrule" in tex
    assert tex.endswith("\\bottomrule\n\\end{tabular}\n")
    assert "raw+minmax" in tex
    assert tex.isascii()


def test_write_table_csv_keeps_fail_cells(runs, tmp_path):
    output = write_table(table3(runs), tmp_path)
    csv = pd.read_csv(output.paths[0], dtype=str)
    assert csv["users_1000"].tolist() == ["fail", "fail"]


def test_latex_escapes_special_characters():
    table = table1()
    broken = table.__class__(
        name="t",
        frame=table.frame,
        headers=("a_b", "50%"),
        rows=(("x & y", "#1"),),
        alignment="ll",
        caption="",
    )
    text = latex(broken)
    assert "a\\_b & 50\\%" in text
    assert "x \\& y & \\#1" in text


def test_table4_divides_tszarr_seconds_by_zarr_direct_seconds(format_benchmark):
    frame = table4(format_benchmark).frame.set_index(
        ["recording", "store", "read_kind", "window_s"]
    )
    row = frame.loc[("R3-1h", "s3", "one", 10)]
    # The synthetic seconds scale with the format index: 6 for tszarr, 7 for
    # zarr-direct. The two share the spread of the repeats.
    assert row["ratio"] == pytest.approx(6 / 7)
    assert row["tszarr_s"] / row["zarr_direct_s"] == pytest.approx(6 / 7)


def test_table4_has_one_row_per_recording_store_read_kind_and_window(
    format_benchmark,
):
    table = table4(format_benchmark)
    assert len(table.frame) == 2 * 3 * 2 * 3 + 3 * 3
    assert list(table.frame["recording"])[:1] == ["R1-1h"]
    assert table.rows[0][:4] == ("R1-1h", "local disk", "one channel", "1 s")
    assert len(table.rows[0]) == len(table.headers)


def test_table4_labels_and_orders_the_subset_read_kinds(format_benchmark):
    table = table4(format_benchmark)
    labels = [
        row[2]
        for row in table.rows
        if row[:2] == ("R3-1h", "local disk") and row[3] == "10 s"
    ]
    assert labels == [
        "one channel",
        "4 channels",
        "16 channels",
        "64 channels",
        "all channels",
    ]


def test_table4_caption_counts_the_reads_of_both_formats(format_benchmark):
    caption = table4(format_benchmark).caption
    assert "1,890 reads" in caption
    assert caption.isascii()


def test_table4_caption_states_the_reads_per_combination_and_the_budget(
    format_benchmark,
):
    caption = table4(format_benchmark).caption
    assert "Each combination holds 21 reads." in caption
    assert (
        "the read budget of 1 h, once it has made at least three reads."
        in caption
    )


def test_table4_caption_leaves_out_the_budget_without_header_keys(
    format_benchmark,
):
    format_benchmark.attrs.clear()
    assert "budget" not in table4(format_benchmark).caption


def test_table4_raises_without_zarr_direct_reads(format_benchmark):
    with pytest.raises(NoDataError):
        table4(format_benchmark[format_benchmark["format"] != "zarr-direct"])


def test_table4_writes_csv_and_tex(format_benchmark, tmp_path):
    output = write_table(table4(format_benchmark), tmp_path)
    assert [p.name for p in output.paths] == ["table4.csv", "table4.tex"]
    assert output.paths[1].read_text().isascii()


def test_table4_caption_counts_only_the_reads_of_the_two_formats(
    format_benchmark,
):
    trimmed = trim_reads(format_benchmark, 5)
    assert trimmed.iloc[0]["format"] not in ("tszarr", "zarr-direct")
    caption = table4(trimmed).caption
    assert "Each combination holds 21 reads." in caption
    assert "between" not in caption
