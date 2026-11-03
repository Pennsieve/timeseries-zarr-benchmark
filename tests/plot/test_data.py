import pytest

from plot.data import (
    as_bool,
    budget_label,
    count_noun,
    failed_request_seconds,
    failed_request_sentence,
    format_bytes,
    format_sig,
    header_number,
    load_csv,
    reads_sentence,
    recording_order,
    select_runs,
    spell_count,
    window_label,
)
from resampling_benchmark.load.generator import REQUEST_TIMEOUT_S
from tests.plot.conftest import FAILED_REQUEST_SECONDS, trim_reads
from tszbench.schemas import FORMAT_OPENS, RESAMPLING_RUNS


def test_load_csv_skips_header_lines_and_keeps_text_columns(results_dir):
    frame = load_csv(results_dir / "storage.csv", ["recording", "member"])
    assert frame["member"].iloc[0] == ""
    assert frame["bytes"].dtype.kind in "if"


def test_load_csv_names_missing_columns(results_dir):
    with pytest.raises(ValueError, match="missing columns"):
        load_csv(results_dir / "storage.csv", ["no_such_column"])


def test_load_csv_keeps_the_header_lines(results_dir):
    frame = load_csv(results_dir / "format-opens.csv", FORMAT_OPENS)
    assert frame.attrs["header"]["read_budget_s"] == "3600"
    assert frame.attrs["header"]["min_reads"] == "3"


def test_header_number_reads_a_finite_number_only():
    header = {"a": "3600", "b": "0.5", "c": "text", "d": "nan"}
    assert header_number(header, "a") == 3600
    assert header_number(header, "b") == 0.5
    assert header_number(header, "c") is None
    assert header_number(header, "d") is None
    assert header_number(header, "e") is None


def test_budget_label_uses_hours_or_minutes_for_a_whole_number():
    assert budget_label(3600) == "1 h"
    assert budget_label(7200.0) == "2 h"
    assert budget_label(1800) == "30 min"
    assert budget_label(90) == "90 s"
    assert budget_label(0.5) == "0.5 s"


def test_reads_sentence_gives_the_fewest_and_most_reads(format_benchmark):
    assert reads_sentence(trim_reads(format_benchmark, 4)).startswith(
        " Each combination holds between four and 21 reads. A combination "
        "stops after the read that brings the sum of its read seconds to at "
        "least the read budget of 1 h, once it has made at least three reads."
    )


def test_reads_sentence_takes_the_budget_from_the_header(format_benchmark):
    format_benchmark.attrs["header"] = {
        "read_budget_s": "1800.0",
        "min_reads": "5",
    }
    assert reads_sentence(format_benchmark) == (
        " Each combination holds 21 reads. A combination stops after the "
        "read that brings the sum of its read seconds to at least the read "
        "budget of 30 min, once it has made at least five reads."
    )


def test_reads_sentence_leaves_out_the_minimum_without_its_key(
    format_benchmark,
):
    format_benchmark.attrs["header"] = {"read_budget_s": "3600"}
    assert reads_sentence(format_benchmark).endswith("read budget of 1 h.")


def test_reads_sentence_leaves_out_the_budget_without_its_key(format_benchmark):
    format_benchmark.attrs["header"] = {"min_reads": "3"}
    assert (
        reads_sentence(format_benchmark) == " Each combination holds 21 reads."
    )


def test_failed_request_seconds_prefers_the_header(runs):
    assert failed_request_seconds(runs) == FAILED_REQUEST_SECONDS


def test_failed_request_seconds_falls_back_to_the_request_timeout(runs):
    runs.attrs.clear()
    assert failed_request_seconds(runs) == REQUEST_TIMEOUT_S


def test_failed_request_sentence_prefers_the_header(runs):
    assert failed_request_sentence(runs, percentiles=True) == (
        " The medians and percentiles count each failed request as at least "
        f"{FAILED_REQUEST_SECONDS} s."
    )


def test_failed_request_sentence_falls_back_to_the_request_timeout(runs):
    runs.attrs.clear()
    assert failed_request_sentence(runs, percentiles=False) == (
        " The medians count each failed request as at least 180 s."
    )


def test_as_bool_reads_true_in_any_case(runs):
    import pandas as pd

    assert list(as_bool(pd.Series(["True", "false", "TRUE", " true "]))) == [
        True,
        False,
        True,
        True,
    ]


def test_recording_order_puts_known_recordings_first():
    assert recording_order(["R3-1h", "R4", "R1", "R3"]) == [
        "R1",
        "R3",
        "R4",
        "R3-1h",
    ]


def test_select_runs_prefers_the_repeated_run(runs):
    chosen = select_runs(runs)
    cell = chosen[
        (chosen["method"] == "server")
        & (chosen["recording"] == "R3")
        & (chosen["window_s"] == 10)
        & (chosen["users"] == 100)
    ]
    assert len(cell) == 1
    assert cell["median_s"].iloc[0] == 7.0
    assert len(chosen) == len(runs) - 1


def test_select_runs_keeps_every_column(runs):
    assert set(RESAMPLING_RUNS) <= set(select_runs(runs).columns)


@pytest.mark.parametrize(
    ("seconds", "label"),
    [
        (1, "1 s"),
        (10, "10 s"),
        (60, "1 min"),
        (600, "10 min"),
        (3600, "1 h"),
        (86_400, "24 h"),
        (90, "90 s"),
    ],
)
def test_window_label(seconds, label):
    assert window_label(seconds) == label


@pytest.mark.parametrize(
    ("value", "digits", "text"),
    [
        (0.04567, 3, "0.0457"),
        (123.456, 3, "123"),
        (1.0, 3, "1.00"),
        (12_345.6, 3, "12,346"),
        (0.0, 3, "0"),
        (5 / 3, 3, "1.67"),
    ],
)
def test_format_sig(value, digits, text):
    assert format_sig(value, digits) == text


def test_format_bytes_uses_decimal_units():
    assert format_bytes(88_473_600) == "88.5 MB"
    assert format_bytes(138_240_000_000) == "138 GB"
    assert format_bytes(12) == "12 B"
    assert format_bytes(4_000_000) == "4 MB"


def test_counts_spell_words_below_ten():
    assert spell_count(3) == "three"
    assert spell_count(10) == "10"
    assert spell_count(12_345) == "12,345"
    assert count_noun(1, "read") == "one read"
    assert count_noun(1_200, "read") == "1,200 reads"
