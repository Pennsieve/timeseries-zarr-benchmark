import pytest

from format_benchmark.positions import (
    MAX_READ_BYTES,
    WINDOWS_S,
    positions,
    repeats,
    skip_reason,
    window_index,
)
from tests.format_benchmark.conftest import TINY_SPEC
from tszbench import schemas
from tszbench.recordings import RECORDINGS


def test_windows_are_one_second_to_one_hour() -> None:
    assert WINDOWS_S == (1, 10, 60, 600, 3600)


def test_repeats_follow_the_plan() -> None:
    assert [repeats(window) for window in WINDOWS_S] == [100, 100, 100, 20, 10]


def test_repeats_reject_a_window_outside_the_plan() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        repeats(5)


def test_window_index_is_the_place_in_the_plan() -> None:
    assert [window_index(window) for window in WINDOWS_S] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("window_s", WINDOWS_S)
def test_positions_hold_the_planned_number_of_reads(window_s: int) -> None:
    spec = RECORDINGS["R2"]
    assert len(positions(spec, window_s)) == repeats(window_s)


def test_positions_repeat_for_the_same_recording_and_window() -> None:
    spec = RECORDINGS["R3"]
    assert positions(spec, 10) == positions(spec, 10)


def test_positions_differ_between_windows() -> None:
    spec = RECORDINGS["R3"]
    starts = [
        [read.start_us for read in positions(spec, window)[:5]]
        for window in (1, 10)
    ]
    assert starts[0] != starts[1]


def test_positions_differ_between_recordings() -> None:
    first = positions(RECORDINGS["R2"], 10)
    second = positions(RECORDINGS["R3"], 10)
    assert [read.start_us for read in first] != [
        read.start_us for read in second
    ]


@pytest.mark.parametrize("window_s", [1, 60, 600, 3600])
def test_positions_keep_the_window_inside_the_recording(
    window_s: int,
) -> None:
    spec = RECORDINGS["R4"]
    for read in positions(spec, window_s):
        assert isinstance(read.start_us, int)
        assert 0 <= read.start_us <= spec.duration_us - window_s * 1_000_000
        assert 0 <= read.channel < spec.channels


def test_positions_cover_every_channel_of_a_small_recording() -> None:
    channels = {read.channel for read in positions(TINY_SPEC, 1)}
    assert channels == set(range(TINY_SPEC.channels))


def test_positions_of_a_recording_with_one_channel_use_channel_zero() -> None:
    assert {read.channel for read in positions(RECORDINGS["R1"], 1)} == {0}


def test_positions_number_the_repeats_from_zero() -> None:
    reads = positions(TINY_SPEC, 10)
    assert [read.repeat for read in reads] == list(range(len(reads)))


def test_positions_cap_keeps_a_prefix() -> None:
    full = positions(TINY_SPEC, 10)
    assert positions(TINY_SPEC, 10, 7) == full[:7]


def test_positions_cap_above_the_plan_changes_nothing() -> None:
    assert positions(TINY_SPEC, 10, 1000) == positions(TINY_SPEC, 10)


def test_positions_reject_a_cap_below_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        positions(TINY_SPEC, 10, 0)


def test_positions_reject_a_window_longer_than_the_recording() -> None:
    with pytest.raises(ValueError, match="longer than recording"):
        positions(TINY_SPEC, 60)


def test_cut_of_a_recording_shares_the_seed_but_not_the_starts() -> None:
    full = positions(RECORDINGS["R1"], 60)
    cut = positions(RECORDINGS["R1"].cut(3600, "1h"), 60)
    assert [read.start_us for read in full] != [read.start_us for read in cut]


@pytest.mark.parametrize("window_s", [600, 3600])
def test_all_channel_reads_of_r4_above_the_limit_are_skipped(
    window_s: int,
) -> None:
    reason = skip_reason(RECORDINGS["R4"], "tszarr", "all", window_s)
    assert reason is not None
    assert str(MAX_READ_BYTES) in reason


def test_all_channel_read_of_r4_at_one_minute_is_measured() -> None:
    assert skip_reason(RECORDINGS["R4"], "tszarr", "all", 60) is None


def test_one_channel_reads_of_r4_are_measured_at_every_window() -> None:
    assert all(
        skip_reason(RECORDINGS["R4"], "tszarr", "one", window) is None
        for window in WINDOWS_S
    )


def test_the_limit_is_exceeded_by_no_read_of_the_other_recordings() -> None:
    for name in ("R1", "R2", "R3"):
        for window in WINDOWS_S:
            assert (
                skip_reason(RECORDINGS[name], "tszarr", "all", window) is None
            )


def test_edf_of_r4_is_skipped() -> None:
    reason = skip_reason(RECORDINGS["R4"], "edf", "one", 1)
    assert reason is not None
    assert "no EDF" in reason


def test_edf_of_r3_is_measured() -> None:
    assert skip_reason(RECORDINGS["R3"], "edf", "one", 1) is None


def test_overhead_has_no_all_channel_read() -> None:
    assert skip_reason(RECORDINGS["R1"], "overhead", "all", 1) is not None
    assert skip_reason(RECORDINGS["R1"], "overhead", "one", 1) is None


def test_a_window_longer_than_the_recording_is_skipped() -> None:
    reason = skip_reason(TINY_SPEC, "tszarr", "one", 60)
    assert reason is not None
    assert "longer than" in reason


def test_skip_reason_rejects_an_unknown_format() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        skip_reason(TINY_SPEC, "parquet", "one", 1)


def test_every_row_of_the_plan_is_known_to_skip_reason() -> None:
    for file_format in schemas.FORMAT_ROWS:
        skip_reason(RECORDINGS["R1"], file_format, "one", 1)
