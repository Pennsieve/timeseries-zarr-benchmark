import pytest

from tszbench.recordings import RECORDINGS, resolve, sample_range


@pytest.mark.parametrize(
    ("recording_id", "samples", "raw_bytes"),
    [
        ("R1", 22_118_400, 88_473_600),
        ("R2", 44_236_800, 5_662_310_400),
        ("R3", 172_800_000, 138_240_000_000),
        ("R4", 108_000_000, 165_888_000_000),
    ],
)
def test_recordings_match_table_1(recording_id, samples, raw_bytes):
    spec = RECORDINGS[recording_id]
    assert spec.samples_per_channel == samples
    assert spec.raw_bytes == raw_bytes


def test_edf_records_of_r3_hold_51200_bytes():
    spec = RECORDINGS["R3"]
    assert spec.edf_record_s is not None
    record_samples = round(spec.edf_record_s * spec.rate_hz)
    assert 2 * spec.channels * record_samples == 51_200


def test_hour_cut_keeps_every_gap_and_the_seed():
    cut = resolve("R3-1h")
    assert cut.id == "R3-1h"
    assert cut.duration_s == 3_600
    assert cut.seed == RECORDINGS["R3"].seed
    assert cut.gaps == RECORDINGS["R3"].gaps


def test_short_cut_drops_and_shortens_gaps():
    cut = RECORDINGS["R1"].cut(1_500, "test")
    assert [(gap.start_s, gap.duration_s) for gap in cut.gaps] == [
        (600.0, 1.0),
        (1200.0, 10.0),
    ]
    assert RECORDINGS["R1"].cut(1_805, "test").gaps[-1].duration_s == 5.0


def test_rejects_an_unknown_recording():
    with pytest.raises(ValueError, match="must be R1, R2, or R3"):
        resolve("R4-1h")


@pytest.mark.parametrize(
    ("start_us", "end_us", "rate_hz", "expected"),
    [
        (0, 10_000_000, 256, (0, 2_560)),
        (1, 10_000_001, 256, (1, 2_561)),
        (3_906, 3_907, 256, (1, 2)),
        (500, 1_000, 2_000, (1, 2)),
        (499, 1_001, 2_000, (1, 3)),
    ],
)
def test_sample_range_holds_samples_that_start_in_the_window(
    start_us, end_us, rate_hz, expected
):
    assert sample_range(start_us, end_us, rate_hz) == expected
