"""The synthetic recordings of the benchmark and their sample grid."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime

START_TIME = datetime(2026, 1, 1, tzinfo=UTC)
"""Wall-clock time of sample 0 of every recording."""

GRID_UV = 0.125
"""Spacing of the sample grid in microvolts. Every stored sample is a multiple."""


@dataclass(frozen=True)
class Gap:
    """A run of NaN samples planted on one channel.

    Attributes:
        channel: Channel index, from 0.
        start_s: Time of the first NaN sample, in seconds from sample 0.
        duration_s: Length of the run in seconds.
    """

    channel: int
    start_s: float
    duration_s: float

    def sample_range(self, rate_hz: int) -> tuple[int, int]:
        """Return the half-open range of sample indices the gap covers."""
        first = round(self.start_s * rate_hz)
        return first, first + round(self.duration_s * rate_hz)


@dataclass(frozen=True)
class RecordingSpec:
    """One synthetic recording.

    Attributes:
        id: Name of the recording and stem of every file written for it.
        modality: Description for Table 1.
        channels: Number of channels.
        rate_hz: Sample rate of every channel.
        duration_s: Length of every channel in seconds.
        rms_uv: Root mean square amplitude of each channel in microvolts.
        bump_hz: Pass band of the spectral bump, low and high edge in hertz.
        seed: Master seed. Each channel draws from a child of it.
        edf_record_s: Duration of one EDF data record in seconds, or None when
            the recording has no EDF file.
        levels: Number of levels a TS-Zarr bundle of the full recording holds.
        gaps: NaN runs planted in the recording.
    """

    id: str
    modality: str
    channels: int
    rate_hz: int
    duration_s: int
    rms_uv: float
    bump_hz: tuple[float, float]
    seed: int
    edf_record_s: float | None
    levels: int
    gaps: tuple[Gap, ...]

    @property
    def samples_per_channel(self) -> int:
        """Number of samples in each channel."""
        return self.rate_hz * self.duration_s

    @property
    def duration_us(self) -> int:
        """Length of every channel in microseconds."""
        return self.duration_s * 1_000_000

    @property
    def raw_bytes(self) -> int:
        """Size of every sample as float32."""
        return 4 * self.channels * self.samples_per_channel

    def cut(self, duration_s: int, suffix: str) -> "RecordingSpec":
        """Return the first duration_s seconds of this recording.

        A cut holds the same samples as the start of the full recording. It keeps
        the gaps that start inside it and shortens a gap that runs past its end.
        Its id is the full id followed by `-` and suffix.
        """
        if not 0 < duration_s <= self.duration_s:
            raise ValueError(
                f"cut of {duration_s} s: must be between 1 s and {self.duration_s} s"
            )
        gaps = tuple(
            replace(
                gap, duration_s=min(gap.duration_s, duration_s - gap.start_s)
            )
            for gap in self.gaps
            if gap.start_s < duration_s
        )
        return replace(
            self, id=f"{self.id}-{suffix}", duration_s=duration_s, gaps=gaps
        )


GAPS = (
    Gap(channel=0, start_s=600.0, duration_s=1.0),
    Gap(channel=0, start_s=1200.0, duration_s=10.0),
    Gap(channel=0, start_s=1800.0, duration_s=600.0),
)
"""The three gaps of every recording. All lie inside the first hour."""

DAY_S = 86_400
HOUR_S = 3_600

RECORDINGS = {
    spec.id: spec
    for spec in (
        RecordingSpec(
            id="R1",
            modality="single-channel ambulatory EEG",
            channels=1,
            rate_hz=256,
            duration_s=DAY_S,
            rms_uv=20.0,
            bump_hz=(8.0, 12.0),
            seed=1,
            edf_record_s=1.0,
            levels=7,
            gaps=GAPS,
        ),
        RecordingSpec(
            id="R2",
            modality="scalp EEG",
            channels=32,
            rate_hz=512,
            duration_s=DAY_S,
            rms_uv=20.0,
            bump_hz=(8.0, 12.0),
            seed=2,
            edf_record_s=1.0,
            levels=7,
            gaps=GAPS,
        ),
        RecordingSpec(
            id="R3",
            modality="intracranial EEG",
            channels=200,
            rate_hz=2000,
            duration_s=DAY_S,
            rms_uv=50.0,
            bump_hz=(8.0, 12.0),
            seed=3,
            edf_record_s=0.064,
            levels=8,
            gaps=GAPS,
        ),
        RecordingSpec(
            id="R4",
            modality="Neuropixels-like",
            channels=384,
            rate_hz=30_000,
            duration_s=HOUR_S,
            rms_uv=100.0,
            bump_hz=(300.0, 3000.0),
            seed=4,
            edf_record_s=None,
            levels=8,
            gaps=GAPS,
        ),
    )
}
"""The four recordings of the benchmark, by id."""


def hour_cut(recording_id: str) -> RecordingSpec:
    """Return the 1 h cut of R1, R2, or R3, with the id suffix `1h`."""
    if recording_id not in {"R1", "R2", "R3"}:
        raise ValueError(f"1 h cut of {recording_id!r}: must be R1, R2, or R3")
    return RECORDINGS[recording_id].cut(HOUR_S, "1h")


def resolve(name: str) -> RecordingSpec:
    """Return the recording named R1 to R4, or the 1 h cut named R1-1h to R3-1h."""
    if name in RECORDINGS:
        return RECORDINGS[name]
    stem, _, suffix = name.partition("-")
    if suffix == "1h":
        return hour_cut(stem)
    raise ValueError(
        f"recording {name!r}: must be one of {sorted(RECORDINGS)} or R1-1h, "
        "R2-1h, R3-1h"
    )


def sample_range(start_us: int, end_us: int, rate_hz: int) -> tuple[int, int]:
    """Return the half-open range of samples whose start lies in the window.

    The window is `[start_us, end_us)` in microseconds from sample 0. A window of
    whole seconds holds exactly `seconds * rate_hz` samples. The range is not
    clipped to the recording.
    """
    first = -((-start_us * rate_hz) // 1_000_000)
    stop = -((-end_us * rate_hz) // 1_000_000)
    return first, stop
