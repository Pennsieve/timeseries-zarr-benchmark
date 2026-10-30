"""Seeded streaming synthesis of the benchmark recordings."""

import math
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfilt

from tszbench.recordings import GRID_UV, RecordingSpec

DIGITAL_GAP = -32768
"""int16 value of a gap sample in a digital block. Other samples lie in
`[-32767, 32767]`."""

DIGITAL_LIMIT = 32767
"""Largest magnitude of a digital sample outside a gap."""

BUMP_POWER = 0.25
"""Share of each channel's variance that the band-passed component carries."""

PINK_LOW_HZ = 0.1
"""Frequency of the slowest pole of the 1/f filter. The spectrum is flat below."""

PINK_SECTIONS_PER_DECADE = 3
"""Number of pole and zero pairs of the 1/f filter in each decade of frequency."""

PINK_HIGH_FRACTION = 0.9
"""Highest zero of the 1/f filter, as a fraction of the Nyquist frequency."""

BUMP_ORDER = 4
"""Order of the Butterworth low-pass prototype of the band-pass filter."""

WARMUP_TIME_CONSTANTS = 10
"""Length of the warm-up stretch in time constants of the slowest pole."""

IMPULSE_TIME_CONSTANTS = 12
"""Length of the impulse response that sets the filter gain, in time constants
of the slowest pole."""

WARMUP_CHUNK_SAMPLES = 65_536
"""Samples the warm-up filters at a time, to bound memory."""

_SECTION_COEFFICIENTS = 6


def _pink_sections(rate_hz: int) -> npt.NDArray[np.float64]:
    """Return second-order sections of a filter with a 1/f power spectrum.

    The poles are spaced evenly on a log axis from `PINK_LOW_HZ` upward, at
    `PINK_SECTIONS_PER_DECADE` per decade. Each first-order section has a zero
    above its pole that makes it drop the amplitude by the factor that gives
    an average slope of -10 dB per decade of power. The bilinear transform maps
    the sections to the digital domain. Pairs of sections share one
    second-order row.
    """
    nyquist = rate_hz / 2
    zero_ratio = 10 ** (0.5 / PINK_SECTIONS_PER_DECADE)
    poles: list[float] = []
    index = 0
    while True:
        pole = PINK_LOW_HZ * 10 ** (index / PINK_SECTIONS_PER_DECADE)
        if pole * zero_ratio > PINK_HIGH_FRACTION * nyquist:
            break
        poles.append(pole)
        index += 1
    sections = []
    for pole in poles:
        omega_pole = 2 * math.pi * pole
        omega_zero = omega_pole * zero_ratio
        twice_rate = 2.0 * rate_hz
        numerator = (
            np.array(
                [twice_rate + omega_zero, omega_zero - twice_rate],
                dtype=np.float64,
            )
            * omega_pole
            / omega_zero
        )
        denominator = np.array(
            [twice_rate + omega_pole, omega_pole - twice_rate],
            dtype=np.float64,
        )
        sections.append(
            (numerator / denominator[0], denominator / denominator[0])
        )
    rows = []
    for first in range(0, len(sections), 2):
        pair = sections[first : first + 2]
        numerator = pair[0][0]
        denominator = pair[0][1]
        for other_numerator, other_denominator in pair[1:]:
            numerator = np.convolve(numerator, other_numerator)
            denominator = np.convolve(denominator, other_denominator)
        row = np.zeros(_SECTION_COEFFICIENTS, dtype=np.float64)
        row[: len(numerator)] = numerator
        row[3] = 1.0
        row[4 : 3 + len(denominator)] = denominator[1:]
        rows.append(row)
    return np.stack(rows)


def _bump_sections(
    rate_hz: int, band_hz: tuple[float, float]
) -> npt.NDArray[np.float64]:
    """Return the second-order sections of the Butterworth band-pass."""
    sections: npt.NDArray[np.float64] = butter(
        BUMP_ORDER, band_hz, btype="bandpass", fs=rate_hz, output="sos"
    )
    return sections


def _unit_variance(
    sections: npt.NDArray[np.float64], length: int
) -> npt.NDArray[np.float64]:
    """Return sections scaled to unit output variance for unit white input.

    The output variance of a linear filter for unit white input is the sum of
    the squared impulse response. The sum runs over `length` samples, which
    spans the decay of the slowest pole.
    """
    impulse = np.zeros(length, dtype=np.float64)
    impulse[0] = 1.0
    response = sosfilt(sections, impulse)
    scaled = sections.copy()
    scaled[0, :3] /= math.sqrt(float(np.sum(response**2)))
    return scaled


@dataclass
class _ChannelState:
    """Generators and filter state of one channel between blocks."""

    background: np.random.Generator
    bump: np.random.Generator
    background_state: npt.NDArray[np.float64]
    bump_state: npt.NDArray[np.float64]


class Synthesizer:
    """Generate the samples of one recording in blocks, in sample order.

    Each channel is Gaussian noise with a 1/f spectrum plus a band-passed
    component over `spec.bump_hz`, scaled to `spec.rms_uv`, quantized to
    `GRID_UV`, and clipped to `[-32767, 32767] * GRID_UV`. Each channel draws
    from its own child of `spec.seed`. Filter state carries across blocks, so
    every block length gives the same samples, and a cut of a recording gives
    the first samples of the full recording. The gaps of the spec are NaN.
    """

    def __init__(self, spec: RecordingSpec, workers: int = 1) -> None:
        """Prepare the filters and the child seeds of spec.

        Args:
            spec: The recording to generate.
            workers: Number of threads that generate channels at once.
        """
        if workers < 1:
            raise ValueError(f"workers {workers}: must be at least 1")
        for gap in spec.gaps:
            if not 0 <= gap.channel < spec.channels:
                raise ValueError(
                    f"gap on channel {gap.channel}: the recording has "
                    f"{spec.channels} channels"
                )
        self._spec = spec
        self._workers = workers
        time_constant = spec.rate_hz / (2 * math.pi * PINK_LOW_HZ)
        self._warmup_samples = round(WARMUP_TIME_CONSTANTS * time_constant)
        impulse_samples = round(IMPULSE_TIME_CONSTANTS * time_constant)
        # Each filter is scaled so its output variance is 1 for unit white input.
        # The variance is the sum of the squared impulse response, so the mix of
        # two independent components has variance 1 before the rms scaling.
        self._pink = _unit_variance(
            _pink_sections(spec.rate_hz), impulse_samples
        )
        self._bump = _unit_variance(
            _bump_sections(spec.rate_hz, spec.bump_hz), impulse_samples
        )
        self._weights = (
            math.sqrt(1.0 - BUMP_POWER) * spec.rms_uv / GRID_UV,
            math.sqrt(BUMP_POWER) * spec.rms_uv / GRID_UV,
        )

    def _start(self, children: np.random.SeedSequence) -> _ChannelState:
        """Return the state of a channel at sample 0, after the warm-up."""
        background_seed, bump_seed, warmup_seed = children.spawn(3)
        warmup = np.random.Generator(np.random.PCG64(warmup_seed))
        state = _ChannelState(
            np.random.Generator(np.random.PCG64(background_seed)),
            np.random.Generator(np.random.PCG64(bump_seed)),
            np.zeros((len(self._pink), 2)),
            np.zeros((len(self._bump), 2)),
        )
        remaining = self._warmup_samples
        while remaining > 0:
            rows = min(remaining, WARMUP_CHUNK_SAMPLES)
            white = warmup.standard_normal((2, rows))
            _, state.background_state = sosfilt(
                self._pink, white[0], zi=state.background_state
            )
            _, state.bump_state = sosfilt(
                self._bump, white[1], zi=state.bump_state
            )
            remaining -= rows
        return state

    def _column(self, state: _ChannelState, rows: int) -> npt.NDArray[np.int16]:
        """Advance one channel by rows samples and return them as digital values."""
        pink, state.background_state = sosfilt(
            self._pink,
            state.background.standard_normal(rows),
            zi=state.background_state,
        )
        bump, state.bump_state = sosfilt(
            self._bump, state.bump.standard_normal(rows), zi=state.bump_state
        )
        mixed = self._weights[0] * pink + self._weights[1] * bump
        np.rint(mixed, out=mixed)
        np.clip(mixed, -DIGITAL_LIMIT, DIGITAL_LIMIT, out=mixed)
        digital: npt.NDArray[np.int16] = mixed.astype(np.int16)
        return digital

    def blocks(self, block_samples: int) -> Iterator[npt.NDArray[np.float32]]:
        """Yield (rows, channels) float32 blocks in microvolts.

        Every block holds block_samples rows except the last, and the rows
        total `spec.samples_per_channel`. One call starts from sample 0.
        """
        for digital in self.digital_blocks(block_samples):
            block = digital.astype(np.float32)
            block *= np.float32(GRID_UV)
            block[digital == DIGITAL_GAP] = np.nan
            yield block

    def digital_blocks(
        self, block_samples: int
    ) -> Iterator[npt.NDArray[np.int16]]:
        """Yield the samples of `blocks` divided by GRID_UV as int16.

        A gap sample is DIGITAL_GAP.
        """
        if block_samples < 1:
            raise ValueError(
                f"block_samples {block_samples}: must be at least 1"
            )
        spec = self._spec
        children = np.random.SeedSequence(spec.seed).spawn(spec.channels)
        with ThreadPoolExecutor(self._workers) as pool:
            states = list(pool.map(self._start, children))
            for first in range(0, spec.samples_per_channel, block_samples):
                rows = min(block_samples, spec.samples_per_channel - first)
                block = np.empty((rows, spec.channels), dtype=np.int16)

                def fill(
                    channel: int,
                    rows: int = rows,
                    block: npt.NDArray[np.int16] = block,
                ) -> None:
                    block[:, channel] = self._column(states[channel], rows)

                list(pool.map(fill, range(spec.channels)))
                for gap in spec.gaps:
                    start, stop = gap.sample_range(spec.rate_hz)
                    low = max(start - first, 0)
                    high = min(stop - first, rows)
                    if low < high:
                        block[low:high, gap.channel] = DIGITAL_GAP
                yield block
