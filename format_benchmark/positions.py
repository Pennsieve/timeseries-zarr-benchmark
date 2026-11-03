"""The windows of the format benchmark and the rules that decide which are measured."""

from dataclasses import dataclass

import numpy as np

from tszbench import schemas
from tszbench.recordings import RecordingSpec

WINDOWS_S = (1, 10, 60, 600)
"""Window lengths in seconds: 1 s, 10 s, 1 min, and 10 min."""

SHORT_REPEATS = 100
"""Reads per combination for windows up to `SHORT_WINDOW_MAX_S`."""

SHORT_WINDOW_MAX_S = 60
"""Longest window that gets `SHORT_REPEATS` reads."""

MEDIUM_REPEATS = 20
"""Reads per combination at a window longer than `SHORT_WINDOW_MAX_S`."""

READ_BUDGET_S = 60
"""Default read budget of a combination, in seconds. A combination stops after
the read that brings the sum of its timed read seconds to at least this value,
once it has made `MIN_READS` reads."""

MIN_READS = 1
"""Default number of timed reads a combination makes before the read budget can
stop it. A combination with fewer planned reads makes all of them."""

READ_BUDGET_KEY = "read_budget_s"
"""Header key of the part, `format-reads.csv`, and `format-opens.csv` files that records
the read budget."""

MIN_READS_KEY = "min_reads"
"""Header key of the same files that records the reads made before the read
budget applies."""

MAX_READ_BYTES = 8_000_000_000
"""A single read of float32 samples above this size is not measured."""

SUBSET_WINDOWS_S = (10, 600)
"""Windows at which the subset read kinds are measured."""

FLOAT32_BYTES = 4

OVERHEAD = "overhead"


@dataclass(frozen=True)
class Position:
    """One timed read.

    Attributes:
        repeat: Index of the read within its combination, from 0.
        channel: Channel of a one-channel read. A subset read derives the first
            channel of its block from it. An all-channel read ignores it.
        start_us: Start of the window in whole microseconds from sample 0.
    """

    repeat: int
    channel: int
    start_us: int


def window_index(window_s: int) -> int:
    """Return the index of window_s in `WINDOWS_S`.

    Raises:
        ValueError: window_s is not one of `WINDOWS_S`.
    """
    if window_s not in WINDOWS_S:
        raise ValueError(f"window {window_s} s: must be one of {WINDOWS_S}")
    return WINDOWS_S.index(window_s)


def repeats(window_s: int) -> int:
    """Return the number of reads of a window of window_s seconds.

    Raises:
        ValueError: window_s is not one of `WINDOWS_S`.
    """
    window_index(window_s)
    if window_s <= SHORT_WINDOW_MAX_S:
        return SHORT_REPEATS
    return MEDIUM_REPEATS


def positions(
    spec: RecordingSpec, window_s: int, max_repeats: int | None = None
) -> list[Position]:
    """Return the reads of one window of one recording.

    The generator is seeded from the recording seed and the index of the window
    in `WINDOWS_S`. Every format, store, and read kind reads the same windows.
    A smaller max_repeats keeps a prefix of the same reads.

    Args:
        spec: The recording.
        window_s: One of `WINDOWS_S`.
        max_repeats: Cap on the number of reads, or None for `repeats`.

    Raises:
        ValueError: window_s is not one of `WINDOWS_S`, is longer than the
            recording, or max_repeats is below 1.
    """
    index = window_index(window_s)
    window_us = window_s * 1_000_000
    if window_us > spec.duration_us:
        raise ValueError(
            f"window {window_s} s: longer than recording {spec.id!r} of "
            f"{spec.duration_s} s"
        )
    if max_repeats is not None and max_repeats < 1:
        raise ValueError(f"max_repeats is {max_repeats}: must be at least 1")
    count = repeats(window_s)
    generator = np.random.default_rng([spec.seed, index])
    starts = generator.integers(
        0, spec.duration_us - window_us + 1, size=count, dtype=np.int64
    )
    channels = generator.integers(0, spec.channels, size=count, dtype=np.int64)
    reads = [
        Position(repeat, int(channel), int(start))
        for repeat, (channel, start) in enumerate(
            zip(channels, starts, strict=True)
        )
    ]
    return reads if max_repeats is None else reads[:max_repeats]


def selection(
    spec: RecordingSpec, read_kind: str, position: Position
) -> int | range | None:
    """Return the channels a read selects.

    A one-channel read gives its channel and an all-channel read gives None. A
    subset read gives a contiguous range of channels. The block starts at the
    channel of the position, folded so that the block fits in the recording.
    Every format reads the same block.

    Raises:
        ValueError: read_kind is not in `schemas.READ_KINDS`, or a subset read
            is as wide as the recording or wider.
    """
    if read_kind not in schemas.READ_KINDS:
        raise ValueError(
            f"read kind {read_kind!r}: must be one of {schemas.READ_KINDS}"
        )
    if read_kind == "all":
        return None
    size = schemas.subset_size(read_kind)
    if size is None:
        return position.channel
    if size >= spec.channels:
        raise ValueError(
            f"read kind {read_kind!r}: needs more than {size} channels, "
            f"recording {spec.id!r} has {spec.channels}"
        )
    first = position.channel % (spec.channels - size + 1)
    return range(first, first + size)


def read_bytes(spec: RecordingSpec, read_kind: str, window_s: int) -> int:
    """Return the size of one read as float32 samples in bytes."""
    if read_kind == "all":
        channels = spec.channels
    else:
        channels = schemas.subset_size(read_kind) or 1
    return window_s * spec.rate_hz * channels * FLOAT32_BYTES


def skip_reason(
    spec: RecordingSpec, file_format: str, read_kind: str, window_s: int
) -> str | None:
    """Return why a combination is not measured, or None when it is.

    A combination is not measured when the recording lacks the format, the
    window is longer than the recording, or one read exceeds
    `MAX_READ_BYTES`. The overhead row has no all-channel or subset read. A
    subset read needs a window in `SUBSET_WINDOWS_S` and a recording with more
    channels than the block.

    Raises:
        ValueError: A name is not in `schemas`, or window_s is not one of
            `WINDOWS_S`.
    """
    if file_format not in schemas.FORMAT_ROWS:
        raise ValueError(
            f"format {file_format!r}: must be one of {schemas.FORMAT_ROWS}"
        )
    if read_kind not in schemas.READ_KINDS:
        raise ValueError(
            f"read kind {read_kind!r}: must be one of {schemas.READ_KINDS}"
        )
    window_index(window_s)
    if file_format == "edf" and spec.edf_record_s is None:
        return f"recording {spec.id} has no EDF file"
    size = schemas.subset_size(read_kind)
    if file_format == OVERHEAD and (read_kind == "all" or size is not None):
        return "the overhead row reads one object"
    if size is not None:
        if window_s not in SUBSET_WINDOWS_S:
            windows = ", ".join(f"{window} s" for window in SUBSET_WINDOWS_S)
            return f"subset reads are measured only at windows of {windows}"
        if size >= spec.channels:
            return (
                f"a block of {size} channels does not fit recording "
                f"{spec.id}, which has {spec.channels}"
            )
    if window_s * 1_000_000 > spec.duration_us:
        return f"window of {window_s} s is longer than recording {spec.id}"
    nbytes = read_bytes(spec, read_kind, window_s)
    if nbytes > MAX_READ_BYTES:
        return f"one read is {nbytes} bytes, above {MAX_READ_BYTES}"
    return None
