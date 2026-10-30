"""EDF writer and a range reader over any fsspec filesystem.

No EDF library reads a window by byte range. pyedflib and MNE take a local path,
and edfio and edfplus load the whole signal. The reader here fetches the data
records under a window and nothing else.

Every signal maps digital `[-32768, 32767]` to physical `[-4096, 4095.875]`
microvolts, so one digital step is `GRID_UV`. The digital minimum marks a gap
sample, and the reader returns it as NaN.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from fsspec.spec import AbstractFileSystem

from tszbench.recordings import (
    START_TIME,
    RecordingSpec,
    sample_range,
)

MERGE_GAP_BYTES = 1 << 20
"""Two byte ranges closer than this merge into one request."""

MAX_REQUEST_BYTES = 16 << 20
"""A request holds at most this many bytes."""

PHYSICAL_MIN_UV = -4096.0
PHYSICAL_MAX_UV = 4095.875
DIGITAL_MIN = -32768
DIGITAL_MAX = 32767

WRITE_BATCH_BYTES = 32 << 20
"""The writer encodes and writes data records in batches of about this size."""

FIXED_HEADER_BYTES = 256
"""Size of the fixed part of the header, and of each signal's fields."""

_DIMENSION = "uV"
_SAMPLE_BYTES = 2
_SIGNAL_FIELDS = (
    ("label", 16),
    ("transducer", 80),
    ("dimension", 8),
    ("physical_min", 8),
    ("physical_max", 8),
    ("digital_min", 8),
    ("digital_max", 8),
    ("prefilter", 80),
    ("samples", 8),
    ("reserved", 32),
)
"""Name and width of each per-signal header field, in file order. The file
stores each field for every signal before the next field."""
_TWO_DIGIT_YEAR_PIVOT = 85


@dataclass(frozen=True)
class EdfHeader:
    """The parsed header of an EDF file.

    Attributes:
        header_bytes: Size of the header, 256 bytes plus 256 per signal.
        records: Number of data records.
        record_s: Duration of one data record in seconds.
        start: Start date and time from the header, naive.
        labels: Label of each signal.
        samples_per_record: Samples of each signal in one data record.
        physical_min: Physical minimum of each signal.
        physical_max: Physical maximum of each signal.
        digital_min: Digital minimum of each signal.
        digital_max: Digital maximum of each signal.
    """

    header_bytes: int
    records: int
    record_s: float
    start: datetime
    labels: tuple[str, ...]
    samples_per_record: tuple[int, ...]
    physical_min: tuple[float, ...]
    physical_max: tuple[float, ...]
    digital_min: tuple[int, ...]
    digital_max: tuple[int, ...]

    @property
    def record_bytes(self) -> int:
        """Size of one data record."""
        return 2 * sum(self.samples_per_record)


def _number(value: float) -> str:
    """Format value as the shortest header text that parses back to it."""
    return str(int(value)) if value == int(value) else repr(float(value))


def _text(value: str, width: int) -> bytes:
    """Return value as ASCII padded with spaces to width."""
    if len(value) > width:
        raise ValueError(
            f"EDF header text {value!r}: must be at most {width} characters"
        )
    return value.ljust(width).encode("ascii")


def _samples_per_record(spec: RecordingSpec) -> int:
    """Return the exact number of samples of one channel in a data record."""
    if spec.edf_record_s is None:
        raise ValueError(f"recording {spec.id}: has no EDF record duration")
    exact = spec.edf_record_s * spec.rate_hz
    samples = round(exact)
    if samples < 1 or abs(exact - samples) > 1e-6:
        raise ValueError(
            f"recording {spec.id}: {spec.edf_record_s} s at {spec.rate_hz} Hz "
            "must hold a whole number of samples"
        )
    if spec.samples_per_channel % samples:
        raise ValueError(
            f"recording {spec.id}: {spec.duration_s} s must be a whole number "
            f"of {spec.edf_record_s} s records"
        )
    return samples


def _encode_header(spec: RecordingSpec, samples_per_record: int) -> bytes:
    """Return the header of spec: the fixed fields, then the signal fields."""
    signals = spec.channels
    fixed = b"".join(
        (
            _text("0", 8),
            _text("X", 80),
            _text(spec.id, 80),
            _text(START_TIME.strftime("%d.%m.%y"), 8),
            _text(START_TIME.strftime("%H.%M.%S"), 8),
            _text(str(FIXED_HEADER_BYTES * (signals + 1)), 8),
            _text("", 44),
            _text(str(spec.samples_per_channel // samples_per_record), 8),
            _text(_number(spec.edf_record_s or 0.0), 8),
            _text(str(signals), 4),
        )
    )
    values = {
        "dimension": _DIMENSION,
        "physical_min": _number(PHYSICAL_MIN_UV),
        "physical_max": _number(PHYSICAL_MAX_UV),
        "digital_min": str(DIGITAL_MIN),
        "digital_max": str(DIGITAL_MAX),
        "samples": str(samples_per_record),
    }
    parts = [fixed]
    for name, width in _SIGNAL_FIELDS:
        for index in range(signals):
            value = (
                f"ch{index:03d}" if name == "label" else values.get(name, "")
            )
            parts.append(_text(value, width))
    return b"".join(parts)


def _regroup(
    blocks: Iterable[npt.NDArray[np.int16]], channels: int, rows: int
) -> Iterator[npt.NDArray[np.int16]]:
    """Yield (rows, channels) arrays cut from blocks, the last one shorter."""
    pending: list[npt.NDArray[np.int16]] = []
    held = 0
    for block in blocks:
        if block.dtype != np.int16 or block.ndim != 2:
            raise ValueError(
                f"block of dtype {block.dtype} and rank {block.ndim}: "
                "must be a rank-2 int16 array"
            )
        if block.shape[1] != channels:
            raise ValueError(
                f"block of {block.shape[1]} channels: must hold {channels}"
            )
        offset = 0
        while offset < block.shape[0]:
            take = min(rows - held, block.shape[0] - offset)
            pending.append(block[offset : offset + take])
            held += take
            offset += take
            if held == rows:
                yield np.concatenate(pending)
                pending, held = [], 0
    if held:
        yield np.concatenate(pending)


def write_edf(
    path: Path,
    spec: RecordingSpec,
    blocks: Iterable[npt.NDArray[np.int16]],
) -> None:
    """Write spec as an EDF file at path, every channel in each data record.

    Args:
        path: File to create.
        spec: The recording. `spec.edf_record_s` sets the record duration.
        blocks: (rows, channels) int16 blocks of any length, in sample order,
            that total `spec.samples_per_channel` rows.

    Raises:
        ValueError: The spec has no EDF record duration, its duration is not
            a whole number of records, or the blocks do not total its samples.
    """
    per_record = _samples_per_record(spec)
    record_bytes = _SAMPLE_BYTES * spec.channels * per_record
    batch_records = max(1, WRITE_BATCH_BYTES // record_bytes)
    written = 0
    try:
        with path.open("wb") as file:
            file.write(_encode_header(spec, per_record))
            for batch in _regroup(
                blocks, spec.channels, batch_records * per_record
            ):
                written += batch.shape[0]
                if (
                    written > spec.samples_per_channel
                    or batch.shape[0] % per_record
                ):
                    break
                records = batch.shape[0] // per_record
                grouped = batch.reshape(records, per_record, spec.channels)
                encoded = grouped.transpose(0, 2, 1).astype("<i2", order="C")
                file.write(encoded)
        if written != spec.samples_per_channel:
            raise ValueError(
                f"blocks of {written} rows: must total "
                f"{spec.samples_per_channel}"
            )
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _field(raw: bytes, name: str, convert: Callable[[str], Any]) -> Any:
    """Return the field text converted, or raise ValueError naming the field."""
    text = raw.decode("ascii", errors="replace").strip()
    try:
        return convert(text)
    except ValueError:
        raise ValueError(
            f"EDF header field {name}: {text!r} does not parse"
        ) from None


def _dotted_integers(text: str) -> list[int]:
    """Return the integers of a text such as 01.02.26."""
    parts = [int(part) for part in text.split(".")]
    if len(parts) != 3:
        raise ValueError(text)
    return parts


def _start(date: bytes, time: bytes) -> datetime:
    """Return the start date (dd.mm.yy) and time (hh.mm.ss) of the header."""
    day, month, year = _field(date, "startdate", _dotted_integers)
    hour, minute, second = _field(time, "starttime", _dotted_integers)
    century = 1900 if year >= _TWO_DIGIT_YEAR_PIVOT else 2000
    try:
        return datetime(century + year, month, day, hour, minute, second)
    except ValueError:
        raise ValueError(
            f"EDF header start {date!r} {time!r}: does not name a date"
        ) from None


def parse_header(read: Callable[[int, int], bytes]) -> EdfHeader:
    """Parse the header of an EDF file.

    Args:
        read: Returns the bytes from a start offset to an end offset. It is
            called twice: for the fixed 256 bytes and for the signal fields.

    Raises:
        ValueError: A field does not parse or the signals differ in sample
            rate.
    """
    fixed = read(0, FIXED_HEADER_BYTES)
    if len(fixed) != FIXED_HEADER_BYTES:
        raise ValueError(
            f"EDF header of {len(fixed)} bytes: must hold at least "
            f"{FIXED_HEADER_BYTES}"
        )
    signals = _field(fixed[252:256], "signals", int)
    header_bytes = _field(fixed[184:192], "header bytes", int)
    if signals < 1 or header_bytes != FIXED_HEADER_BYTES * (signals + 1):
        raise ValueError(
            f"EDF header of {header_bytes} bytes with {signals} signals: "
            f"must hold {FIXED_HEADER_BYTES} bytes per signal and one more"
        )
    records = _field(fixed[236:244], "records", int)
    record_s = _field(fixed[244:252], "record duration", float)
    if records < 0 or record_s <= 0:
        raise ValueError(
            f"EDF header with {records} records of {record_s} s: "
            "must hold a known count of records of positive duration"
        )
    body = read(FIXED_HEADER_BYTES, header_bytes)
    if len(body) != header_bytes - FIXED_HEADER_BYTES:
        raise ValueError(
            f"EDF signal header of {len(body)} bytes: must hold "
            f"{header_bytes - FIXED_HEADER_BYTES}"
        )
    columns: dict[str, list[bytes]] = {}
    offset = 0
    for name, width in _SIGNAL_FIELDS:
        columns[name] = [
            body[offset + index * width : offset + (index + 1) * width]
            for index in range(signals)
        ]
        offset += signals * width
    samples = tuple(_field(raw, "samples", int) for raw in columns["samples"])
    if len(set(samples)) != 1 or samples[0] < 1:
        raise ValueError(
            f"EDF signals of {sorted(set(samples))} samples per record: "
            "must share one positive sample rate"
        )
    return EdfHeader(
        header_bytes=header_bytes,
        records=records,
        record_s=record_s,
        start=_start(fixed[168:176], fixed[176:184]),
        labels=tuple(
            raw.decode("ascii", errors="replace").strip()
            for raw in columns["label"]
        ),
        samples_per_record=samples,
        physical_min=tuple(
            _field(raw, "physical minimum", float)
            for raw in columns["physical_min"]
        ),
        physical_max=tuple(
            _field(raw, "physical maximum", float)
            for raw in columns["physical_max"]
        ),
        digital_min=tuple(
            _field(raw, "digital minimum", int)
            for raw in columns["digital_min"]
        ),
        digital_max=tuple(
            _field(raw, "digital maximum", int)
            for raw in columns["digital_max"]
        ),
    )


def plan_requests(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the requests that fetch ranges, merged and split.

    Sorted half-open byte ranges closer than MERGE_GAP_BYTES merge into one,
    and a merged range longer than MAX_REQUEST_BYTES splits into pieces of at
    most that size.
    """
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if merged and start - merged[-1][1] < MERGE_GAP_BYTES:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    requests: list[tuple[int, int]] = []
    for start, end in merged:
        while end - start > MAX_REQUEST_BYTES:
            requests.append((start, start + MAX_REQUEST_BYTES))
            start += MAX_REQUEST_BYTES
        requests.append((start, end))
    return requests


def _join_adjacent(
    pieces: Sequence[tuple[int, bytes]],
) -> list[tuple[int, npt.NDArray[np.uint8]]]:
    """Join pieces that touch into segments, each as a start and its bytes."""
    groups: list[tuple[int, list[bytes]]] = []
    end = -1
    for start, data in pieces:
        if groups and start == end:
            groups[-1][1].append(data)
        else:
            groups.append((start, [data]))
        end = start + len(data)
    return [
        (
            start,
            np.frombuffer(
                parts[0] if len(parts) == 1 else b"".join(parts), dtype=np.uint8
            ),
        )
        for start, parts in groups
    ]


def _gather(
    segments: Sequence[tuple[int, npt.NDArray[np.uint8]]],
    starts: npt.NDArray[np.int64],
    length: int,
    stride: int,
) -> npt.NDArray[np.uint8]:
    """Copy the rows that begin at `starts` out of the segments.

    A row is `length` bytes. Rows of one segment lie `stride` bytes apart.
    """
    raw = np.empty((len(starts), length), dtype=np.uint8)
    for seg_start, seg in segments:
        low = int(np.searchsorted(starts, seg_start, side="left"))
        high = int(np.searchsorted(starts, seg_start + len(seg), side="left"))
        if high == low:
            continue
        local = int(starts[low]) - seg_start
        view = np.lib.stride_tricks.as_strided(
            seg[local:],
            shape=(high - low, length),
            strides=(stride, 1),
            writeable=False,
        )
        raw[low:high] = view
    return raw


class EdfRangeReader:
    """Read windows of an EDF file by byte range.

    Attributes:
        header: The parsed header, fetched when the reader opens.
    """

    def __init__(self, filesystem: AbstractFileSystem, path: str) -> None:
        """Open path on filesystem and fetch its header.

        Raises:
            ValueError: The header does not parse or the sample rate is not a
                whole number of hertz.
        """
        self._filesystem = filesystem
        self._path = path
        self.header = parse_header(
            lambda start, end: bytes(filesystem.cat_file(path, start, end))
        )
        rate = self.rate_hz
        if abs(rate - round(rate)) > 1e-6:
            raise ValueError(f"EDF sample rate of {rate} Hz: must be whole")
        header = self.header
        digital_min = np.array(header.digital_min, dtype=np.float64)
        digital_max = np.array(header.digital_max, dtype=np.float64)
        physical_min = np.array(header.physical_min, dtype=np.float64)
        physical_max = np.array(header.physical_max, dtype=np.float64)
        self._gain = (physical_max - physical_min) / (digital_max - digital_min)
        self._shift = physical_min - self._gain * digital_min
        self._gain32 = self._gain.astype(np.float32)
        self._shift32 = self._shift.astype(np.float32)
        self._digital_min = np.array(header.digital_min, dtype=np.int16)

    @property
    def rate_hz(self) -> float:
        """Sample rate shared by every signal."""
        return self.header.samples_per_record[0] / self.header.record_s

    def _fetch(
        self, starts: npt.NDArray[np.int64], length: int
    ) -> list[tuple[int, npt.NDArray[np.uint8]]]:
        """Fetch `length` bytes at each start and join the pieces that touch.

        Raises:
            ValueError: The file returns fewer bytes than requested.
        """
        requests = plan_requests([(int(s), int(s) + length) for s in starts])
        data = self._filesystem.cat_ranges(
            [self._path] * len(requests),
            [start for start, _ in requests],
            [end for _, end in requests],
            on_error="raise",
        )
        for (start, end), piece in zip(requests, data, strict=True):
            if len(piece) != end - start:
                raise ValueError(
                    f"EDF bytes {start} to {end}: the file returned "
                    f"{len(piece)} bytes"
                )
        return _join_adjacent(
            [
                (start, bytes(piece))
                for (start, _), piece in zip(requests, data, strict=True)
            ]
        )

    def read(
        self, start_us: int, end_us: int, channel: int | None = None
    ) -> npt.NDArray[np.float32]:
        """Return the samples of `tszbench.recordings.sample_range` in microvolts.

        One channel gives a rank-1 array. `channel=None` gives every channel as
        (samples, channels). A gap sample is NaN. The window is clipped to the
        file. The requests run at once, as many as the filesystem allows.

        Raises:
            ValueError: channel is not an index of a signal.
        """
        header = self.header
        channels = len(header.labels)
        if channel is not None and not 0 <= channel < channels:
            raise ValueError(f"channel {channel}: must be in [0, {channels})")
        per_record = header.samples_per_record[0]
        record_bytes = header.record_bytes
        first, stop = sample_range(start_us, end_us, round(self.rate_hz))
        first = max(first, 0)
        stop = min(stop, per_record * header.records)
        columns = [channel] if channel is not None else list(range(channels))
        if stop <= first:
            empty = np.empty((0, len(columns)), dtype=np.float32)
            return empty[:, 0] if channel is not None else empty

        first_record = first // per_record
        records = (stop - 1) // per_record - first_record + 1
        base = header.header_bytes + first_record * record_bytes
        if channel is None:
            count, length, stride, shift = 1, records * record_bytes, 0, 0
        else:
            count = records
            length = _SAMPLE_BYTES * per_record
            stride = record_bytes
            shift = length * channel
        starts = base + shift + stride * np.arange(count, dtype=np.int64)

        segments = self._fetch(starts, length)
        raw = _gather(segments, starts, length, stride)
        del segments

        digital = (
            raw.reshape(-1)
            .view("<i2")
            .reshape(records, len(columns), per_record)
            .transpose(0, 2, 1)
            .reshape(records * per_record, len(columns))
        )
        offset = first - first_record * per_record
        digital = digital[offset : offset + (stop - first)]
        del raw
        physical = digital.astype(np.float32)
        physical *= self._gain32[columns]
        physical += self._shift32[columns]
        physical[digital == self._digital_min[columns]] = np.nan
        return physical[:, 0] if channel is not None else physical
