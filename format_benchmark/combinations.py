"""The combinations of the format benchmark and the files each one writes."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from format_benchmark.positions import (
    MIN_READS,
    MIN_READS_KEY,
    OVERHEAD,
    READ_BUDGET_KEY,
    READ_BUDGET_S,
    WINDOWS_S,
    positions,
    skip_reason,
)
from tszbench.csvmeta import read_header, read_rows
from tszbench.recordings import RECORDINGS, RecordingSpec, resolve
from tszbench.schemas import (
    FORMAT_PARTS_DIRECTORY,
    FORMAT_ROWS,
    READ_KINDS,
    STORES,
)


def recording(name: str) -> RecordingSpec:
    """Return the recording named name, as `tszbench.recordings.resolve`."""
    return resolve(name)


@dataclass(frozen=True)
class Combination:
    """One recording, store, format, read kind, and window.

    A combination runs in one process and writes one CSV of reads and one CSV
    of the open.
    """

    recording: str
    store: str
    file_format: str
    read_kind: str
    window_s: int

    @property
    def stem(self) -> str:
        """Name of the combination inside its recording directory."""
        return (
            f"{self.store}-{self.file_format}-{self.read_kind}-{self.window_s}s"
        )

    def reads_path(self, results: Path) -> Path:
        """Return the CSV of the timed reads."""
        return (
            results
            / FORMAT_PARTS_DIRECTORY
            / self.recording
            / f"{self.stem}.csv"
        )

    def open_path(self, results: Path) -> Path:
        """Return the CSV of the open."""
        return (
            results
            / FORMAT_PARTS_DIRECTORY
            / self.recording
            / f"{self.stem}-open.csv"
        )


def plan(
    recordings: Sequence[str],
    stores: Sequence[str],
    formats: Sequence[str],
    read_kinds: Sequence[str],
    windows: Sequence[int],
) -> tuple[list[Combination], list[tuple[Combination, str]]]:
    """Return the combinations to measure and the ones skipped with a reason.

    Both lists follow one order: recording, store, format, read kind, window,
    each in the order given. The overhead row is measured once for each
    recording and store, at its first window that fits, as a one-channel read.
    """
    measured: list[Combination] = []
    skipped: list[tuple[Combination, str]] = []
    for name in recordings:
        spec = recording(name)
        for store in stores:
            for file_format in formats:
                overhead_taken = False
                for read_kind in read_kinds:
                    for window_s in windows:
                        combination = Combination(
                            name, store, file_format, read_kind, window_s
                        )
                        reason = skip_reason(
                            spec, file_format, read_kind, window_s
                        )
                        if reason is None and file_format == OVERHEAD:
                            if overhead_taken:
                                reason = "overhead is measured at one window"
                            overhead_taken = True
                        if reason is None:
                            measured.append(combination)
                        else:
                            skipped.append((combination, reason))
    return measured, skipped


def expected_rows(combination: Combination, max_repeats: int | None) -> int:
    """Return the number of timed reads the combination writes."""
    return len(
        positions(
            recording(combination.recording), combination.window_s, max_repeats
        )
    )


def _same_options(
    header: dict[str, str], read_budget_s: float, min_reads: int
) -> bool:
    """Return whether a part header records read_budget_s and min_reads."""
    try:
        return (
            float(header[READ_BUDGET_KEY]) == read_budget_s
            and int(header[MIN_READS_KEY]) == min_reads
        )
    except (KeyError, ValueError):
        return False


def is_complete(
    combination: Combination,
    results: Path,
    max_repeats: int | None,
    read_budget_s: float = READ_BUDGET_S,
    min_reads: int = MIN_READS,
) -> bool:
    """Return whether both CSVs of the combination exist and are whole.

    The open CSV holds one row, whose `planned_reads` equals the planned count
    of the combination. The reads CSV holds exactly `reads` rows. The part is
    complete when `reads` equals `planned_reads`. A part that made fewer reads
    is complete when its header records read_budget_s and min_reads and it made
    at least the smaller of min_reads and `planned_reads`.
    """
    reads_path = combination.reads_path(results)
    open_path = combination.open_path(results)
    if not (reads_path.is_file() and open_path.is_file()):
        return False
    opened = read_rows(open_path)
    if len(opened) != 1:
        return False
    planned = expected_rows(combination, max_repeats)
    try:
        made = int(opened[0]["reads"])
        recorded = int(opened[0]["planned_reads"])
    except (KeyError, ValueError):
        return False
    if recorded != planned or len(read_rows(reads_path)) != made:
        return False
    if made == planned:
        return True
    return made >= min(min_reads, planned) and _same_options(
        read_header(reads_path), read_budget_s, min_reads
    )


def parse_part(path: Path) -> Combination | None:
    """Return the combination whose reads CSV is path, or None.

    The recording is the name of the parent directory. None means the file is
    not a reads CSV of a combination, for example an open CSV or a stray file.
    """
    if path.suffix != ".csv" or path.stem.endswith("-open"):
        return None
    words = path.stem.split("-")
    if len(words) < 4:
        return None
    store, *middle, read_kind, window = words
    if not (window.endswith("s") and window[:-1].isdigit()):
        return None
    combination = Combination(
        path.parent.name,
        store,
        "-".join(middle),
        read_kind,
        int(window[:-1]),
    )
    if (
        store not in STORES
        or combination.file_format not in FORMAT_ROWS
        or read_kind not in READ_KINDS
        or combination.window_s not in WINDOWS_S
    ):
        return None
    try:
        recording(combination.recording)
    except ValueError:
        return None
    return combination


def sort_key(combination: Combination) -> tuple[int, str, int, int, int, int]:
    """Return the key that orders combinations as `plan` does.

    The order is recording, store, format, read kind, and window. A recording
    outside `RECORDINGS`, such as a 1 h cut, follows them by name.
    """
    names = list(RECORDINGS)
    index = (
        names.index(combination.recording)
        if combination.recording in names
        else len(names)
    )
    return (
        index,
        combination.recording,
        STORES.index(combination.store),
        FORMAT_ROWS.index(combination.file_format),
        READ_KINDS.index(combination.read_kind),
        WINDOWS_S.index(combination.window_s),
    )


def complete_parts(
    results: Path,
    max_repeats: int | None,
    read_budget_s: float = READ_BUDGET_S,
    min_reads: int = MIN_READS,
) -> list[Combination]:
    """Return every combination with whole CSVs under results, in plan order.

    Completeness is as in `is_complete`.

    The search covers every recording directory, so a run of a few
    combinations does not hide the ones that earlier runs finished.
    """
    found = [
        combination
        for path in (results / FORMAT_PARTS_DIRECTORY).glob("*/*.csv")
        if (combination := parse_part(path)) is not None
        and is_complete(
            combination, results, max_repeats, read_budget_s, min_reads
        )
    ]
    return sorted(found, key=sort_key)
