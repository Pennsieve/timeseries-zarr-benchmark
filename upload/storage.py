"""Size at rest of the files of one recording."""

from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

from tszbench import schemas
from tszbench.csvmeta import CsvWriter, read_rows, run_metadata

MEMBERS = ("raw", "minmax", "mean", "var", "count", "metadata")
"""Members of a TS-Zarr bundle that have a row of their own in the storage CSV."""

_STATISTICS = ("minmax", "mean", "var", "count")
_METADATA_FILES = ("zarr.json", "recording.json")


def present_formats(root: Path, recording_id: str) -> list[str]:
    """Return the formats that have a file under root, in `schemas.FORMATS` order.

    Raises:
        FileNotFoundError: No format of the recording exists under root.
    """
    found = [
        file_format
        for file_format in schemas.FORMATS
        if (root / schemas.relative_path(recording_id, file_format)).exists()
    ]
    if not found:
        raise FileNotFoundError(
            f"recording {recording_id!r}: no file under {root}, expected one of "
            f"{[schemas.relative_path(recording_id, f) for f in schemas.FORMATS]}"
        )
    return found


def list_files(path: Path) -> Iterator[Path]:
    """Yield path if it is a file, or every file below it if it is a directory."""
    if path.is_file():
        yield path
        return
    for child in sorted(path.rglob("*")):
        if child.is_file():
            yield child


def measure(path: Path) -> tuple[int, int]:
    """Return the sum of the file sizes in bytes and the number of files.

    A single file counts as one object.
    """
    sizes = [file.stat().st_size for file in list_files(path)]
    return sum(sizes), len(sizes)


def classify(relative: PurePosixPath) -> str:
    """Return the member of a TS-Zarr bundle that a file belongs to.

    Args:
        relative: Path of the file below the bundle directory.

    Returns:
        One of `MEMBERS`. Every `zarr.json` and `recording.json` is `metadata`.
        A chunk file below `<channel>/raw` is `raw`. A chunk file below
        `<channel>/<level>/<statistic>` is that statistic.

    Raises:
        ValueError: The file lies in no continuous channel array.
    """
    parts = relative.parts
    if parts[-1] in _METADATA_FILES:
        return "metadata"
    if len(parts) >= 3 and parts[1] == "raw":
        return "raw"
    if len(parts) >= 4 and parts[1].isdigit() and parts[2] in _STATISTICS:
        return parts[2]
    raise ValueError(
        f"bundle file {relative.as_posix()!r}: must be a zarr.json, "
        "recording.json, or chunk of <channel>/raw or "
        "<channel>/<level>/{minmax,mean,var,count}"
    )


def bundle_members(bundle: Path) -> dict[str, tuple[int, int]]:
    """Return bytes and file count of every member of a TS-Zarr bundle.

    Every entry of `MEMBERS` is present. A member with no file has 0 bytes and
    0 objects.
    """
    totals = dict.fromkeys(MEMBERS, (0, 0))
    for file in list_files(bundle):
        member = classify(PurePosixPath(file.relative_to(bundle).as_posix()))
        size, count = totals[member]
        totals[member] = (size + file.stat().st_size, count + 1)
    return totals


def storage_rows(root: Path, recording_id: str) -> list[dict[str, object]]:
    """Return the `schemas.STORAGE` rows of one recording under root.

    Each format present has one row with an empty `member`. The `tszarr` row is
    followed by one row per entry of `MEMBERS`.
    """
    rows: list[dict[str, object]] = []
    for file_format in present_formats(root, recording_id):
        path = root / schemas.relative_path(recording_id, file_format)
        size, objects = measure(path)
        rows.append(
            {
                "recording": recording_id,
                "format": file_format,
                "member": "",
                "bytes": size,
                "objects": objects,
            }
        )
        if file_format == "tszarr":
            rows.extend(
                {
                    "recording": recording_id,
                    "format": file_format,
                    "member": member,
                    "bytes": member_bytes,
                    "objects": member_objects,
                }
                for member, (member_bytes, member_objects) in bundle_members(
                    path
                ).items()
            )
    return rows


def write_storage(
    root: Path, recording_ids: Sequence[str], output: Path
) -> list[dict[str, object]]:
    """Measure recordings under root and write their rows to output.

    The rows of other recordings already in output stay, ahead of the new
    rows. Measuring a recording again replaces its rows.

    Returns:
        Every row written, as `schemas.STORAGE`.

    Raises:
        FileNotFoundError: A recording has no format under root.
    """
    measured = [
        row
        for recording_id in recording_ids
        for row in storage_rows(root, recording_id)
    ]
    kept: list[dict[str, object]] = []
    if output.is_file():
        kept = [
            dict(row)
            for row in read_rows(output)
            if row["recording"] not in recording_ids
        ]
    rows = kept + measured
    recordings = list(dict.fromkeys(str(row["recording"]) for row in rows))
    header = run_metadata(recordings=",".join(recordings))
    with CsvWriter(output, schemas.STORAGE, header) as writer:
        writer.write_all(rows)
    return rows
