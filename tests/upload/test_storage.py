import csv
import shutil
from pathlib import Path, PurePosixPath

import pytest

from tszbench import schemas
from tszbench.csvmeta import read_header
from upload import storage


def _row(
    rows: list[dict[str, object]], file_format: str, member: str = ""
) -> dict[str, object]:
    (match,) = [
        r for r in rows if r["format"] == file_format and r["member"] == member
    ]
    return match


def test_classify_sorts_bundle_files_by_path() -> None:
    cases = {
        "zarr.json": "metadata",
        "recording.json": "metadata",
        "0/zarr.json": "metadata",
        "0/raw/zarr.json": "metadata",
        "0/3/mean/zarr.json": "metadata",
        "0/raw/c/0": "raw",
        "12/raw/c/4": "raw",
        "0/1/minmax/c/0/0": "minmax",
        "0/2/mean/c/1": "mean",
        "3/4/var/c/0": "var",
        "3/4/count/c/0": "count",
    }
    for path, member in cases.items():
        assert storage.classify(PurePosixPath(path)) == member, path


def test_classify_rejects_a_file_outside_continuous_channels() -> None:
    with pytest.raises(ValueError, match="events/text/c/0"):
        storage.classify(PurePosixPath("events/text/c/0"))


def test_member_rows_sum_to_the_bundle_total(root: Path) -> None:
    rows = storage.storage_rows(root, "R9")
    total = _row(rows, "tszarr")
    members = [r for r in rows if r["format"] == "tszarr" and r["member"]]
    assert [r["member"] for r in members] == list(storage.MEMBERS)
    assert sum(int(str(r["bytes"])) for r in members) == total["bytes"]
    assert sum(int(str(r["objects"])) for r in members) == total["objects"]


def test_bundle_members_match_the_files_on_disk(root: Path) -> None:
    bundle = root / "R9" / schemas.FILE_NAMES["tszarr"]
    members = storage.bundle_members(bundle)
    raw_chunks = [p for p in bundle.glob("*/raw/c/*") if p.is_file()]
    assert members["raw"] == (sum(p.stat().st_size for p in raw_chunks), 2)
    assert members["minmax"][1] == 2 * 5
    assert members["count"][1] == 2 * 5
    assert members["var"][1] == 2 * 5
    assert members["metadata"][0] >= (bundle / "zarr.json").stat().st_size


def test_member_without_files_has_zero_bytes_and_objects(root: Path) -> None:
    bundle = root / "R9" / schemas.FILE_NAMES["tszarr"]
    for chunk in bundle.glob("*/*/var/c/*"):
        chunk.unlink()
    assert storage.bundle_members(bundle)["var"] == (0, 0)


def test_single_file_format_counts_one_object(root: Path) -> None:
    row = _row(storage.storage_rows(root, "R9"), "edf")
    assert (row["bytes"], row["objects"]) == (1000, 1)


def test_directory_format_sums_every_file(root: Path) -> None:
    row = _row(storage.storage_rows(root, "R9"), "nwb-zarr-time")
    assert (row["bytes"], row["objects"]) == (110, 2)


def test_rows_follow_schema_format_order(root: Path) -> None:
    rows = storage.storage_rows(root, "R9")
    whole = [r["format"] for r in rows if not r["member"]]
    assert whole == list(schemas.FORMATS)


def test_absent_format_has_no_row(root: Path) -> None:
    (root / "R9" / schemas.FILE_NAMES["edf"]).unlink()
    formats = {r["format"] for r in storage.storage_rows(root, "R9")}
    assert "edf" not in formats
    assert "tszarr" in formats


def test_recording_with_only_tszarr_has_rows_for_that_format(
    root: Path,
) -> None:
    for file_format in schemas.FORMATS:
        if file_format != "tszarr":
            path = root / schemas.relative_path("R9", file_format)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    rows = storage.storage_rows(root, "R9")
    assert {r["format"] for r in rows} == {"tszarr"}
    assert [r["member"] for r in rows][1:] == list(storage.MEMBERS)


def test_missing_recording_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="R1"):
        storage.storage_rows(tmp_path, "R1")


def test_csv_columns_match_the_schema(root: Path, tmp_path: Path) -> None:
    output = tmp_path / "out" / "storage.csv"
    rows = storage.write_storage(root, ["R9"], output)
    with output.open() as file:
        lines = [line for line in file if not line.startswith("# ")]
    parsed = list(csv.DictReader(lines))
    assert list(parsed[0]) == list(schemas.STORAGE)
    assert len(parsed) == len(rows)
    assert read_header(output)["recordings"] == "R9"
    assert "commit" in read_header(output)


def test_measuring_a_recording_again_keeps_the_rows_of_others(
    root: Path, tmp_path: Path
) -> None:
    other = root / "R8"
    other.mkdir()
    (other / schemas.FILE_NAMES["edf"]).write_bytes(b"e" * 50)
    output = tmp_path / "storage.csv"
    storage.write_storage(root, ["R8"], output)
    storage.write_storage(root, ["R9"], output)
    rows = storage.write_storage(root, ["R9"], output)
    recordings = [row["recording"] for row in rows]
    assert recordings.count("R8") == 1
    assert recordings.count("R9") == len(storage.storage_rows(root, "R9"))
    assert recordings[0] == "R8"
    assert read_header(output)["recordings"] == "R8,R9"
