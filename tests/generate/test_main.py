import csv
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from generate.__main__ import main, measure
from generate.formats import formats_of
from tszbench import schemas
from tszbench.recordings import RECORDINGS, Gap, RecordingSpec

REPOSITORY = Path(__file__).resolve().parents[2]
SPEC = replace(
    RECORDINGS["R3"],
    id="tiny",
    channels=3,
    rate_hz=512,
    duration_s=20,
    edf_record_s=1.0,
    seed=11,
    gaps=(Gap(channel=0, start_s=2.0, duration_s=1.0),),
)


@pytest.fixture
def tiny(monkeypatch: pytest.MonkeyPatch) -> RecordingSpec:
    """Make the recording name `tiny` resolve to a 20 s recording."""
    resolve = {"tiny": SPEC, "tiny-no-edf": replace(SPEC, edf_record_s=None)}

    def fake(name: str) -> RecordingSpec:
        if name not in resolve:
            raise ValueError(f"recording {name!r}: unknown")
        return resolve[name]

    monkeypatch.setattr("generate.__main__.resolve", fake)
    return SPEC


def read_rows(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    lines = path.read_text().splitlines()
    header = {
        key: value
        for key, _, value in (
            line[2:].partition(": ") for line in lines if line.startswith("# ")
        )
    }
    body = [line for line in lines if not line.startswith("# ")]
    return header, list(csv.DictReader(body))


def run(root: Path, results: Path, *extra: str, recording: str = "tiny") -> int:
    return main(
        [
            "--recording",
            recording,
            "--root",
            str(root),
            "--results",
            str(results),
            *extra,
        ]
    )


def test_the_command_writes_every_format_and_a_row_for_each(
    tiny: RecordingSpec, tmp_path: Path
):
    root, results = tmp_path / "files", tmp_path / "results"
    assert run(root, results, "--workers", "2") == 0
    for name in schemas.FORMATS:
        assert (root / schemas.relative_path(tiny.id, name)).exists()
    _, rows = read_rows(results / "generation-tiny.csv")
    assert [row["format"] for row in rows] == list(schemas.FORMATS)
    for row in rows:
        path = root / schemas.relative_path(tiny.id, row["format"])
        size, objects = measure(path)
        assert row["recording"] == "tiny"
        assert int(row["bytes"]) == size > 0
        assert int(row["objects"]) == objects
        assert float(row["seconds"]) > 0


def test_a_file_is_one_object_and_a_zarr_store_holds_many(
    tiny: RecordingSpec, tmp_path: Path
):
    root = tmp_path / "files"
    run(root, tmp_path / "results")
    _, rows = read_rows(tmp_path / "results" / "generation-tiny.csv")
    objects = {row["format"]: int(row["objects"]) for row in rows}
    assert objects["edf"] == 1
    assert objects["nwb-hdf5-time"] == 1
    assert objects["nwb-hdf5-channel"] == 1
    assert objects["nwb-zarr-time"] > 1
    assert objects["tszarr"] > 1
    assert objects["nwb-zarr-channel"] > objects["nwb-zarr-time"]


def test_the_csv_header_records_the_recording_seed_and_workers(
    tiny: RecordingSpec, tmp_path: Path
):
    run(tmp_path / "files", tmp_path / "results", "--workers", "3")
    header, _ = read_rows(tmp_path / "results" / "generation-tiny.csv")
    assert header["recording"] == "tiny"
    assert header["seed"] == "11"
    assert header["workers"] == "3"
    assert header["version.timeseries-zarr"] != "absent"


def test_the_csv_columns_are_the_generation_columns(
    tiny: RecordingSpec, tmp_path: Path
):
    run(tmp_path / "files", tmp_path / "results", "--formats", "edf")
    path = tmp_path / "results" / "generation-tiny.csv"
    body = [
        line
        for line in path.read_text().splitlines()
        if not line.startswith("# ")
    ]
    assert body[0].split(",") == list(schemas.GENERATION)


def test_the_command_writes_the_one_byte_object(
    tiny: RecordingSpec, tmp_path: Path
):
    root = tmp_path / "files"
    run(root, tmp_path / "results", "--formats", "edf")
    assert (root / schemas.ONE_BYTE_OBJECT).stat().st_size == 1


def test_the_formats_option_limits_the_files_in_the_order_given(
    tiny: RecordingSpec, tmp_path: Path
):
    root = tmp_path / "files"
    run(root, tmp_path / "results", "--formats", "tszarr,edf")
    _, rows = read_rows(tmp_path / "results" / "generation-tiny.csv")
    assert [row["format"] for row in rows] == ["tszarr", "edf"]
    written = {path.name for path in (root / "tiny").iterdir()}
    assert written == {schemas.FILE_NAMES["tszarr"], schemas.FILE_NAMES["edf"]}


def test_a_run_of_one_format_keeps_the_rows_of_the_others(
    tiny: RecordingSpec, tmp_path: Path
):
    root, results = tmp_path / "files", tmp_path / "results"
    run(root, results, "--formats", "edf,tszarr")
    _, first = read_rows(results / "generation-tiny.csv")
    run(root, results, "--formats", "tszarr-raw")
    _, rows = read_rows(results / "generation-tiny.csv")
    assert [row["format"] for row in rows] == ["edf", "tszarr", "tszarr-raw"]
    assert rows[:2] == first


def test_a_run_replaces_the_row_of_a_format_it_writes_again(
    tiny: RecordingSpec, tmp_path: Path
):
    root, results = tmp_path / "files", tmp_path / "results"
    run(root, results, "--formats", "edf,tszarr")
    run(root, results, "--formats", "edf", "--overwrite")
    _, rows = read_rows(results / "generation-tiny.csv")
    assert [row["format"] for row in rows] == ["tszarr", "edf"]


def test_a_recording_without_edf_writes_every_other_format_by_default(
    tmp_path: Path, tiny: RecordingSpec
):
    assert formats_of(replace(SPEC, edf_record_s=None)) == schemas.FORMATS[1:]
    root = tmp_path / "files"
    run(root, tmp_path / "results", recording="tiny-no-edf")
    _, rows = read_rows(tmp_path / "results" / "generation-tiny.csv")
    assert [row["format"] for row in rows] == list(schemas.FORMATS[1:])


def test_edf_is_refused_for_a_recording_without_edf(
    tiny: RecordingSpec, tmp_path: Path
):
    with pytest.raises(SystemExit) as error:
        run(
            tmp_path / "files",
            tmp_path / "results",
            "--formats",
            "edf",
            recording="tiny-no-edf",
        )
    assert error.value.code == 2
    assert not (tmp_path / "files").exists()


def test_an_unknown_format_is_refused(tiny: RecordingSpec, tmp_path: Path):
    with pytest.raises(SystemExit) as error:
        run(tmp_path / "files", tmp_path / "results", "--formats", "parquet")
    assert error.value.code == 2


def test_an_unknown_recording_is_refused(tiny: RecordingSpec, tmp_path: Path):
    with pytest.raises(SystemExit) as error:
        run(tmp_path / "files", tmp_path / "results", recording="R9")
    assert error.value.code == 2


def test_an_existing_file_is_not_overwritten_without_the_option(
    tiny: RecordingSpec, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    root, results = tmp_path / "files", tmp_path / "results"
    run(root, results, "--formats", "edf,tszarr")
    edf = root / schemas.relative_path("tiny", "edf")
    edf.write_bytes(b"sentinel")
    csv_text = (results / "generation-tiny.csv").read_text()
    capsys.readouterr()
    assert run(root, results, "--formats", "nwb-hdf5-time,edf") == 1
    assert "overwrite" in capsys.readouterr().err
    assert edf.read_bytes() == b"sentinel"
    assert not (root / schemas.relative_path("tiny", "nwb-hdf5-time")).exists()
    assert (results / "generation-tiny.csv").read_text() == csv_text


def test_the_overwrite_option_replaces_an_existing_file(
    tiny: RecordingSpec, tmp_path: Path
):
    root, results = tmp_path / "files", tmp_path / "results"
    run(root, results, "--formats", "edf,nwb-zarr-time")
    edf = root / schemas.relative_path("tiny", "edf")
    stale = root / schemas.relative_path("tiny", "nwb-zarr-time") / "stale"
    edf.write_bytes(b"sentinel")
    stale.write_bytes(b"stale")
    assert (
        run(root, results, "--formats", "edf,nwb-zarr-time", "--overwrite") == 0
    )
    assert edf.stat().st_size > len(b"sentinel")
    assert not stale.exists()


def test_the_command_runs_as_a_module_on_a_one_hour_cut(tmp_path: Path):
    root, results = tmp_path / "files", tmp_path / "results"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "generate",
            "--recording",
            "R1-1h",
            "--formats",
            "edf,nwb-hdf5-time,tszarr",
            "--root",
            str(root),
            "--results",
            str(results),
        ],
        capture_output=True,
        text=True,
        cwd=REPOSITORY,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    header, rows = read_rows(results / "generation-R1-1h.csv")
    assert header["recording"] == "R1-1h"
    assert header["seed"] == "1"
    assert [row["format"] for row in rows] == ["edf", "nwb-hdf5-time", "tszarr"]
    edf_bytes = 256 * 2 + 2 * 256 * 3600
    assert int(rows[0]["bytes"]) == edf_bytes
    assert (root / schemas.ONE_BYTE_OBJECT).exists()
