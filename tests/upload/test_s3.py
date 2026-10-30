import csv
from collections.abc import Sequence
from pathlib import Path

import pytest

from tszbench import schemas
from tszbench.csvmeta import read_header
from upload import s3, storage

BUCKET = "bench-bucket"


class FakeRunner:
    """Records each command and answers `aws --version`."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> str:
        self.commands.append(list(command))
        return (
            "aws-cli/2.17.0 Python/3.12.4 Linux/6.8.0\n"
            if (command[1] == "--version")
            else ""
        )


class StepClock:
    """Advances by a fixed step on every call."""

    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as file:
        return list(csv.DictReader(line for line in file if line[0] != "#"))


def test_s3_uri_joins_prefix_and_path() -> None:
    assert s3.s3_uri("b", "run1/", "R1/edf.edf") == "s3://b/run1/R1/edf.edf"
    assert s3.s3_uri("b", "", "R1/edf.edf") == "s3://b/R1/edf.edf"


def test_copy_command_of_a_directory_is_recursive() -> None:
    command = s3.copy_command(
        Path("/d/R1/tszarr.zarr"), "s3://b/R1/tszarr.zarr", recursive=True
    )
    assert command == [
        "aws",
        "s3",
        "cp",
        "--recursive",
        "/d/R1/tszarr.zarr",
        "s3://b/R1/tszarr.zarr",
    ]


def test_copy_command_of_a_file_is_plain() -> None:
    command = s3.copy_command(
        Path("/d/R1/edf.edf"), "s3://b/R1/edf.edf", recursive=False
    )
    assert command == ["aws", "s3", "cp", "/d/R1/edf.edf", "s3://b/R1/edf.edf"]


def test_region_option_ends_the_command() -> None:
    command = s3.sync_command("s3://b/", Path("/d"), region="us-east-1")
    assert command == [
        "aws",
        "s3",
        "sync",
        "s3://b/",
        "/d",
        "--region",
        "us-east-1",
    ]


def test_upload_runs_one_copy_per_present_format(
    root: Path, tmp_path: Path
) -> None:
    runner = FakeRunner()
    s3.upload_recording(
        root,
        "R9",
        BUCKET,
        "",
        tmp_path / "up.csv",
        run=runner,
        clock=StepClock(2.0),
    )
    copies = [c for c in runner.commands if c[:3] == ["aws", "s3", "cp"]]
    targets = {c[-1]: "--recursive" in c for c in copies}
    for file_format in schemas.FORMATS:
        uri = f"s3://{BUCKET}/{schemas.relative_path('R9', file_format)}"
        assert targets[uri] == ("zarr" in file_format)
    assert f"s3://{BUCKET}/{schemas.ONE_BYTE_OBJECT}" in targets


def test_upload_uploads_the_one_byte_object_first(
    root: Path, tmp_path: Path
) -> None:
    runner = FakeRunner()
    s3.upload_recording(
        root,
        "R9",
        BUCKET,
        "p",
        tmp_path / "up.csv",
        run=runner,
        clock=StepClock(1.0),
    )
    copies = [c for c in runner.commands if c[:3] == ["aws", "s3", "cp"]]
    assert copies[0][-1] == f"s3://{BUCKET}/p/{schemas.ONE_BYTE_OBJECT}"


def test_upload_rows_match_schema_and_sizes(root: Path, tmp_path: Path) -> None:
    output = tmp_path / "up.csv"
    s3.upload_recording(
        root, "R9", BUCKET, "", output, run=FakeRunner(), clock=StepClock(2.0)
    )
    rows = _read_rows(output)
    assert list(rows[0]) == list(schemas.UPLOAD)
    assert [r["format"] for r in rows] == list(schemas.FORMATS)
    for row in rows:
        size, objects = storage.measure(
            root / schemas.relative_path("R9", row["format"])
        )
        assert int(row["bytes"]) == size
        assert int(row["objects"]) == objects
        assert float(row["seconds"]) == 2.0
        assert float(row["bytes_per_second"]) == size / 2.0


def test_upload_header_records_the_aws_cli_version(
    root: Path, tmp_path: Path
) -> None:
    output = tmp_path / "up.csv"
    s3.upload_recording(
        root, "R9", BUCKET, "p", output, run=FakeRunner(), clock=StepClock(1.0)
    )
    header = read_header(output)
    assert header["aws_cli"].startswith("aws-cli/2.17.0")
    assert header["bucket"] == BUCKET
    assert header["prefix"] == "p"


def test_upload_skips_absent_formats(root: Path, tmp_path: Path) -> None:
    (root / "R9" / schemas.FILE_NAMES["edf"]).unlink()
    output = tmp_path / "up.csv"
    s3.upload_recording(
        root, "R9", BUCKET, "", output, run=FakeRunner(), clock=StepClock(1.0)
    )
    assert "edf" not in {r["format"] for r in _read_rows(output)}


def test_upload_of_a_missing_recording_raises(tmp_path: Path) -> None:
    runner = FakeRunner()
    with pytest.raises(FileNotFoundError):
        s3.upload_recording(
            tmp_path, "R1", BUCKET, "", tmp_path / "up.csv", run=runner
        )
    assert runner.commands == []


def test_one_byte_upload_sends_a_single_byte(tmp_path: Path) -> None:
    sizes: list[int] = []

    def run(command: Sequence[str]) -> str:
        sizes.append(Path(command[3]).stat().st_size)
        return ""

    s3.upload_one_byte(BUCKET, "", run=run)
    assert sizes == [1]


def test_pull_syncs_the_bucket_to_the_root(tmp_path: Path) -> None:
    runner = FakeRunner()
    s3.pull(tmp_path / "data", BUCKET, "run1", run=runner)
    assert runner.commands == [
        ["aws", "s3", "sync", f"s3://{BUCKET}/run1/", str(tmp_path / "data")]
    ]
    assert (tmp_path / "data").is_dir()


def test_run_command_reports_the_standard_error() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        s3.run_command(["sh", "-c", "echo boom >&2; exit 3"])
    assert s3.run_command(["echo", "hi"]) == "hi\n"


def test_endpoint_option_follows_the_region() -> None:
    command = s3.copy_command(
        Path("/d/f"),
        "s3://b/f",
        recursive=False,
        region="us-east-1",
        endpoint_url="http://127.0.0.1:5000",
    )
    assert command[-4:] == [
        "--region",
        "us-east-1",
        "--endpoint-url",
        "http://127.0.0.1:5000",
    ]
