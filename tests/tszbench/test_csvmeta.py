from pathlib import Path

import pytest

from tszbench import csvmeta
from tszbench.csvmeta import CsvWriter, read_header, read_rows, run_metadata

COLUMNS = ("recording", "bytes")


def test_read_rows_leaves_out_the_metadata_lines(tmp_path: Path) -> None:
    path = tmp_path / "part.csv"
    with CsvWriter(path, COLUMNS, {"date": "d"}) as writer:
        for index in range(3):
            writer.write({"recording": "R1", "bytes": index})
    rows = read_rows(path)
    assert rows == [{"recording": "R1", "bytes": str(i)} for i in range(3)]
    assert read_header(path) == {"date": "d"}


def test_the_metadata_records_the_ami_and_the_instance_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = {"instance-type": "c6i.8xlarge", "ami-id": "ami-0abc"}
    monkeypatch.setattr(csvmeta, "_instance_metadata", items.get)
    metadata = run_metadata()
    assert metadata["instance"] == "c6i.8xlarge"
    assert metadata["ami"] == "ami-0abc"


def test_the_metadata_names_the_ami_none_off_ec2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(csvmeta, "_instance_metadata", lambda item: None)
    metadata = run_metadata()
    assert (metadata["instance"], metadata["ami"]) == ("local", "none")


@pytest.mark.parametrize("name", ["httpx", "pandas", "matplotlib"])
def test_the_metadata_records_the_version_of_the_plot_and_client_libraries(
    name: str,
) -> None:
    assert run_metadata()[f"version.{name}"] != "absent"
