from pathlib import Path

import pytest

from tszbench import schemas
from upload import __main__ as cli


def test_storage_subcommand_writes_the_csv(root: Path, tmp_path: Path) -> None:
    status = cli.main(
        [
            "storage",
            "--root",
            str(root),
            "--recordings",
            "R9",
            "--results",
            str(tmp_path),
        ]
    )
    assert status == 0
    output = tmp_path / schemas.STORAGE_FILE
    assert output.read_text().count("R9,tszarr,") == 7


def test_subcommand_is_required() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_s3_subcommand_requires_a_bucket(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["s3", "--root", str(tmp_path), "--recording", "R1"])
