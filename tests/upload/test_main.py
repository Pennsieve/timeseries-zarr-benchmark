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


def pull_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str
) -> dict[str, object]:
    """Run the `pull` subcommand with `cli.pull` replaced, and return its keywords."""
    seen: dict[str, object] = {}

    def fake_pull(*args: object, **kwargs: object) -> None:
        seen.update(kwargs)

    monkeypatch.setattr(cli, "pull", fake_pull)
    status = cli.main(
        ["pull", "--root", str(tmp_path), "--bucket", "bucket", *extra]
    )
    assert status == 0
    return seen


def test_pull_subcommand_passes_the_listed_recordings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = pull_arguments(monkeypatch, tmp_path, "--recordings", "R1-1h,R4")
    assert seen["recordings"] == ["R1-1h", "R4"]


def test_pull_subcommand_syncs_everything_without_recordings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert pull_arguments(monkeypatch, tmp_path)["recordings"] is None
