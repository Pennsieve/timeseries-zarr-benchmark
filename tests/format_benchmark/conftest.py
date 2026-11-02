"""A tiny recording in every format, served by every store."""

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from generate.formats import formats_of, write_format, write_one_byte_object
from tszbench.fs import StoreRoots
from tszbench.recordings import RECORDINGS, Gap, RecordingSpec

TINY_ID = "tiny"
TINY_SPEC = replace(
    RECORDINGS["R3"],
    id=TINY_ID,
    channels=6,
    rate_hz=512,
    duration_s=40,
    edf_record_s=1.0,
    gaps=(
        Gap(channel=0, start_s=5.0, duration_s=1.0),
        Gap(channel=0, start_s=20.25, duration_s=0.5),
    ),
)
"""Six channels at 512 Hz for 40 s. Windows of 1 s and 10 s fit, and a block of
`channels4` fits."""

TINY_WINDOWS_S = (1, 10)
"""Windows of `format_benchmark.positions.WINDOWS_S` that fit `TINY_SPEC`."""


@pytest.fixture(scope="session")
def tiny_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a store root that holds `TINY_SPEC` in every format."""
    root = tmp_path_factory.mktemp("format_benchmark-root")
    for file_format in formats_of(TINY_SPEC):
        write_format(TINY_SPEC, file_format, root)
    write_one_byte_object(root)
    return root


@pytest.fixture
def tiny_recording(monkeypatch: pytest.MonkeyPatch) -> RecordingSpec:
    """Make `format_benchmark.combinations.recording` resolve the id `tiny`."""
    import format_benchmark.combinations as combinations

    resolve = combinations.resolve

    def resolve_tiny(name: str) -> RecordingSpec:
        return TINY_SPEC if name == TINY_ID else resolve(name)

    monkeypatch.setattr(combinations, "resolve", resolve_tiny)
    return TINY_SPEC


@pytest.fixture(
    params=[
        "local",
        pytest.param("http", marks=pytest.mark.network),
        pytest.param("s3", marks=pytest.mark.network),
    ]
)
def store(request: pytest.FixtureRequest) -> str:
    """Return each store in turn."""
    return str(request.param)


@pytest.fixture
def roots(
    store: str, tiny_root: Path, request: pytest.FixtureRequest
) -> Iterator[StoreRoots]:
    """Return the roots of `tiny_root`, starting a server only for the store."""
    if store == "local":
        yield StoreRoots(local=tiny_root, http="", s3="")
        return
    factory: Callable[[Path], StoreRoots] = request.getfixturevalue(
        "store_roots"
    )
    yield factory(tiny_root)
