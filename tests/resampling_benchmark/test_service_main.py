import resource

import pytest

from resampling_benchmark.service.__main__ import (
    OPEN_FILES_MINIMUM,
    raise_open_file_limit,
)


class FakeLimits:
    """Stands in for the open file limit of the process."""

    def __init__(self, soft: int, hard: int) -> None:
        self.limits = (soft, hard)

    def getrlimit(self, which: int) -> tuple[int, int]:
        assert which == resource.RLIMIT_NOFILE
        return self.limits

    def setrlimit(self, which: int, limits: tuple[int, int]) -> None:
        assert which == resource.RLIMIT_NOFILE
        self.limits = limits


@pytest.fixture
def limits(monkeypatch: pytest.MonkeyPatch) -> FakeLimits:
    fake = FakeLimits(1024, resource.RLIM_INFINITY)
    monkeypatch.setattr(resource, "getrlimit", fake.getrlimit)
    monkeypatch.setattr(resource, "setrlimit", fake.setrlimit)
    return fake


def test_a_soft_limit_of_1024_is_raised_to_the_minimum(
    limits: FakeLimits,
) -> None:
    assert raise_open_file_limit() == OPEN_FILES_MINIMUM
    assert limits.limits == (OPEN_FILES_MINIMUM, resource.RLIM_INFINITY)


def test_the_limit_stops_at_the_hard_limit(limits: FakeLimits) -> None:
    limits.limits = (1024, 4096)
    assert raise_open_file_limit() == 4096
    assert limits.limits == (4096, 4096)


def test_a_soft_limit_above_the_minimum_stays(limits: FakeLimits) -> None:
    limits.limits = (OPEN_FILES_MINIMUM * 2, OPEN_FILES_MINIMUM * 4)
    assert raise_open_file_limit() == OPEN_FILES_MINIMUM * 2
    assert limits.limits == (OPEN_FILES_MINIMUM * 2, OPEN_FILES_MINIMUM * 4)


def test_an_unlimited_soft_limit_stays(limits: FakeLimits) -> None:
    limits.limits = (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    assert raise_open_file_limit() == resource.RLIM_INFINITY
    assert limits.limits == (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
