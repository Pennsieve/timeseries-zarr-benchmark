import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tszbench.rangeserver import serve_directory

pytestmark = pytest.mark.network


def fetch(url: str, headers: dict[str, str]) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return int(response.status), bytes(response.read())
    except urllib.error.HTTPError as error:
        return error.code, b""


def test_a_byte_range_returns_those_bytes(tmp_path: Path) -> None:
    (tmp_path / "file").write_bytes(bytes(range(10)))
    running = serve_directory(tmp_path)
    try:
        status, body = fetch(running.url + "file", {"Range": "bytes=2-4"})
        assert (status, body) == (206, bytes([2, 3, 4]))
        status, body = fetch(running.url + "file", {"Range": "bytes=-3"})
        assert (status, body) == (206, bytes([7, 8, 9]))
        status, _ = fetch(running.url + "file", {"Range": "bytes=10-"})
        assert status == 416
    finally:
        running.stop()


def test_a_path_outside_the_directory_is_not_found(tmp_path: Path) -> None:
    served = tmp_path / "served"
    served.mkdir()
    (tmp_path / "secret").write_bytes(b"x")
    running = serve_directory(served)
    try:
        status, _ = fetch(running.url + "..%2Fsecret", {})
        assert status == 404
    finally:
        running.stop()
