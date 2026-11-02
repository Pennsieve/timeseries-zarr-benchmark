import asyncio
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import zarr
from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.local import LocalFileSystem

from tszbench.fs import (
    CountingFileSystem,
    Counts,
    StoreRoots,
    TaskCounter,
    count_task,
    location,
    make_filesystem,
    zarr_store,
)
from tszbench.schemas import STORES

BLOB = "rec/blob.bin"
BLOB_SIZE = 100_000


class SleepingFileSystem(AsyncFileSystem):  # type: ignore[misc]
    """Asynchronous inner filesystem that records its peak concurrency."""

    cachable = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(asynchronous=True, **kwargs)
        self.current = 0
        self.peak = 0

    async def _sleep(self) -> None:
        self.current += 1
        self.peak = max(self.peak, self.current)
        try:
            await asyncio.sleep(0.01)
        finally:
            self.current -= 1

    async def _cat_file(
        self,
        path: str,
        start: int | None = None,
        end: int | None = None,
        **kwargs: Any,
    ) -> bytes:
        await self._sleep()
        return b"x" * ((end or 0) - (start or 0))

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        await self._sleep()
        return {"name": path, "size": 10, "type": "file"}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    directory = tmp_path / "tree"
    (directory / "rec").mkdir(parents=True)
    rng = np.random.default_rng(0)
    (directory / BLOB).write_bytes(rng.bytes(BLOB_SIZE))
    (directory / "rec" / "small.txt").write_bytes(b"hello")
    return directory


@pytest.fixture
def roots(tree: Path, store_roots: Callable[[Path], StoreRoots]) -> StoreRoots:
    return store_roots(tree)


def blob_bytes(tree: Path) -> bytes:
    return (tree / BLOB).read_bytes()


def test_counts_subtract_field_by_field() -> None:
    assert Counts(10, 500) - Counts(4, 120) == Counts(6, 380)


def test_make_filesystem_rejects_an_unknown_store(tmp_path: Path) -> None:
    roots = StoreRoots(tmp_path, "http://x/", "b/p")
    with pytest.raises(ValueError, match="local, http, or s3"):
        make_filesystem("gcs", roots)
    with pytest.raises(ValueError, match="local, http, or s3"):
        location("gcs", roots, "a")


def test_location_joins_the_relative_path_under_each_root(
    tmp_path: Path,
) -> None:
    roots = StoreRoots(tmp_path, "http://h:1/", "bucket/prefix")
    assert location("local", roots, "r/f") == str(tmp_path / "r" / "f")
    assert location("http", roots, "r/f") == "http://h:1/r/f"
    assert location("s3", roots, "r/f") == "bucket/prefix/r/f"


def test_wrapper_rejects_a_cap_below_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        CountingFileSystem(LocalFileSystem(), 0)


@pytest.mark.network
@pytest.mark.parametrize("store", STORES)
def test_cat_file_counts_one_request_and_its_payload(
    store: str, roots: StoreRoots, tree: Path
) -> None:
    fs = make_filesystem(store, roots)
    path = location(store, roots, BLOB)
    expected = blob_bytes(tree)
    before = fs.counts()
    assert fs.cat_file(path, 10, 110) == expected[10:110]
    assert fs.counts() - before == Counts(1, 100)
    assert fs.cat_file(path, -50) == expected[-50:]
    assert fs.counts() - before == Counts(2, 150)
    assert fs.cat_file(path, 90_000, None) == expected[90_000:]
    assert fs.counts() - before == Counts(3, 150 + 10_000)
    assert fs.cat_file(path) == expected
    assert fs.counts() - before == Counts(4, 150 + 10_000 + BLOB_SIZE)
    fs.close()


@pytest.mark.network
@pytest.mark.parametrize("store", STORES)
def test_cat_ranges_counts_one_request_per_range(
    store: str, roots: StoreRoots, tree: Path
) -> None:
    fs = make_filesystem(store, roots)
    path = location(store, roots, BLOB)
    small = location(store, roots, "rec/small.txt")
    expected = blob_bytes(tree)
    out = fs.cat_ranges(
        [path, path, small], [0, 500, 1], [100, 700, 4], on_error="return"
    )
    assert out == [expected[0:100], expected[500:700], b"ell"]
    assert fs.counts() == Counts(3, 100 + 200 + 3)
    fs.close()


@pytest.mark.network
@pytest.mark.parametrize("store", STORES)
def test_cat_ranges_returns_the_error_of_a_missing_object(
    store: str, roots: StoreRoots
) -> None:
    fs = make_filesystem(store, roots)
    good = location(store, roots, "rec/small.txt")
    missing = location(store, roots, "rec/missing.bin")
    out = fs.cat_ranges([good, missing], [0, 0], [5, 5], on_error="return")
    assert out[0] == b"hello"
    assert isinstance(out[1], FileNotFoundError)
    assert fs.counts().requests == 2
    with pytest.raises(FileNotFoundError):
        fs.cat_ranges([missing], [0], [5], on_error="raise")
    fs.close()


@pytest.mark.network
@pytest.mark.parametrize("store", STORES)
def test_info_and_exists_count_one_request_and_no_bytes(
    store: str, roots: StoreRoots
) -> None:
    fs = make_filesystem(store, roots)
    path = location(store, roots, BLOB)
    assert fs.info(path)["size"] == BLOB_SIZE
    assert fs.counts() == Counts(1, 0)
    assert fs.exists(path) is True
    assert fs.counts() == Counts(2, 0)
    assert fs.exists(location(store, roots, "rec/missing.bin")) is False
    assert fs.counts() == Counts(3, 0)
    fs.close()


@pytest.mark.network
@pytest.mark.parametrize("store", STORES)
def test_ls_lists_the_objects_of_a_directory(
    store: str, roots: StoreRoots
) -> None:
    fs = make_filesystem(store, roots)
    directory = location(store, roots, "rec")
    names = sorted(
        entry.rstrip("/").rsplit("/", 1)[-1]
        for entry in fs.ls(directory, detail=False)
    )
    assert names == ["blob.bin", "small.txt"]
    assert fs.counts().requests >= 1
    assert fs.counts().bytes == 0
    fs.close()


@pytest.mark.network
@pytest.mark.parametrize("store", STORES)
def test_find_lists_every_object_under_a_path(
    store: str, roots: StoreRoots
) -> None:
    fs = make_filesystem(store, roots)
    found = fs.find(location(store, roots, "rec"), detail=False)
    assert sorted(name.rsplit("/", 1)[-1] for name in found) == [
        "blob.bin",
        "small.txt",
    ]
    fs.close()


@pytest.mark.network
@pytest.mark.parametrize("store", STORES)
def test_open_reads_exactly_the_bytes_asked_for(
    store: str, roots: StoreRoots, tree: Path
) -> None:
    fs = make_filesystem(store, roots)
    expected = blob_bytes(tree)
    with fs.open(location(store, roots, BLOB), "rb") as handle:
        opened = fs.counts()
        assert opened == Counts(1, 0)
        handle.seek(1_000)
        assert handle.read(300) == expected[1_000:1_300]
        assert fs.counts() - opened == Counts(1, 300)
        handle.seek(-20, 2)
        assert handle.read() == expected[-20:]
        assert fs.counts() - opened == Counts(2, 320)
    fs.close()


def test_open_refuses_to_write(tmp_path: Path) -> None:
    fs = CountingFileSystem(LocalFileSystem(skip_instance_cache=True))
    with pytest.raises(NotImplementedError):
        fs.open(str(tmp_path / "out.bin"), "wb")
    fs.close()


@pytest.mark.network
@pytest.mark.parametrize("store", STORES)
def test_sharded_zarr_array_reads_back_through_the_store(
    store: str, tmp_path: Path, store_roots: Callable[[Path], StoreRoots]
) -> None:
    directory = tmp_path / "zarr-tree"
    (directory / "rec").mkdir(parents=True)
    values = np.arange(4_000, dtype=np.float32).reshape(1_000, 4)
    array = zarr.create_array(
        store=str(directory / "rec" / "a.zarr"),
        shape=values.shape,
        chunks=(100, 4),
        shards=(500, 4),
        dtype="float32",
    )
    array[:] = values
    roots = store_roots(directory)
    fs = make_filesystem(store, roots)
    opened = zarr.open_array(
        zarr_store(fs, location(store, roots, "rec/a.zarr")), mode="r"
    )
    assert opened.shards == (500, 4)
    before = fs.counts()
    np.testing.assert_array_equal(opened[120:260, :], values[120:260, :])
    delta = fs.counts() - before
    assert delta.requests > 0
    assert delta.bytes > 0
    assert zarr_store(fs, "x").read_only
    fs.close()


@pytest.mark.network
@pytest.mark.parametrize("store", STORES)
def test_h5py_reads_a_dataset_through_open(
    store: str, tmp_path: Path, store_roots: Callable[[Path], StoreRoots]
) -> None:
    directory = tmp_path / "h5-tree"
    (directory / "rec").mkdir(parents=True)
    values = np.arange(40_000, dtype=np.float32).reshape(10_000, 4)
    with h5py.File(directory / "rec" / "d.h5", "w") as handle:
        handle.create_dataset("data", data=values, chunks=(500, 4))
    size = (directory / "rec" / "d.h5").stat().st_size
    roots = store_roots(directory)
    fs = make_filesystem(store, roots)
    with h5py.File(
        fs.open(location(store, roots, "rec/d.h5"), "rb"), "r"
    ) as handle:
        before = fs.counts()
        np.testing.assert_array_equal(
            handle["data"][2_000:2_100], values[2_000:2_100]
        )
        delta = fs.counts() - before
    assert delta.requests > 0
    assert 0 < delta.bytes < size
    fs.close()


@pytest.mark.network
@pytest.mark.parametrize("store", ["local", "http"])
def test_pynwb_reads_an_electrical_series_through_open(
    store: str, tmp_path: Path, store_roots: Callable[[Path], StoreRoots]
) -> None:
    from datetime import UTC, datetime

    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.ecephys import ElectricalSeries

    directory = tmp_path / "nwb-tree"
    (directory / "rec").mkdir(parents=True)
    values = np.arange(8_000, dtype=np.float32).reshape(2_000, 4)
    nwbfile = NWBFile(
        session_description="test",
        identifier="id",
        session_start_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    device = nwbfile.create_device(name="device")
    group = nwbfile.create_electrode_group(
        name="group", description="d", location="l", device=device
    )
    for _ in range(4):
        nwbfile.add_electrode(location="l", group=group, group_name="group")
    region = nwbfile.create_electrode_table_region(
        region=[0, 1, 2, 3], description="all"
    )
    nwbfile.add_acquisition(
        ElectricalSeries(
            name="series",
            data=values,
            electrodes=region,
            starting_time=0.0,
            rate=100.0,
        )
    )
    with NWBHDF5IO(str(directory / "rec" / "f.nwb"), "w") as io:
        io.write(nwbfile)
    roots = store_roots(directory)
    fs = make_filesystem(store, roots)
    handle = h5py.File(fs.open(location(store, roots, "rec/f.nwb"), "rb"), "r")
    with NWBHDF5IO(file=handle, mode="r") as io:
        read = io.read()
        before = fs.counts()
        np.testing.assert_array_equal(
            read.acquisition["series"].data[100:200], values[100:200]
        )
        assert (fs.counts() - before).requests > 0
    fs.close()


def test_cap_holds_under_200_concurrent_coroutines() -> None:
    inner = SleepingFileSystem()
    fs = CountingFileSystem(inner, max_in_flight=8)

    async def main() -> list[bytes]:
        return list(
            await asyncio.gather(
                *(fs._cat_file("p", 0, 10) for _ in range(200))
            )
        )

    out = asyncio.run(main())
    assert out == [b"x" * 10] * 200
    assert inner.peak == 8
    assert fs.counts() == Counts(200, 2_000)
    fs.close()


def test_cap_holds_across_two_event_loops_and_blocking_threads() -> None:
    inner = SleepingFileSystem()
    fs = CountingFileSystem(inner, max_in_flight=8)
    errors: list[BaseException] = []

    def on_loop() -> None:
        try:

            async def main() -> None:
                await asyncio.gather(
                    *(fs._cat_file("p", 0, 1) for _ in range(100))
                )

            asyncio.run(main())
        except BaseException as error:
            errors.append(error)

    def blocking() -> None:
        try:
            for _ in range(10):
                fs.cat_ranges(["p"] * 10, [0] * 10, [1] * 10)
                fs.info("p")
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=on_loop) for _ in range(2)]
    threads += [threading.Thread(target=blocking) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert inner.peak == 8
    assert fs.counts().requests == 2 * 100 + 4 * 10 * 11
    fs.close()


def test_cap_applies_to_a_synchronous_inner_filesystem(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"a" * 100)
    peak = 0
    current = 0
    guard = threading.Lock()

    class Slow(LocalFileSystem):  # type: ignore[misc]
        cachable = False

        def cat_file(
            self, path: str, start: Any = None, end: Any = None, **kw: Any
        ) -> bytes:
            nonlocal peak, current
            with guard:
                current += 1
                peak = max(peak, current)
            threading.Event().wait(0.01)
            with guard:
                current -= 1
            return super().cat_file(path, start, end, **kw)  # type: ignore[no-any-return]

    fs = CountingFileSystem(Slow(), max_in_flight=4)
    out = fs.cat_ranges([str(path)] * 60, [0] * 60, [10] * 60)
    assert out == [b"a" * 10] * 60
    assert peak == 4
    assert fs.counts() == Counts(60, 600)
    fs.close()


def test_counts_are_consistent_under_concurrent_threads(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(bytes(range(256)) * 40)
    fs = CountingFileSystem(LocalFileSystem(skip_instance_cache=True))
    errors: list[BaseException] = []

    def worker(seed: int) -> None:
        try:
            for index in range(25):
                size = 1 + (seed * 25 + index) % 50
                assert len(fs.cat_file(str(path), 0, size)) == size
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    expected = sum(1 + (n * 25 + i) % 50 for n in range(8) for i in range(25))
    assert fs.counts() == Counts(200, expected)
    fs.close()


THREADS = 16


def run_threads(work: Callable[[int], None]) -> None:
    """Run work(index) on THREADS threads that start together."""
    barrier = threading.Barrier(THREADS)
    errors: list[BaseException] = []

    def target(index: int) -> None:
        try:
            barrier.wait()
            work(index)
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=target, args=(index,))
        for index in range(THREADS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def sum_counts(counts: Iterable[Counts]) -> Counts:
    items = list(counts)
    return Counts(sum(c.requests for c in items), sum(c.bytes for c in items))


def check_threads_count_their_own_ranges(
    fs: CountingFileSystem, path: str, content: bytes
) -> None:
    """Read distinct ranges on 16 threads and check each counter."""
    counters: dict[int, TaskCounter] = {}
    expected = {
        index: Counts(index + 1, (index + 1) * (100 + index))
        for index in range(THREADS)
    }
    before = fs.counts()

    def work(index: int) -> None:
        size = 100 + index
        with count_task() as counter:
            for step in range(index + 1):
                start = 1_000 * index + 10 * step
                data = fs.cat_file(path, start, start + size)
                assert data == content[start : start + size]
            counters[index] = counter

    run_threads(work)
    assert {i: c.counts() for i, c in counters.items()} == expected
    assert fs.counts() - before == sum_counts(expected.values())


def make_sharded_tree(directory: Path) -> np.ndarray:
    (directory / "rec").mkdir(parents=True)
    values = np.arange(64_000, dtype=np.float32).reshape(16_000, 4)
    array = zarr.create_array(
        store=str(directory / "rec" / "a.zarr"),
        shape=values.shape,
        chunks=(100, 4),
        shards=(1_000, 4),
        dtype="float32",
    )
    array[:] = values
    return values


def check_threads_count_their_own_zarr_slices(
    fs: CountingFileSystem, path: str, values: np.ndarray
) -> None:
    """Slice a sharded array on 16 threads and check each counter."""
    opened = zarr.open_array(zarr_store(fs, path), mode="r")

    def rows(index: int) -> slice:
        start = 900 * index + 37
        return slice(start, start + 150 + 40 * index)

    expected: dict[int, Counts] = {}
    for index in range(THREADS):
        with count_task() as counter:
            np.testing.assert_array_equal(
                opened[rows(index), :], values[rows(index), :]
            )
        expected[index] = counter.counts()
        assert expected[index].requests > 0
        assert expected[index].bytes > 0
    assert len({c.bytes for c in expected.values()}) > 1

    counters: dict[int, TaskCounter] = {}
    before = fs.counts()

    def work(index: int) -> None:
        with count_task() as counter:
            np.testing.assert_array_equal(
                opened[rows(index), :], values[rows(index), :]
            )
            counters[index] = counter

    run_threads(work)
    assert {i: c.counts() for i, c in counters.items()} == expected
    assert fs.counts() - before == sum_counts(expected.values())


def test_task_counters_hold_the_requests_of_their_own_threads(
    tmp_path: Path,
) -> None:
    content = np.random.default_rng(1).bytes(40_000)
    path = tmp_path / "f.bin"
    path.write_bytes(content)
    fs = CountingFileSystem(LocalFileSystem(skip_instance_cache=True))
    check_threads_count_their_own_ranges(fs, str(path), content)
    fs.close()


def test_task_counters_hold_the_requests_of_their_own_zarr_slices(
    tmp_path: Path,
) -> None:
    values = make_sharded_tree(tmp_path)
    fs = CountingFileSystem(LocalFileSystem(skip_instance_cache=True))
    check_threads_count_their_own_zarr_slices(
        fs, str(tmp_path / "rec" / "a.zarr"), values
    )
    fs.close()


@pytest.mark.network
def test_task_counters_hold_the_requests_of_their_own_threads_on_s3(
    tmp_path: Path, store_roots: Callable[[Path], StoreRoots]
) -> None:
    content = np.random.default_rng(1).bytes(40_000)
    (tmp_path / "rec").mkdir()
    (tmp_path / "rec" / "f.bin").write_bytes(content)
    values = make_sharded_tree(tmp_path / "zarr")
    roots = store_roots(tmp_path)
    fs = make_filesystem("s3", roots)
    check_threads_count_their_own_ranges(
        fs, location("s3", roots, "rec/f.bin"), content
    )
    check_threads_count_their_own_zarr_slices(
        fs, location("s3", roots, "zarr/rec/a.zarr"), values
    )
    fs.close()


def test_a_read_with_no_counter_adds_to_the_totals_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"a" * 100)
    fs = CountingFileSystem(LocalFileSystem(skip_instance_cache=True))
    with count_task() as counter:
        fs.cat_file(str(path), 0, 10)
    fs.cat_file(str(path), 0, 20)
    assert counter.counts() == Counts(1, 10)
    assert fs.counts() == Counts(2, 30)
    fs.close()
