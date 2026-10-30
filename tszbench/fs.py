"""fsspec filesystem wrapper that counts requests and bytes and caps concurrency.

Every benchmark read goes through `CountingFileSystem`, so the store is the only
thing that changes between rows.
"""

import asyncio
import functools
import threading
import warnings
import weakref
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from s3fs import S3FileSystem
from zarr.storage import FsspecStore

MAX_IN_FLIGHT = 32
"""Requests in flight at once through one CountingFileSystem."""

CLOSE_TIMEOUT_SECONDS = 5.0
"""Time `CountingFileSystem.close` waits for the inner sessions and the loop."""


@dataclass(frozen=True)
class Counts:
    """Requests sent to the inner filesystem and the payload bytes returned.

    Attributes:
        requests: One per byte-range or whole-object read, per range of a
            multi-range read, and per metadata call (info, exists, ls, find).
        bytes: Payload bytes of the reads. Metadata calls add 0.
    """

    requests: int
    bytes: int

    def __sub__(self, other: "Counts") -> "Counts":
        """Return the counts accrued since other was taken."""
        return Counts(self.requests - other.requests, self.bytes - other.bytes)


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def _await[T](awaitable: Awaitable[T]) -> T:
    return await awaitable


async def _close_inner(inner: AbstractFileSystem) -> None:
    session = getattr(inner, "_session", None)
    if session is not None:
        await session.close()
    s3_creator = getattr(inner, "_s3creator", None)
    if s3_creator is not None:
        await s3_creator.__aexit__(None, None, None)


def _shutdown(
    loop: asyncio.AbstractEventLoop,
    thread: threading.Thread,
    executor: ThreadPoolExecutor,
    inner: AbstractFileSystem,
) -> None:
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(_close_inner(inner), loop)
        try:
            future.result(CLOSE_TIMEOUT_SECONDS)
        except Exception:
            future.cancel()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(CLOSE_TIMEOUT_SECONDS)
    if not loop.is_running() and not loop.is_closed():
        loop.close()
    executor.shutdown(wait=False)


class CountingFile(AbstractBufferedFile):  # type: ignore[misc]
    """Read-only file that fetches every byte range through its filesystem.

    There is no read-ahead and no cache, so each `read` that is not empty sends
    one request of exactly the bytes asked for.
    """

    def _fetch_range(self, start: int, end: int) -> Any:
        return self.fs.cat_file(self.path, start, end)

    def _initiate_upload(self) -> None:
        raise NotImplementedError("CountingFile is read-only")

    def _upload_chunk(self, final: bool = False) -> bool:
        raise NotImplementedError("CountingFile is read-only")


class CountingFileSystem(AsyncFileSystem):  # type: ignore[misc]
    """Pass reads to an inner filesystem, counting each request and its bytes.

    The wrapper is asynchronous over a synchronous or asynchronous inner
    filesystem, so zarr's FsspecStore takes it as is. At most `max_in_flight`
    requests run at once. `open` returns a read-only file whose reads go through
    the wrapper with no read-ahead and no cache, so h5py and the EDF range reader
    are counted and capped the same way as zarr.

    Counts are safe to read from any thread. A metadata call or a read made
    before `counts()` is taken is included in it.

    Every request runs on one private event loop that owns the inner
    filesystem, its HTTP session, and the cap. A coroutine awaited on any
    other loop, such as the loop of zarr, hands the request to that loop. A
    blocking call, such as `cat_file` or `open`, runs on the private loop and
    waits for the result. The cap therefore holds across all callers.
    """

    cachable = False

    def __init__(
        self, inner: AbstractFileSystem, max_in_flight: int = MAX_IN_FLIGHT
    ) -> None:
        """Wrap inner.

        Args:
            inner: A synchronous filesystem, or an asynchronous one made with
                `asynchronous=True`. A blocking asynchronous instance is
                rebuilt in asynchronous mode, without the instance cache.
            max_in_flight: Cap on requests in flight.

        Raises:
            ValueError: max_in_flight is below 1.
        """
        if max_in_flight < 1:
            raise ValueError(
                f"max_in_flight must be at least 1, got {max_in_flight}"
            )
        if inner.async_impl and not inner.asynchronous:
            inner = type(inner)(
                *inner.storage_args,
                **{
                    **inner.storage_options,
                    "asynchronous": True,
                    "skip_instance_cache": True,
                },
            )
        self._inner = inner
        self._inner_is_async = bool(inner.async_impl)
        self._max_in_flight = max_in_flight
        self._lock = threading.Lock()
        self._requests = 0
        self._bytes = 0
        self._worker_loop = asyncio.new_event_loop()
        self._semaphore = asyncio.Semaphore(max_in_flight)
        self._executor = ThreadPoolExecutor(
            max_workers=max_in_flight, thread_name_prefix="tszbench-fs"
        )
        thread = threading.Thread(
            target=_run_loop,
            args=(self._worker_loop,),
            name="tszbench-fs-loop",
            daemon=True,
        )
        thread.start()
        self._finalizer = weakref.finalize(
            self, _shutdown, self._worker_loop, thread, self._executor, inner
        )
        super().__init__(asynchronous=False, loop=self._worker_loop)

    @property
    def inner(self) -> AbstractFileSystem:
        """The wrapped filesystem."""
        return self._inner

    @property
    def max_in_flight(self) -> int:
        """Cap on requests in flight."""
        return self._max_in_flight

    def counts(self) -> Counts:
        """Return the totals since the wrapper was made."""
        with self._lock:
            return Counts(self._requests, self._bytes)

    def close(self) -> None:
        """Close the inner sessions and stop the private loop and threads."""
        self._finalizer()

    def _add(self, requests: int, payload_bytes: int) -> None:
        with self._lock:
            self._requests += requests
            self._bytes += payload_bytes

    async def _on_worker_loop[T](self, make: Callable[[], Awaitable[T]]) -> T:
        try:
            running: asyncio.AbstractEventLoop | None = (
                asyncio.get_running_loop()
            )
        except RuntimeError:
            running = None
        if running is self._worker_loop:
            return await make()
        future = asyncio.run_coroutine_threadsafe(
            _await(make()), self._worker_loop
        )
        return await asyncio.wrap_future(future)

    async def _call_inner(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if self._inner_is_async:
            return await getattr(self._inner, "_" + name)(*args, **kwargs)
        call = functools.partial(getattr(self._inner, name), *args, **kwargs)
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, call
        )

    async def _metadata(self, name: str, *args: Any, **kwargs: Any) -> Any:
        async with self._semaphore:
            self._add(1, 0)
            return await self._call_inner(name, *args, **kwargs)

    async def _read(
        self, path: str, start: int | None, end: int | None, **kwargs: Any
    ) -> Any:
        async with self._semaphore:
            self._add(1, 0)
            data = await self._call_inner(
                "cat_file", path, start=start, end=end, **kwargs
            )
            self._add(0, len(data))
            return data

    async def _cat_file(
        self,
        path: str,
        start: int | None = None,
        end: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Read one byte range, or the whole object, as one request.

        A negative start with no end reads that many bytes from the end.
        """
        return await self._on_worker_loop(
            lambda: self._read(path, start, end, **kwargs)
        )

    async def _cat_ranges(
        self,
        paths: list[str],
        starts: Sequence[int | None] | int | None,
        ends: Sequence[int | None] | int | None,
        max_gap: int | None = None,
        batch_size: int | None = None,
        on_error: str = "return",
        **kwargs: Any,
    ) -> list[Any]:
        """Read one byte range of each path, one request per range.

        With `on_error="return"` a failed range holds its exception.
        """
        if max_gap is not None:
            raise NotImplementedError("max_gap is not supported")
        starts_list = (
            list(starts)
            if isinstance(starts, Sequence)
            else [starts] * len(paths)
        )
        ends_list = (
            list(ends) if isinstance(ends, Sequence) else [ends] * len(paths)
        )
        if len(starts_list) != len(paths) or len(ends_list) != len(paths):
            raise ValueError("paths, starts, and ends differ in length")

        async def gather() -> list[Any]:
            return list(
                await asyncio.gather(
                    *(
                        self._read(p, s, e, **kwargs)
                        for p, s, e in zip(
                            paths, starts_list, ends_list, strict=True
                        )
                    ),
                    return_exceptions=True,
                )
            )

        out = await self._on_worker_loop(gather)
        if on_error != "return":
            for item in out:
                if isinstance(item, BaseException):
                    raise item
        return out

    async def _info(self, path: str, **kwargs: Any) -> Any:
        return await self._on_worker_loop(
            lambda: self._metadata("info", path, **kwargs)
        )

    async def _exists(self, path: str, **kwargs: Any) -> bool:
        if isinstance(self._inner, HTTPFileSystem):
            # The HTTP exists call sends a GET and drops the response, which
            # closes the keep-alive connection. A HEAD keeps it open.
            try:
                await self._info(path, **kwargs)
            except FileNotFoundError:
                return False
            return True
        return bool(
            await self._on_worker_loop(
                lambda: self._metadata("exists", path, **kwargs)
            )
        )

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        return await self._on_worker_loop(
            lambda: self._metadata("ls", path, detail=detail, **kwargs)
        )

    async def _find(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        **kwargs: Any,
    ) -> Any:
        return await self._on_worker_loop(
            lambda: self._metadata(
                "find", path, maxdepth=maxdepth, withdirs=withdirs, **kwargs
            )
        )

    async def _pipe_file(self, path: str, value: Any, **kwargs: Any) -> None:
        raise NotImplementedError("CountingFileSystem is read-only")

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        raise NotImplementedError("CountingFileSystem is read-only")

    async def _put_file(self, lpath: str, rpath: str, **kwargs: Any) -> None:
        raise NotImplementedError("CountingFileSystem is read-only")

    async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        raise NotImplementedError("CountingFileSystem is read-only")

    async def _mkdir(
        self, path: str, create_parents: bool = True, **kwargs: Any
    ) -> None:
        raise NotImplementedError("CountingFileSystem is read-only")

    async def _makedirs(self, path: str, exist_ok: bool = False) -> None:
        raise NotImplementedError("CountingFileSystem is read-only")

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | str | None = None,
        autocommit: bool = True,
        cache_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CountingFile:
        if mode != "rb":
            raise NotImplementedError(
                "CountingFileSystem opens files for reading only"
            )
        kwargs.pop("cache_type", None)
        return CountingFile(
            self,
            path,
            mode,
            block_size=block_size or "default",
            autocommit=autocommit,
            cache_type="none",
            cache_options=cache_options,
            **kwargs,
        )


@dataclass(frozen=True)
class StoreRoots:
    """Where every file lives in each store.

    Attributes:
        local: Directory that holds `<recording id>/<file name>`.
        http: Base URL of the file server, ending in `/`.
        s3: Bucket and prefix, as `bucket/prefix`, with no scheme.
        s3_endpoint_url: Endpoint of an S3-compatible server, or None for AWS.
    """

    local: Path
    http: str
    s3: str
    s3_endpoint_url: str | None = None


def make_filesystem(
    store: str, roots: StoreRoots, max_in_flight: int = MAX_IN_FLIGHT
) -> CountingFileSystem:
    """Return a new CountingFileSystem over the inner filesystem of store.

    Local reads use `LocalFileSystem`. HTTP reads use `HTTPFileSystem` over
    HTTP/1.1 keep-alive connections and ask for no compression. S3 reads use
    `S3FileSystem` with signed requests, with credentials from the environment
    or the instance role.

    Args:
        store: `local`, `http`, or `s3`.
        roots: Where the files live. Only `s3_endpoint_url` is read here.
        max_in_flight: Cap on requests in flight.

    Raises:
        ValueError: store is not one of the three.
    """
    inner: AbstractFileSystem
    if store == "local":
        inner = LocalFileSystem(skip_instance_cache=True)
    elif store == "http":
        inner = HTTPFileSystem(
            asynchronous=True,
            skip_instance_cache=True,
            client_kwargs={"headers": {"Accept-Encoding": "identity"}},
        )
    elif store == "s3":
        options: dict[str, Any] = {}
        if roots.s3_endpoint_url is not None:
            options["endpoint_url"] = roots.s3_endpoint_url
        inner = S3FileSystem(
            asynchronous=True,
            skip_instance_cache=True,
            anon=False,
            config_kwargs={"max_pool_connections": max_in_flight},
            **options,
        )
    else:
        raise ValueError(f"store must be local, http, or s3, got {store!r}")
    return CountingFileSystem(inner, max_in_flight)


def location(store: str, roots: StoreRoots, relative: str) -> str:
    """Return the path or URL of relative under the root of store.

    Raises:
        ValueError: store is not one of the three.
    """
    if store == "local":
        return str(roots.local / relative)
    if store == "http":
        return roots.http.rstrip("/") + "/" + relative
    if store == "s3":
        return roots.s3.rstrip("/") + "/" + relative
    raise ValueError(f"store must be local, http, or s3, got {store!r}")


def zarr_store(filesystem: CountingFileSystem, path: str) -> FsspecStore:
    """Return a read-only FsspecStore over filesystem at path."""
    with warnings.catch_warnings():
        # The wrapper blocks for synchronous calls and zarr awaits it, so it is
        # not in fsspec's asynchronous mode.
        warnings.filterwarnings("ignore", message=".*asynchronous=True.*")
        return FsspecStore(filesystem, read_only=True, path=path)
