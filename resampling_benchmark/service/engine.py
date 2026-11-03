"""Process pool that reads raw samples from S3 and reduces them to columns.

Every pool process owns one `CountingFileSystem` and one open bundle per
recording. A process runs up to `TASK_THREADS` tasks at once on its own
threads. Each task's S3 requests and bytes come from the counter of its
context, not from the totals of the process.
"""

import multiprocessing
import os
import threading
import time
from collections.abc import Sequence
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from multiprocessing.synchronize import Barrier as ProcessBarrier
from threading import BrokenBarrierError

import numpy as np
import numpy.typing as npt
from timeseries_zarr import Bundle, ContinuousChannel

from resampling_benchmark.service.groups import (
    Group,
    plan_groups,
    sample_bounds_us,
)
from resampling_benchmark.service.store import open_recording
from tszbench.fs import (
    CountingFileSystem,
    StoreRoots,
    count_task,
    make_filesystem,
)
from tszbench.minmax import column_minmax
from tszbench.recordings import sample_range

DEFAULT_PROCESSES = os.cpu_count() or 1
"""Processes of the pool, one per vCPU."""

TASK_THREADS = 1
"""Tasks a pool process runs at once."""

STARTUP_TIMEOUT_S = 900.0
"""Time every pool process has to open its recordings."""


@dataclass(frozen=True)
class RecordingInfo:
    """Shape of an open recording.

    Attributes:
        channels: Number of continuous channels.
        rate_hz: Sample rate of every channel.
        samples: Samples in every channel.
    """

    channels: int
    rate_hz: int
    samples: int


@dataclass(frozen=True)
class Task:
    """One group of columns of one channel.

    Attributes:
        recording: Recording id.
        channel: Index of the channel in key order.
        group: The samples and columns of the task.
    """

    recording: str
    channel: int
    group: Group


@dataclass(frozen=True)
class TaskResult:
    """Output and cost of one task.

    Attributes:
        channel: Index of the channel.
        first_column: Index of the first column of the group.
        values: Array of shape (rows, 2), float32.
        read_seconds: Seconds spent reading from S3 and decoding.
        reduce_seconds: Seconds spent trimming and reducing.
        requests: Requests the task sent to S3.
        bytes: Payload bytes the task received from S3.
        pid: Operating system id of the process that ran the task.
    """

    channel: int
    first_column: int
    values: npt.NDArray[np.float32]
    read_seconds: float
    reduce_seconds: float
    requests: int
    bytes: int
    pid: int


@dataclass(frozen=True)
class WindowPlan:
    """The tasks that answer one request.

    Attributes:
        info: Shape of the recording.
        tasks: One task per channel and group, in channel then group order.
        groups: Number of groups of each channel.
    """

    info: RecordingInfo
    tasks: Sequence[Task]
    groups: int


@dataclass
class _Handle:
    bundle: Bundle
    channels: list[ContinuousChannel]
    info: RecordingInfo


class _State:
    """Filesystem and open bundles that belong to one pool process."""

    def __init__(self, roots: StoreRoots, recordings: Sequence[str]) -> None:
        self.filesystem: CountingFileSystem = make_filesystem("s3", roots)
        self.handles: dict[str, _Handle] = {}
        for recording in recordings:
            bundle = open_recording(self.filesystem, roots, recording)
            channels = list(bundle.continuous.values())
            if not channels:
                raise ValueError(
                    f"recording {recording}: no continuous channel"
                )
            rates = {channel.rate_hz for channel in channels}
            lengths = {channel.num_samples for channel in channels}
            if len(rates) != 1 or len(lengths) != 1:
                raise ValueError(
                    f"recording {recording}: channels differ in rate or length"
                )
            rate = rates.pop()
            if rate != int(rate):
                raise ValueError(f"recording {recording}: rate {rate} Hz")
            self.handles[recording] = _Handle(
                bundle,
                channels,
                RecordingInfo(len(channels), int(rate), lengths.pop()),
            )

    def close(self) -> None:
        for handle in self.handles.values():
            handle.bundle.close()
        self.filesystem.close()


_STATE: _State | None = None
_STARTUP_ERROR: BaseException | None = None
_BARRIER: ProcessBarrier | None = None
_THREADS: ThreadPoolExecutor | None = None


def _initialize(
    roots: StoreRoots,
    recordings: Sequence[str],
    barrier: ProcessBarrier,
    task_threads: int,
) -> None:
    """Open every recording and start the task threads in a new pool process.

    A failure waits in `_STARTUP_ERROR` so that `_ready` reports it with its
    own message.
    """
    global _STATE, _STARTUP_ERROR, _BARRIER, _THREADS
    _BARRIER = barrier
    _THREADS = ThreadPoolExecutor(
        max_workers=task_threads, thread_name_prefix="task"
    )
    try:
        _STATE = _State(roots, recordings)
    except Exception as error:
        _STARTUP_ERROR = error


@dataclass(frozen=True)
class _Ready:
    pid: int
    infos: dict[str, RecordingInfo]
    bundles_opened: int


def _ready() -> _Ready:
    """Wait until every pool process holds its recordings, then describe them.

    Each of the `processes` calls blocks at the barrier, so every call runs in
    a different process.
    """
    barrier = _BARRIER
    if barrier is None:
        raise RuntimeError("pool process was not initialized")
    if _STATE is None:
        barrier.abort()
        raise _STARTUP_ERROR or RuntimeError("recordings did not open")
    barrier.wait(STARTUP_TIMEOUT_S)
    return _Ready(
        pid=os.getpid(),
        infos={name: handle.info for name, handle in _STATE.handles.items()},
        bundles_opened=len(_STATE.handles),
    )


def run_task(task: Task) -> TaskResult:
    """Read and reduce one task on the calling thread of a pool process."""
    if _STATE is None:
        raise RuntimeError("pool process was not initialized")
    state = _STATE
    handle = state.handles[task.recording]
    channel = handle.channels[task.channel]
    group = task.group
    lower_us, upper_us = sample_bounds_us(
        group.first, group.stop, handle.info.rate_hz
    )
    with count_task() as counter:
        started = time.perf_counter()
        samples = channel.read(lower_us, upper_us)
        read_seconds = time.perf_counter() - started
    spent = counter.counts()
    started = time.perf_counter()
    skip = group.first - samples.first
    values = np.asarray(
        samples.values[skip : skip + group.samples], dtype=np.float32
    )
    if values.shape[0] != group.samples:
        raise ValueError(
            f"channel {task.channel}: read {values.shape[0]} samples, "
            f"expected {group.samples}"
        )
    reduced = column_minmax(values, values, group.columns)
    reduce_seconds = time.perf_counter() - started
    return TaskResult(
        channel=task.channel,
        first_column=group.first_column,
        values=reduced,
        read_seconds=read_seconds,
        reduce_seconds=reduce_seconds,
        requests=spent.requests,
        bytes=spent.bytes,
        pid=os.getpid(),
    )


def run_batch(tasks: Sequence[Task]) -> list[TaskResult]:
    """Run the tasks of a batch at once on the task threads of the process.

    Returns:
        One result per task, in the order of tasks. Returns after every task
        of the batch has ended.

    Raises:
        Exception: The first error, in task order, of a task that failed.
    """
    if _THREADS is None:
        raise RuntimeError("pool process was not initialized")
    futures = [_THREADS.submit(run_task, task) for task in tasks]
    wait(futures)
    return [future.result() for future in futures]


def split_batches(tasks: Sequence[Task], size: int) -> list[list[Task]]:
    """Split tasks into consecutive batches of at most size tasks."""
    return [list(tasks[i : i + size]) for i in range(0, len(tasks), size)]


class Engine:
    """Reads and reduces the tasks of requests on one pool of processes.

    Attributes:
        processes: Processes in the pool.
        task_threads: Tasks a process runs at once, and the batch size.
        pids: Operating system ids of the pool processes, in ascending order.
        bundles_opened: Bundles each pool process opened at startup.
        executor: The pool.
    """

    def __init__(
        self,
        roots: StoreRoots,
        recordings: Sequence[str],
        processes: int = DEFAULT_PROCESSES,
        task_threads: int = TASK_THREADS,
    ) -> None:
        """Start the pool and open every recording in every process.

        Raises:
            ValueError: processes or task_threads is below 1.
            Exception: The error that stopped a recording from opening.
        """
        if processes < 1:
            raise ValueError(f"processes is {processes}: must be at least 1")
        if task_threads < 1:
            raise ValueError(
                f"task_threads is {task_threads}: must be at least 1"
            )
        self.processes = processes
        self.task_threads = task_threads
        self._recordings = tuple(recordings)
        self._lock = threading.Lock()
        self._in_flight = 0
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(processes)
        self.executor = ProcessPoolExecutor(
            max_workers=processes,
            mp_context=context,
            initializer=_initialize,
            initargs=(roots, self._recordings, barrier, task_threads),
        )
        futures = [self.executor.submit(_ready) for _ in range(processes)]
        readies: list[_Ready] = []
        errors: list[BaseException] = []
        for future in futures:
            try:
                readies.append(future.result(timeout=STARTUP_TIMEOUT_S + 60))
            except BaseException as error:
                errors.append(error)
        if errors:
            self.executor.shutdown(wait=False, cancel_futures=True)
            raise next(
                (e for e in errors if not isinstance(e, BrokenBarrierError)),
                errors[0],
            )
        self.pids = tuple(sorted(ready.pid for ready in readies))
        self.bundles_opened = readies[0].bundles_opened
        self._infos = readies[0].infos

    def info(self, recording: str) -> RecordingInfo:
        """Return the shape of recording.

        Raises:
            KeyError: The recording is not open.
        """
        return self._infos[recording]

    @property
    def recordings(self) -> tuple[str, ...]:
        """Ids of the open recordings."""
        return self._recordings

    @property
    def in_flight(self) -> int:
        """Tasks submitted and not yet finished or canceled."""
        with self._lock:
            return self._in_flight

    def plan(self, recording: str, start_us: int, end_us: int) -> WindowPlan:
        """Return the tasks that answer a window.

        Raises:
            KeyError: The recording is not open.
            ValueError: The window is empty or reaches outside the recording.
        """
        info = self.info(recording)
        first, stop = sample_range(start_us, end_us, info.rate_hz)
        if first < 0 or stop > info.samples or stop <= first:
            raise ValueError(
                f"window [{start_us}, {end_us}) us: holds samples "
                f"[{first}, {stop}), must be a non-empty range inside "
                f"[0, {info.samples})"
            )
        groups = plan_groups(first, stop - first)
        tasks = [
            Task(recording, channel, group)
            for channel in range(info.channels)
            for group in groups
        ]
        return WindowPlan(info, tasks, len(groups))

    def submit(self, tasks: Sequence[Task]) -> "list[Future[list[TaskResult]]]":
        """Queue tasks on the pool as batches of at most `task_threads` tasks.

        Each batch is one pool call. A task counts as in flight until its
        batch ends or is canceled.

        Returns:
            One future per batch, in task order. Each holds one result per
            task of its batch.
        """
        futures: list[Future[list[TaskResult]]] = []
        try:
            for batch in split_batches(tasks, self.task_threads):
                futures.append(self._submit_batch(batch))
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        return futures

    def _submit_batch(self, batch: list[Task]) -> "Future[list[TaskResult]]":
        size = len(batch)
        with self._lock:
            self._in_flight += size
        try:
            future = self.executor.submit(run_batch, batch)
        except BaseException:
            self._finished(size)
            raise
        future.add_done_callback(lambda _: self._finished(size))
        return future

    def _finished(self, size: int) -> None:
        with self._lock:
            self._in_flight -= size

    def close(self) -> None:
        """Stop the pool. Tasks that have not started are canceled."""
        self.executor.shutdown(wait=True, cancel_futures=True)


def assemble(
    results: Sequence[TaskResult], channels: int, groups: int
) -> npt.NDArray[np.float32]:
    """Join task results into one array of shape (channels, rows, 2).

    Args:
        results: Results of every task of the request, in any order.
        channels: Number of channels.
        groups: Number of groups of each channel.

    Raises:
        ValueError: A channel lacks a group or the channels differ in rows.
    """
    by_channel: list[list[TaskResult]] = [[] for _ in range(channels)]
    for result in results:
        by_channel[result.channel].append(result)
    joined = []
    for index, parts in enumerate(by_channel):
        if len(parts) != groups:
            raise ValueError(
                f"channel {index}: {len(parts)} groups, expected {groups}"
            )
        parts.sort(key=lambda part: part.first_column)
        joined.append(np.concatenate([part.values for part in parts]))
    return np.stack(joined)
