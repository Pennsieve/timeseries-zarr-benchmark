"""Host monitor of the resampling benchmark: `python -m resampling_benchmark.load.monitor`.

Samples CPU use and network throughput once per second from `/proc`. Linux
only.
"""

import argparse
import signal
import socket
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from tszbench.csvmeta import CsvWriter, run_metadata
from tszbench.schemas import MONITOR

ROLES = ("server", "generator")
PROC_STAT = Path("/proc/stat")
PROC_NET_DEV = Path("/proc/net/dev")
SAMPLE_INTERVAL_S = 1.0
LOOPBACK = "lo"
CPU_TOTAL_FIELDS = 8
"""Leading fields of the `cpu` line that add up to busy plus idle time. The
guest fields that follow are already part of `user` and `nice`."""
IDLE_FIELDS = (3, 4)
"""Fields of the `cpu` line that count idle time: `idle` and `iowait`."""
RX_BYTES_FIELD = 0
TX_BYTES_FIELD = 8


@dataclass(frozen=True)
class Reading:
    """Cumulative counters at one moment.

    Attributes:
        total_ticks: Clock ticks the CPUs spent in every state.
        idle_ticks: Clock ticks the CPUs spent idle.
        rx_bytes: Bytes received on every interface except loopback.
        tx_bytes: Bytes sent on every interface except loopback.
    """

    total_ticks: int
    idle_ticks: int
    rx_bytes: int
    tx_bytes: int


def read_cpu(path: Path) -> tuple[int, int]:
    """Return the total and idle clock ticks of the aggregate `cpu` line.

    Raises:
        ValueError: The file has no aggregate `cpu` line.
    """
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "cpu":
            ticks = [int(value) for value in parts[1:]]
            total = sum(ticks[:CPU_TOTAL_FIELDS])
            idle = sum(ticks[index] for index in IDLE_FIELDS)
            return total, idle
    raise ValueError(f"{path}: no aggregate cpu line")


def read_network(path: Path) -> tuple[int, int]:
    """Return the received and sent bytes of every interface but loopback."""
    received = sent = 0
    for line in path.read_text().splitlines()[2:]:
        name, _, rest = line.partition(":")
        if not rest or name.strip() == LOOPBACK:
            continue
        fields = rest.split()
        received += int(fields[RX_BYTES_FIELD])
        sent += int(fields[TX_BYTES_FIELD])
    return received, sent


def read_counters(stat: Path = PROC_STAT, net: Path = PROC_NET_DEV) -> Reading:
    """Read the CPU and network counters."""
    total, idle = read_cpu(stat)
    received, sent = read_network(net)
    return Reading(total, idle, received, sent)


def rates(
    before: Reading, after: Reading, seconds: float
) -> tuple[float, float, float]:
    """Return CPU use in percent and receive and send bytes per second.

    Raises:
        ValueError: seconds is not positive.
    """
    if seconds <= 0:
        raise ValueError(f"interval of {seconds} s: must be positive")
    ticks = after.total_ticks - before.total_ticks
    idle = after.idle_ticks - before.idle_ticks
    cpu = 100.0 * (1.0 - idle / ticks) if ticks > 0 else 0.0
    return (
        cpu,
        (after.rx_bytes - before.rx_bytes) / seconds,
        (after.tx_bytes - before.tx_bytes) / seconds,
    )


def monitor(
    writer: CsvWriter,
    run: str,
    role: str,
    stop: threading.Event,
    interval: float = SAMPLE_INTERVAL_S,
    stat: Path = PROC_STAT,
    net: Path = PROC_NET_DEV,
    host: str | None = None,
    clock: Callable[[], float] = time.time,
    max_samples: int | None = None,
) -> int:
    """Write one row per interval until stop is set.

    Args:
        writer: Open CSV writer with the `MONITOR` columns.
        run: Run id.
        role: `server` or `generator`.
        stop: Event that ends the loop.
        interval: Seconds between samples.
        stat: Path of `/proc/stat`.
        net: Path of `/proc/net/dev`.
        host: Host name, default the local name.
        clock: Source of `time_s`, Unix seconds.
        max_samples: Stop after this many rows, default no limit.

    Returns:
        The number of rows written.
    """
    name = host or socket.gethostname()
    previous = read_counters(stat, net)
    previous_time = clock()
    written = 0
    while not stop.wait(interval):
        current = read_counters(stat, net)
        now = clock()
        cpu, rx, tx = rates(previous, current, now - previous_time)
        writer.write(
            {
                "run": run,
                "host": name,
                "role": role,
                "time_s": now,
                "cpu_percent": cpu,
                "rx_bytes_per_s": rx,
                "tx_bytes_per_s": tx,
            }
        )
        previous, previous_time = current, now
        written += 1
        if max_samples is not None and written >= max_samples:
            break
    return written


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m resampling_benchmark.load.monitor",
        description="Sample CPU and network use once per second until stopped.",
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=SAMPLE_INTERVAL_S)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run until SIGINT or SIGTERM.

    Returns:
        0 after a stop, 1 when `/proc` has no CPU or network counters.
    """
    arguments = parse_arguments(argv)
    if not (PROC_STAT.exists() and PROC_NET_DEV.exists()):
        print(
            "resampling_benchmark.load.monitor: needs /proc, which only Linux has"
        )
        return 1
    stop = threading.Event()

    def handle(signal_number: int, frame: FrameType | None) -> None:
        stop.set()

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)
    header = run_metadata(run=arguments.run, role=arguments.role)
    with CsvWriter(arguments.out, MONITOR, header) as writer:
        monitor(
            writer,
            arguments.run,
            arguments.role,
            stop,
            interval=arguments.interval,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
