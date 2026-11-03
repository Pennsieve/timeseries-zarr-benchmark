import threading
from pathlib import Path

import pytest

from resampling_benchmark.load import monitor
from resampling_benchmark.load.monitor import Reading, rates, read_counters
from tszbench.csvmeta import CsvWriter
from tszbench.schemas import MONITOR

STAT = """cpu  100 10 50 800 40 5 5 0 7 0
cpu0 50 5 25 400 20 2 3 0 3 0
intr 12345
"""
NET = """Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 9000 10 0 0 0 0 0 0 9000 10 0 0 0 0 0 0
  eth0: 1000 10 0 0 0 0 0 0 400 5 0 0 0 0 0 0
  eth1: 500 4 0 0 0 0 0 0 100 2 0 0 0 0 0 0
"""


MEMINFO = """MemTotal:       65536000 kB
MemFree:         1000000 kB
MemAvailable:   49152000 kB
Buffers:          100000 kB
"""


@pytest.fixture
def proc(tmp_path: Path) -> tuple[Path, Path]:
    stat = tmp_path / "stat"
    net = tmp_path / "net_dev"
    stat.write_text(STAT)
    net.write_text(NET)
    return stat, net


@pytest.fixture
def meminfo(tmp_path: Path) -> Path:
    path = tmp_path / "meminfo"
    path.write_text(MEMINFO)
    return path


def test_cpu_ticks_sum_the_first_eight_fields_and_count_idle_and_iowait(
    proc: tuple[Path, Path],
) -> None:
    total, idle = monitor.read_cpu(proc[0])
    assert total == 100 + 10 + 50 + 800 + 40 + 5 + 5 + 0
    assert idle == 800 + 40


def test_network_bytes_leave_out_the_loopback_interface(
    proc: tuple[Path, Path],
) -> None:
    assert monitor.read_network(proc[1]) == (1500, 500)


def test_counters_read_both_files(proc: tuple[Path, Path]) -> None:
    assert read_counters(*proc) == Reading(1010, 840, 1500, 500)


def test_memory_percent_is_total_minus_available_over_total(
    meminfo: Path,
) -> None:
    assert monitor.read_memory_percent(meminfo) == pytest.approx(25.0)


def test_a_meminfo_file_without_an_available_line_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "meminfo"
    path.write_text("MemTotal: 100 kB\nMemFree: 50 kB\n")
    with pytest.raises(ValueError, match="no MemTotal and MemAvailable"):
        monitor.read_memory_percent(path)


def test_a_stat_file_without_a_cpu_line_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "stat"
    path.write_text("intr 1\n")
    with pytest.raises(ValueError, match="no aggregate cpu line"):
        monitor.read_cpu(path)


def test_rates_give_percent_busy_and_bytes_per_second() -> None:
    before = Reading(1000, 800, 1000, 100)
    after = Reading(1200, 850, 3000, 600)
    cpu, rx, tx = rates(before, after, 2.0)
    assert cpu == pytest.approx(75.0)
    assert rx == pytest.approx(1000.0)
    assert tx == pytest.approx(250.0)


def test_rates_report_zero_cpu_when_no_ticks_passed() -> None:
    same = Reading(5, 5, 0, 0)
    assert rates(same, same, 1.0) == (0.0, 0.0, 0.0)


def test_rates_reject_a_non_positive_interval() -> None:
    reading = Reading(1, 1, 0, 0)
    with pytest.raises(ValueError, match="must be positive"):
        rates(reading, reading, 0.0)


def test_the_loop_writes_one_row_per_sample(
    proc: tuple[Path, Path], meminfo: Path, tmp_path: Path
) -> None:
    stat, net = proc
    times = iter([100.0, 101.0, 102.0, 103.0])
    out = tmp_path / "monitor.csv"
    with CsvWriter(out, MONITOR, {"run": "r"}) as writer:
        count = monitor.monitor(
            writer,
            "r",
            "server",
            threading.Event(),
            interval=0.0,
            stat=stat,
            net=net,
            meminfo=meminfo,
            host="h",
            clock=lambda: next(times),
            max_samples=3,
        )
    assert count == 3
    lines = [
        line
        for line in out.read_text().splitlines()
        if not line.startswith("#")
    ]
    assert lines[0].split(",") == list(MONITOR)
    assert len(lines) == 4
    assert lines[1].split(",")[:4] == ["r", "h", "server", "101.0"]
    assert float(lines[1].split(",")[MONITOR.index("memory_percent")]) == (
        pytest.approx(25.0)
    )


def test_the_loop_stops_when_the_event_is_set(
    proc: tuple[Path, Path], meminfo: Path, tmp_path: Path
) -> None:
    stop = threading.Event()
    stop.set()
    with CsvWriter(tmp_path / "m.csv", MONITOR, {}) as writer:
        assert (
            monitor.monitor(
                writer,
                "r",
                "generator",
                stop,
                stat=proc[0],
                net=proc[1],
                meminfo=meminfo,
            )
            == 0
        )


def test_the_loop_sees_a_change_between_two_samples(
    proc: tuple[Path, Path], meminfo: Path, tmp_path: Path
) -> None:
    stat, net = proc
    clock = iter([0.0, 1.0])
    out = tmp_path / "m.csv"
    stop = threading.Event()

    def wait(timeout: float | None = None) -> bool:
        net.write_text(NET.replace("eth0: 1000", "eth0: 3000"))
        stat.write_text(
            STAT.replace("cpu  100 10 50 800", "cpu  300 10 50 800")
        )
        return False

    stop.wait = wait  # type: ignore[method-assign]
    with CsvWriter(out, MONITOR, {}) as writer:
        monitor.monitor(
            writer,
            "r",
            "generator",
            stop,
            stat=stat,
            net=net,
            meminfo=meminfo,
            host="h",
            clock=lambda: next(clock),
            max_samples=1,
        )
    row = out.read_text().splitlines()[-1].split(",")
    assert float(row[4]) == pytest.approx(100.0)
    assert float(row[5]) == pytest.approx(2000.0)
    assert float(row[6]) == pytest.approx(0.0)


def test_the_command_line_needs_a_run_and_a_role() -> None:
    arguments = monitor.parse_arguments(
        ["--run", "r", "--role", "server", "--out", "x.csv"]
    )
    assert (arguments.run, arguments.role) == ("r", "server")
    with pytest.raises(SystemExit):
        monitor.parse_arguments(["--run", "r", "--role", "other", "--out", "x"])
