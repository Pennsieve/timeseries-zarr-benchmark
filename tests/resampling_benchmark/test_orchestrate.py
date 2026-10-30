import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from resampling_benchmark.load import orchestrate
from resampling_benchmark.load.orchestrate import (
    LINK_BYTES_PER_S,
    Hosts,
    LocalRunner,
    RunSpec,
    Settings,
    execute_run,
    execute_spec,
    fetch_server_metadata,
    generator_command,
    kill_pattern,
    monitor_command,
    parse_hosts,
    peak_receive_rate,
    plan_runs,
    run_grid,
    split_generators,
    users_on,
    wait_for_idle,
)
from resampling_benchmark.summarize import find_runs, summarize, summarize_run
from tests.resampling_benchmark.conftest import TINY_ID, TINY_SPEC
from tszbench.csvmeta import CsvWriter
from tszbench.fs import StoreRoots
from tszbench.schemas import (
    COMPLETE_FILE,
    IDLE_WAIT_FILE,
    LINK_LIMITED_FILE,
    MONITOR,
    RESAMPLING_REQUESTS,
    SERVER_METADATA_FILE,
)

HOSTS_TEXT = """
# role private_ip
file_server 10.0.0.9
client 10.0.0.8
resampling_server 10.0.0.1
load_generator 10.0.1.1
load_generator 10.0.1.2
client 10.0.2.1
"""


def test_the_hosts_file_names_one_server_and_the_generators() -> None:
    assert parse_hosts(HOSTS_TEXT) == Hosts(
        "10.0.0.1", ("10.0.1.1", "10.0.1.2")
    )


@pytest.mark.parametrize(
    "text",
    [
        "load_generator 10.0.1.1\n",
        "resampling_server a\nresampling_server b\nload_generator c\n",
        "resampling_server a\n",
        "server a\ngenerator b\n",
    ],
)
def test_a_hosts_file_without_one_server_and_a_generator_is_rejected(
    text: str,
) -> None:
    with pytest.raises(ValueError, match="exactly one resampling_server"):
        parse_hosts(text)


def test_a_malformed_hosts_line_is_rejected() -> None:
    with pytest.raises(ValueError, match="line 1"):
        parse_hosts("server\n")


def test_the_plan_nests_method_recording_window_and_users() -> None:
    runs = plan_runs()
    assert len(runs) == 2 * (10 + 10 + 20 + 16)
    assert runs[0] == RunSpec("server", "R1", 10, 1)
    assert runs[1] == RunSpec("server", "R1", 10, 100)
    assert runs[2] == RunSpec("server", "R1", 60, 1)
    methods = [run.method for run in runs]
    assert methods == sorted(methods, key=["server", "precomputed"].index)
    server = [run for run in runs if run.method == "server"]
    assert [run.recording for run in server] == sorted(
        run.recording for run in server
    )


def test_r3_runs_every_user_count_over_every_window() -> None:
    runs = [
        run for run in plan_runs(methods=["server"]) if run.recording == "R3"
    ]
    assert [(run.window_s, run.users) for run in runs[:8]] == [
        (10, 1),
        (10, 10),
        (10, 100),
        (10, 1000),
        (60, 1),
        (60, 10),
        (60, 100),
        (60, 1000),
    ]
    assert {run.window_s for run in runs} == {10, 60, 600, 3600, 86400}


def test_r4_stops_at_one_hour_and_r1_and_r2_skip_10_and_1000_users() -> None:
    runs = plan_runs(methods=["precomputed"])
    r4 = [run for run in runs if run.recording == "R4"]
    assert max(run.window_s for run in r4) == 3_600
    assert {run.users for run in r4} == {1, 10, 100, 1000}
    for name in ("R1", "R2"):
        own = [run for run in runs if run.recording == name]
        assert {run.users for run in own} == {1, 100}
        assert {run.window_s for run in own} == {10, 60, 600, 3600, 86400}


def test_filters_narrow_the_plan() -> None:
    runs = plan_runs(
        methods=["server"], recordings=["R3"], windows=[10], users=[2]
    )
    assert runs == [RunSpec("server", "R3", 10, 2)]


def test_a_cut_runs_after_the_full_recordings_and_only_windows_that_fit() -> (
    None
):
    runs = plan_runs(methods=["server"], recordings=["R1-1h", "R2"], users=[1])
    assert [run.recording for run in runs[:5]] == ["R2"] * 5
    assert [(run.recording, run.window_s) for run in runs[5:]] == [
        ("R1-1h", 10),
        ("R1-1h", 60),
        ("R1-1h", 600),
        ("R1-1h", 3600),
    ]


def test_an_unknown_recording_is_rejected() -> None:
    with pytest.raises(ValueError, match="R9"):
        plan_runs(recordings=["R9"])


def test_a_run_is_named_by_method_recording_window_and_users() -> None:
    assert RunSpec("server", "R3", 600, 100).name == "server-R3-600-100"


def test_about_one_hundred_users_share_a_generator() -> None:
    assert split_generators(1, 10) == (1, 1)
    assert split_generators(100, 10) == (1, 100)
    assert split_generators(101, 10) == (2, 51)
    assert split_generators(1000, 10) == (10, 100)
    assert split_generators(1000, 4) == (4, 250)


def test_users_are_dealt_to_generators_in_turn() -> None:
    counts = [users_on(10, 3, index) for index in range(3)]
    assert counts == [4, 3, 3]
    assert sum(users_on(1000, 7, index) for index in range(7)) == 1000


def test_the_generator_command_carries_the_run_and_the_common_start() -> None:
    spec = RunSpec("server", "R3", 600, 100)
    command = generator_command(
        "python",
        spec,
        "server-R3-600-100",
        100,
        2,
        1,
        1234.5,
        "http://10.0.0.1:8000",
        "bucket/prefix",
        None,
        "out/requests-1.csv",
        False,
    )
    assert command[:3] == [
        "python",
        "-m",
        "resampling_benchmark.load.generator",
    ]
    pairs = dict(zip(command[3::2], command[4::2], strict=False))
    assert pairs["--users"] == "50"
    assert pairs["--total-users"] == "100"
    assert pairs["--generators"] == "2"
    assert pairs["--generator-index"] == "1"
    assert pairs["--start-at"] == "1234.5"
    assert pairs["--server-url"] == "http://10.0.0.1:8000"
    assert "--repeated" not in command


def test_the_precomputed_command_has_no_server_url_and_a_repeat_flag() -> None:
    spec = RunSpec("precomputed", "R3", 10, 1)
    command = generator_command(
        "python",
        spec,
        "r",
        1,
        1,
        0,
        0.0,
        "http://unused",
        "bucket/prefix",
        "http://s3",
        "o.csv",
        True,
    )
    assert "--server-url" not in command
    assert command[-1] == "--repeated"
    assert "--s3-endpoint-url" in command


def test_the_monitor_command_names_the_role() -> None:
    command = monitor_command("python", "r", "server", "m.csv")
    assert command[2] == "resampling_benchmark.load.monitor"
    assert command[command.index("--role") + 1] == "server"


def test_the_kill_pattern_matches_its_command_but_not_its_own_shell() -> None:
    pattern = kill_pattern(
        ["py", "-m", "resampling_benchmark.load.monitor", "--run", "r1"]
    )
    assert pattern.startswith("'[r]esampling_benchmark")
    assert pattern.endswith("--run r1( |$)'")


class FakeJob:
    def __init__(self, runner: "FakeRunner", argv: Sequence[str]) -> None:
        self.runner = runner
        self.argv = argv

    def wait(self, timeout: float | None = None) -> int:
        return self.runner.status

    def stop(self) -> None:
        self.runner.stopped.append(" ".join(self.argv))


class FakeRunner:
    """Records launches and writes canned CSVs on fetch."""

    def __init__(self, peaks: dict[str, float] | None = None) -> None:
        self.peaks = peaks or {}
        self.launches: list[tuple[str, list[str]]] = []
        self.stopped: list[str] = []
        self.status = 0

    def launch(self, host: str, argv: Sequence[str]) -> FakeJob:
        self.launches.append((host, list(argv)))
        return FakeJob(self, argv)

    def prepare(self, host: str, directory: str) -> None:
        pass

    def fetch(self, host: str, remote: str, local: Path) -> bool:
        run = Path(remote).parent.name
        if "requests-" in remote:
            with CsvWriter(local, RESAMPLING_REQUESTS, {}):
                pass
        else:
            with CsvWriter(local, MONITOR, {}) as writer:
                writer.write(
                    {
                        "run": run,
                        "host": host,
                        "role": "generator",
                        "time_s": 0.0,
                        "cpu_percent": 1.0,
                        "rx_bytes_per_s": self.peaks.get(run, 0.0),
                        "tx_bytes_per_s": 0.0,
                    }
                )
        return True

    def generator_launches(self) -> list[list[str]]:
        return [
            argv
            for _, argv in self.launches
            if "resampling_benchmark.load.generator" in argv
        ]


def settings(tmp_path: Path, generators: int, **changes: object) -> Settings:
    base = Settings(
        hosts=Hosts(
            "10.0.0.1", tuple(f"10.0.1.{i}" for i in range(generators))
        ),
        server_url="http://10.0.0.1:8000",
        s3_root="bucket/prefix",
        s3_endpoint_url=None,
        out=tmp_path / "out",
        remote_dir="remote",
        start_delay_s=5.0,
    )
    from dataclasses import replace

    return replace(base, **changes)  # type: ignore[arg-type]


def test_a_run_starts_monitors_first_and_every_generator_at_one_time(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    spec = RunSpec("server", "R3", 10, 250)
    result = execute_run(
        spec,
        spec.name,
        3,
        settings(tmp_path, 3),
        runner,
        False,
        clock=lambda: 1000.0,
    )
    roles = [
        "monitor" if "resampling_benchmark.load.monitor" in a else "gen"
        for _, a in runner.launches
    ]
    assert roles == ["monitor"] * 4 + ["gen"] * 3
    starts = {
        argv[argv.index("--start-at") + 1]
        for argv in runner.generator_launches()
    }
    assert starts == {"1005.0"}
    users = [
        int(argv[argv.index("--users") + 1])
        for argv in runner.generator_launches()
    ]
    assert users == [84, 83, 83]
    assert len(runner.stopped) == 4
    assert (result.directory / "requests-2.csv").exists()
    assert (result.directory / "monitor-server.csv").exists()
    assert result.users_per_generator == 84
    assert result.generators == 3


def test_a_failed_generator_raises_after_copying_the_files(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    runner.status = 2
    spec = RunSpec("server", "R3", 10, 1)
    with pytest.raises(RuntimeError, match=r"exit statuses \[2\]"):
        execute_run(spec, spec.name, 1, settings(tmp_path, 1), runner, False)
    assert (tmp_path / "out" / spec.name / "requests-0.csv").exists()


def test_peak_receive_rate_is_the_highest_rate_of_any_file(
    tmp_path: Path,
) -> None:
    paths = []
    for index, rate in enumerate((5.0, 9.0)):
        path = tmp_path / f"m{index}.csv"
        with CsvWriter(path, MONITOR, {}) as writer:
            for value in (rate, rate / 2):
                writer.write(
                    {
                        "run": "r",
                        "host": "h",
                        "role": "generator",
                        "time_s": 0.0,
                        "cpu_percent": 0.0,
                        "rx_bytes_per_s": value,
                        "tx_bytes_per_s": 0.0,
                    }
                )
        paths.append(path)
    assert peak_receive_rate([*paths, tmp_path / "absent.csv"]) == 9.0
    assert peak_receive_rate([tmp_path / "absent.csv"]) is None


def test_a_run_below_half_the_link_is_not_repeated(tmp_path: Path) -> None:
    spec = RunSpec("precomputed", "R3", 10, 100)
    runner = FakeRunner({spec.name: 0.49 * LINK_BYTES_PER_S})
    results = execute_spec(spec, settings(tmp_path, 4), runner)
    assert [result.run for result in results] == [spec.name]
    assert not results[0].repeated
    assert not (results[0].directory / LINK_LIMITED_FILE).exists()


def test_a_run_above_half_the_link_repeats_with_half_the_users_per_generator(
    tmp_path: Path,
) -> None:
    spec = RunSpec("precomputed", "R3", 10, 100)
    runner = FakeRunner({spec.name: 0.51 * LINK_BYTES_PER_S})
    results = execute_spec(spec, settings(tmp_path, 4), runner)
    assert [result.run for result in results] == [
        spec.name,
        f"{spec.name}-repeat",
    ]
    first, second = results
    assert (first.generators, first.users_per_generator) == (1, 100)
    assert (second.generators, second.users_per_generator) == (2, 50)
    assert second.repeated
    repeat_commands = runner.generator_launches()[1:]
    assert len(repeat_commands) == 2
    assert all("--repeated" in argv for argv in repeat_commands)
    assert not any(
        "--repeated" in argv for argv in runner.generator_launches()[:1]
    )


def test_a_repeat_is_not_repeated_again(tmp_path: Path) -> None:
    spec = RunSpec("precomputed", "R3", 10, 100)
    runner = FakeRunner(
        {
            spec.name: 0.9 * LINK_BYTES_PER_S,
            f"{spec.name}-repeat": 0.9 * LINK_BYTES_PER_S,
        }
    )
    results = execute_spec(spec, settings(tmp_path, 8), runner)
    assert len(results) == 2
    first, second = results
    assert not (first.directory / LINK_LIMITED_FILE).exists()
    assert (second.directory / LINK_LIMITED_FILE).exists()


def test_a_repeat_below_half_the_link_carries_no_link_limited_file(
    tmp_path: Path,
) -> None:
    spec = RunSpec("precomputed", "R3", 10, 100)
    runner = FakeRunner(
        {spec.name: 0.9 * LINK_BYTES_PER_S, f"{spec.name}-repeat": 0.4e9}
    )
    results = execute_spec(spec, settings(tmp_path, 8), runner)
    assert len(results) == 2
    assert not any(
        (result.directory / LINK_LIMITED_FILE).exists() for result in results
    )


def test_a_new_run_removes_the_link_limited_file_of_an_earlier_one(
    tmp_path: Path,
) -> None:
    spec = RunSpec("precomputed", "R3", 10, 1000)
    directory = tmp_path / "out" / spec.name
    directory.mkdir(parents=True)
    (directory / LINK_LIMITED_FILE).touch()
    execute_spec(spec, settings(tmp_path, 10), FakeRunner())
    assert not (directory / LINK_LIMITED_FILE).exists()


def test_a_repeat_that_needs_more_generators_than_the_hosts_file_holds_is_logged_and_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    spec = RunSpec("precomputed", "R3", 10, 1000)
    runner = FakeRunner({spec.name: 0.9 * LINK_BYTES_PER_S})
    with caplog.at_level(
        logging.WARNING, logger="resampling_benchmark.orchestrate"
    ):
        results = execute_spec(spec, settings(tmp_path, 10), runner)
    assert [result.run for result in results] == [spec.name]
    assert "needs 20 generators and 10 are available" in caplog.text
    assert "keeping the first run" in caplog.text
    assert (results[0].directory / LINK_LIMITED_FILE).exists()


def test_a_run_with_one_user_per_generator_cannot_be_halved(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    spec = RunSpec("precomputed", "R3", 10, 1)
    runner = FakeRunner({spec.name: 0.9 * LINK_BYTES_PER_S})
    with caplog.at_level(
        logging.WARNING, logger="resampling_benchmark.orchestrate"
    ):
        results = execute_spec(spec, settings(tmp_path, 10), runner)
    assert len(results) == 1
    assert "keeping the first run" in caplog.text
    assert (results[0].directory / LINK_LIMITED_FILE).exists()


def test_too_few_generators_for_the_users_are_logged_and_used_in_full(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    spec = RunSpec("server", "R3", 10, 1000)
    runner = FakeRunner()
    with caplog.at_level(
        logging.WARNING, logger="resampling_benchmark.orchestrate"
    ):
        (result,) = execute_spec(spec, settings(tmp_path, 4), runner)
    assert result.generators == 4
    assert result.users_per_generator == 250
    assert "needs 10 generators" in caplog.text


def test_a_run_saves_the_server_metadata_it_is_given(tmp_path: Path) -> None:
    seen: list[str] = []

    def metadata(url: str) -> dict[str, str]:
        seen.append(url)
        return {"instance": "c6i.8xlarge", "ami": "ami-1"}

    spec = RunSpec("server", "R3", 10, 1)
    result = execute_run(
        spec,
        spec.name,
        1,
        settings(tmp_path, 1, server_metadata=metadata),
        FakeRunner(),
        False,
    )
    assert seen == ["http://10.0.0.1:8000"]
    saved = json.loads((result.directory / SERVER_METADATA_FILE).read_text())
    assert saved == {"instance": "c6i.8xlarge", "ami": "ami-1"}


def test_a_failed_metadata_lookup_is_logged_and_leaves_the_file_out(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def metadata(url: str) -> dict[str, str]:
        raise httpx.ConnectError("refused")

    spec = RunSpec("server", "R3", 10, 1)
    with caplog.at_level(
        logging.WARNING, logger="resampling_benchmark.orchestrate"
    ):
        result = execute_run(
            spec,
            spec.name,
            1,
            settings(tmp_path, 1, server_metadata=metadata),
            FakeRunner(),
            False,
        )
    assert not (result.directory / SERVER_METADATA_FILE).exists()
    assert "no server metadata" in caplog.text


def test_the_server_metadata_comes_from_the_health_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def get(url: str, timeout: float) -> httpx.Response:
        assert url == "http://s/health"
        return httpx.Response(
            200,
            json={"in_flight": 0, "metadata": {"instance": "c6i", "n": 3}},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(orchestrate.httpx, "get", get)
    assert fetch_server_metadata("http://s") == {"instance": "c6i", "n": "3"}


class FakeHealth:
    """Answers `/health` with a scripted `in_flight` count."""

    def __init__(self, counts: Sequence[int]) -> None:
        self.counts = list(counts)
        self.polls = 0

    def __call__(self, url: str, timeout: float) -> httpx.Response:
        self.polls += 1
        count = self.counts[min(self.polls, len(self.counts)) - 1]
        return httpx.Response(200, json={"in_flight": count})


def test_waiting_for_idle_polls_until_no_task_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    health = FakeHealth([40, 7, 0])
    monkeypatch.setattr(orchestrate.httpx, "get", health)
    sleeps: list[float] = []
    waited = wait_for_idle("http://s", poll_s=0.5, sleep=sleeps.append)
    assert health.polls == 3
    assert sleeps == [0.5, 0.5]
    assert waited >= 0


def test_an_idle_server_is_not_waited_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    health = FakeHealth([0])
    monkeypatch.setattr(orchestrate.httpx, "get", health)
    sleeps: list[float] = []
    assert wait_for_idle("http://s", sleep=sleeps.append) < 1.0
    assert sleeps == []


def test_a_server_that_stays_busy_raises_after_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestrate.httpx, "get", FakeHealth([3]))
    with pytest.raises(TimeoutError, match="3 tasks in flight"):
        wait_for_idle("http://s", timeout=0.0, sleep=lambda seconds: None)


def test_a_server_run_waits_for_idle_and_records_the_wait(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    seen: list[tuple[str, int]] = []

    def idle(url: str) -> float:
        seen.append((url, len(runner.launches)))
        return 12.5

    spec = RunSpec("server", "R3", 10, 1)
    result = execute_run(
        spec,
        spec.name,
        1,
        settings(tmp_path, 1, idle_check=idle),
        runner,
        False,
    )
    assert seen == [("http://10.0.0.1:8000", 0)]
    assert (result.directory / IDLE_WAIT_FILE).read_text() == "12.500\n"


def test_a_precomputed_run_does_not_wait_for_the_server(
    tmp_path: Path,
) -> None:
    def idle(url: str) -> float:
        raise AssertionError("the precomputed method has no server")

    spec = RunSpec("precomputed", "R3", 10, 1)
    execute_run(
        spec,
        spec.name,
        1,
        settings(tmp_path, 1, idle_check=idle),
        FakeRunner(),
        False,
    )


def test_the_wait_runs_between_two_server_runs(tmp_path: Path) -> None:
    runner = FakeRunner()
    events: list[str] = []

    def idle(url: str) -> float:
        events.append(f"idle after {len(runner.generator_launches())}")
        return 0.0

    runs = [RunSpec("server", "R3", 10, 1), RunSpec("server", "R4", 10, 1)]
    failures = run_grid(runs, settings(tmp_path, 1, idle_check=idle), runner)
    assert failures == 0
    assert events == ["idle after 0", "idle after 1"]


def test_a_server_that_never_goes_idle_fails_the_run_and_the_grid_continues(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()

    def busy(url: str) -> float:
        raise TimeoutError("server stays busy")

    runs = [RunSpec("server", "R3", 10, 1), RunSpec("precomputed", "R3", 10, 1)]
    failures = run_grid(runs, settings(tmp_path, 1, idle_check=busy), runner)
    assert failures == 1
    assert len(runner.generator_launches()) == 1


def test_skip_done_runs_again_a_run_whose_generator_failed(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    runner.status = 1
    spec = RunSpec("server", "R3", 10, 1)
    configured = settings(tmp_path, 1)
    assert run_grid([spec], configured, runner, skip_done=True) == 1
    directory = tmp_path / "out" / spec.name
    assert (directory / "requests-0.csv").exists()
    assert not (directory / COMPLETE_FILE).exists()
    runner.status = 0
    assert run_grid([spec], configured, runner, skip_done=True) == 0
    assert len(runner.generator_launches()) == 2
    assert (directory / COMPLETE_FILE).exists()


def test_skip_done_skips_a_finished_run(tmp_path: Path) -> None:
    runner = FakeRunner()
    spec = RunSpec("server", "R3", 10, 1)
    configured = settings(tmp_path, 1)
    run_grid([spec], configured, runner, skip_done=True)
    run_grid([spec], configured, runner, skip_done=True)
    assert len(runner.generator_launches()) == 1


def test_a_run_started_again_drops_its_old_complete_file(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    spec = RunSpec("server", "R3", 10, 1)
    configured = settings(tmp_path, 1)
    run_grid([spec], configured, runner)
    runner.status = 1
    assert run_grid([spec], configured, runner) == 1
    assert not (tmp_path / "out" / spec.name / COMPLETE_FILE).exists()


@pytest.mark.network
@pytest.mark.parametrize("method", ["server", "precomputed"])
def test_a_local_run_of_two_users_writes_the_requests_and_summarizes(
    method: str, server_url: str, tiny_roots: StoreRoots, tmp_path: Path
) -> None:
    on_linux = Path("/proc/stat").exists()
    local = Settings(
        hosts=Hosts("localhost", ("localhost",)),
        server_url=server_url,
        s3_root=tiny_roots.s3,
        s3_endpoint_url=tiny_roots.s3_endpoint_url,
        out=tmp_path / "raw",
        remote_dir=str(tmp_path / "remote"),
        python=sys.executable,
        monitors=on_linux,
        processes=1,
        measured_requests=2,
        start_delay_s=1.0,
        duration_s=TINY_SPEC.duration_s,
    )
    spec = RunSpec(method, TINY_ID, 10, 2)
    results = execute_spec(spec, local, LocalRunner())
    assert [result.run for result in results] == [spec.name]
    files = find_runs(tmp_path / "raw")
    assert [run.run for run in files] == [spec.name]
    row = summarize_run(files[0], lambda name: TINY_SPEC)
    assert row["requests"] == 4
    assert row["failures"] == 0
    assert row["users"] == 2
    assert row["generators"] == 1
    assert row["raw_bytes"] == 4 * 10 * 2000 * 4
    assert float(str(row["median_s"])) > 0
    assert (row["server_cpu_max"] != "") == on_linux
    requests, runs = summarize(
        tmp_path / "raw", tmp_path / "results", lambda name: TINY_SPEC
    )
    assert (requests, runs) == (6, 1)


def test_the_orchestrator_help_lists_its_options() -> None:
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "resampling_benchmark.load.orchestrate",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0
    assert "--hosts" in done.stdout
    assert "--local" in done.stdout
