import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = (ROOT / "docs" / "running.md").read_text(encoding="utf-8")


def test_every_background_command_over_ssh_detaches_standard_input() -> None:
    commands = [
        match.group(0)
        for match in re.finditer(r"ssh [^\n]*nohup(?:.|\n)*?&\"", RUNBOOK)
    ]
    assert commands
    for command in commands:
        assert "< /dev/null" in command


def test_the_runbook_sets_the_load_generator_count() -> None:
    assert "load_generator_count" in RUNBOOK


def test_the_runbook_waits_for_health_before_the_first_request() -> None:
    assert RUNBOOK.index("/health") < RUNBOOK.index("end_us=86400000000")
