"""Write `envs/requirements.lock` for Linux x86_64 and Python 3.12.

Run it as `python envs/lock.py`. The pins come from the dependencies and the
`test` and `dev` extras of `pyproject.toml`, resolved against the wheels of
the Ubuntu 24.04 instances. The host that runs the script can be any
platform.
"""

import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "envs" / "requirements.lock"

PLATFORMS = (
    "manylinux2014_x86_64",
    "manylinux_2_17_x86_64",
    "manylinux_2_28_x86_64",
    "manylinux_2_34_x86_64",
    "manylinux_2_39_x86_64",
    "linux_x86_64",
)
"""Wheel tags of Ubuntu 24.04 on x86_64, which has glibc 2.39."""


def requirements(project: dict[str, object]) -> list[str]:
    """Return the dependencies of a `[project]` table and of every extra.

    Args:
        project: The parsed `[project]` table of `pyproject.toml`.
    """
    dependencies: list[str] = list(project["dependencies"])  # type: ignore[call-overload]
    extras = project.get("optional-dependencies", {})
    for group in extras.values():  # type: ignore[attr-defined]
        dependencies.extend(group)
    return dependencies


def pins_from_report(report: dict[str, object]) -> list[str]:
    """Return the sorted `name==version` pins of a pip installation report.

    Args:
        report: The parsed JSON of `pip install --report`.

    Returns:
        One pin for each distribution the report installs.
    """
    pins = {
        f"{item['metadata']['name']}=={item['metadata']['version']}"
        for item in report["install"]  # type: ignore[attr-defined]
    }
    return sorted(pins, key=str.lower)


def resolve(requirements: list[str]) -> list[str]:
    """Resolve requirements to the pins of Linux x86_64 wheels with pip.

    Args:
        requirements: Requirement strings from `pyproject.toml`.

    Returns:
        The pins of the requirements and of everything they need.
    """
    with tempfile.TemporaryDirectory() as scratch:
        listing = Path(scratch) / "requirements.txt"
        listing.write_text("\n".join(requirements) + "\n", encoding="ascii")
        report = Path(scratch) / "report.json"
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--quiet",
            "--target",
            str(Path(scratch) / "target"),
            "--python-version",
            "3.12",
            "--implementation",
            "cp",
            "--only-binary=:all:",
            "--report",
            str(report),
            "-r",
            str(listing),
        ]
        for platform in PLATFORMS:
            command += ["--platform", platform]
        subprocess.run(command, check=True)
        return pins_from_report(json.loads(report.read_text(encoding="utf-8")))


def main() -> None:
    """Write the lock file."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    lines = resolve(requirements(project))
    LOCK.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"wrote {len(lines)} pins to {LOCK}")


if __name__ == "__main__":
    main()
