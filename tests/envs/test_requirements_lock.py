import importlib.util
import re
import tomllib
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "envs" / "requirements.lock"


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "envs" / f"{name}.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_lines() -> list[str]:
    return [
        line
        for line in LOCK.read_text(encoding="ascii").splitlines()
        if line and not line.startswith("#")
    ]


def test_the_lock_file_exists_and_is_ascii() -> None:
    assert LOCK.is_file()
    LOCK.read_text(encoding="ascii")


def test_every_line_of_the_lock_is_an_exact_pin() -> None:
    assert all(
        re.fullmatch(r"[\w.-]+==[\w.!+]+", line) for line in lock_lines()
    )


def test_the_lock_names_each_distribution_once() -> None:
    names = [normalize(re.split(r"[=\s]", line)[0]) for line in lock_lines()]
    assert len(names) == len(set(names))


def test_the_lock_pins_every_requirement_of_pyproject() -> None:
    table = project()
    requirements = list(table["dependencies"])  # type: ignore[call-overload]
    for group in table["optional-dependencies"].values():  # type: ignore[attr-defined]
        requirements.extend(group)
    pinned = {
        normalize(re.split(r"[=\s]", line)[0]): line.split("==")[-1]
        for line in lock_lines()
    }
    for requirement in requirements:
        name = normalize(re.split(r"[\s@=<>!~\[;]", requirement, maxsplit=1)[0])
        assert name in pinned, requirement
        exact = re.fullmatch(r"[\w.-]+(\[[\w,]+\])?==([\w.!+]+)", requirement)
        if exact:
            assert pinned[name] == exact.group(2), requirement


def test_requirements_joins_the_dependencies_and_every_extra() -> None:
    lock = load_script("lock")
    joined = lock.requirements(
        {
            "dependencies": ["numpy==2.5.4"],
            "optional-dependencies": {"test": ["pytest>=8"], "dev": ["ruff"]},
        }
    )
    assert joined == ["numpy==2.5.4", "pytest>=8", "ruff"]


def test_pins_from_report_sorts_and_removes_duplicates() -> None:
    lock = load_script("lock")
    report = {
        "install": [
            {"metadata": {"name": "zarr", "version": "3.4.1"}},
            {"metadata": {"name": "Flask", "version": "3.1.3"}},
            {"metadata": {"name": "zarr", "version": "3.4.1"}},
        ]
    }
    assert lock.pins_from_report(report) == ["Flask==3.1.3", "zarr==3.4.1"]


def test_import_all_lists_the_stage_packages_and_skips_entry_points() -> None:
    import_all = load_script("import_all")
    names = import_all.module_names()
    for expected in (
        "tszbench",
        "resampling_benchmark.service.app",
        "resampling_benchmark.load.orchestrate",
        "plot.figure1",
        "smoke",
    ):
        assert expected in names
    assert not [name for name in names if name.endswith("__main__")]
    assert not [name for name in names if name.startswith(("tests", "envs"))]
