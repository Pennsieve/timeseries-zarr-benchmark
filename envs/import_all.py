"""Import every module of the repository: `python envs/import_all.py`.

Run it from the repository root in a venv that holds the lock file and nothing
else. A missing library or an import of a name that the pinned version lacks
stops the run with the traceback.
"""

import importlib
import pkgutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = ("tests", "envs", "terraform", "docs", "results")
"""Top-level directories that hold no stage package."""


def module_names() -> list[str]:
    """Return the dotted name of every importable module except entry points.

    Returns:
        Sorted names of the modules under the stage packages, without
        `__main__` modules.
    """
    packages = [
        entry.name
        for entry in sorted(ROOT.iterdir())
        if (entry / "__init__.py").is_file() and entry.name not in EXCLUDED
    ]
    names: set[str] = set(packages)
    for package in packages:
        for module in pkgutil.walk_packages(
            [str(ROOT / package)], prefix=f"{package}."
        ):
            if module.name.rsplit(".", 1)[-1] != "__main__":
                names.add(module.name)
    return sorted(names)


def main() -> None:
    """Import each module and print the count."""
    sys.path.insert(0, str(ROOT))
    names = module_names()
    for name in names:
        importlib.import_module(name)
    print(f"imported {len(names)} modules")


if __name__ == "__main__":
    main()
