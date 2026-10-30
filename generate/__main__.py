"""Command line of the generate stage: `python -m generate`."""

import argparse
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from generate.formats import formats_of, write_format, write_one_byte_object
from tszbench import schemas
from tszbench.csvmeta import CsvWriter, run_metadata
from tszbench.recordings import resolve


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of the generate command."""
    parser = argparse.ArgumentParser(
        prog="python -m generate",
        description="Write the files of one recording and time each write.",
    )
    parser.add_argument(
        "--recording", required=True, help="R1 to R4, or R1-1h to R3-1h"
    )
    parser.add_argument(
        "--formats",
        default=None,
        help=(
            "comma-separated formats of "
            f"{','.join(schemas.FORMATS)} (default every format of the "
            "recording)"
        ),
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="directory of the files"
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="directory of generation-<recording>.csv",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="threads that generate channels at once (default 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace files that exist",
    )
    return parser


def measure(path: Path) -> tuple[int, int]:
    """Return the bytes at rest and the number of files below path.

    A single file counts as one object.
    """
    if path.is_file():
        return path.stat().st_size, 1
    sizes = [
        child.stat().st_size for child in path.rglob("*") if child.is_file()
    ]
    return sum(sizes), len(sizes)


def _remove(path: Path) -> None:
    """Delete a file or a directory tree."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    """Write the files of one recording and return the exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        spec = resolve(args.recording)
    except ValueError as error:
        parser.error(str(error))
    available = formats_of(spec)
    if args.formats is None:
        wanted = available
    else:
        wanted = tuple(name for name in args.formats.split(",") if name)
        for name in wanted:
            if name not in schemas.FORMATS:
                parser.error(
                    f"format {name!r}: must be one of {list(schemas.FORMATS)}"
                )
            if name not in available:
                parser.error(
                    f"format {name!r}: recording {spec.id!r} has only "
                    f"{list(available)}"
                )
    if args.workers < 1:
        parser.error(f"workers {args.workers}: must be at least 1")

    targets = {
        name: args.root / schemas.relative_path(spec.id, name)
        for name in wanted
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        print(
            f"refusing to overwrite {existing}: pass --overwrite to replace",
            file=sys.stderr,
        )
        return 1

    header = run_metadata(
        recording=spec.id, seed=spec.seed, workers=args.workers
    )
    csv_path = args.results / schemas.generation_file(spec.id)
    with CsvWriter(csv_path, schemas.GENERATION, header) as writer:
        for name, path in targets.items():
            if path.exists():
                _remove(path)
            started = time.perf_counter()
            write_format(spec, name, args.root, args.workers)
            seconds = time.perf_counter() - started
            size, objects = measure(path)
            writer.write(
                {
                    "recording": spec.id,
                    "format": name,
                    "seconds": seconds,
                    "bytes": size,
                    "objects": objects,
                }
            )
            print(
                f"{spec.id} {name}: {seconds:.1f} s, {size} bytes, "
                f"{objects} objects",
                flush=True,
            )
    write_one_byte_object(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
