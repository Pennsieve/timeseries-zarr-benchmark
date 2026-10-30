"""Command line of the upload stage: `python -m upload <subcommand>`."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from tszbench import schemas
from upload.s3 import pull, upload_recording
from upload.storage import write_storage

RESULTS = Path("results")
"""Default directory of the result CSVs."""


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of the `s3`, `pull`, and `storage` subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m upload", description=__doc__
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_store(command: argparse.ArgumentParser) -> None:
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--bucket", required=True)
        command.add_argument("--prefix", default="")
        command.add_argument("--region", default=None)
        command.add_argument(
            "--s3-endpoint-url",
            default=None,
            help="endpoint of an S3-compatible server; omit it for AWS",
        )

    upload = commands.add_parser(
        "s3", help="time `aws s3 cp` of each format of one recording"
    )
    add_store(upload)
    upload.add_argument("--recording", required=True)
    upload.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
        help="directory of upload-<recording>.csv (default results)",
    )

    download = commands.add_parser(
        "pull", help="sync the bucket to a local root with `aws s3 sync`"
    )
    add_store(download)
    download.add_argument(
        "--recordings",
        default=None,
        help=(
            "comma-separated recordings to sync, such as R1-1h,R4; omit it "
            "to sync everything below the prefix"
        ),
    )

    storage = commands.add_parser(
        "storage",
        help=f"write the bytes at rest of recordings to {schemas.STORAGE_FILE}",
    )
    storage.add_argument("--root", type=Path, required=True)
    storage.add_argument(
        "--recordings",
        required=True,
        help=(
            "comma-separated recordings; their rows replace any rows of the "
            f"same recordings in {schemas.STORAGE_FILE}"
        ),
    )
    storage.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
        help=f"directory of {schemas.STORAGE_FILE} (default results)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand and return the exit status."""
    args = build_parser().parse_args(argv)
    if args.command == "s3":
        upload_recording(
            args.root,
            args.recording,
            args.bucket,
            args.prefix,
            args.results / schemas.upload_file(args.recording),
            region=args.region,
            endpoint_url=args.s3_endpoint_url,
        )
    elif args.command == "pull":
        pull(
            args.root,
            args.bucket,
            args.prefix,
            recordings=(
                None
                if args.recordings is None
                else [name for name in args.recordings.split(",") if name]
            ),
            region=args.region,
            endpoint_url=args.s3_endpoint_url,
        )
    else:
        recordings = [name for name in args.recordings.split(",") if name]
        write_storage(
            args.root, recordings, args.results / schemas.STORAGE_FILE
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
