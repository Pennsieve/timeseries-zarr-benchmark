# timeseries-zarr-benchmark

This repository produces the two latency benchmarks of the TS-Zarr paper.
TS-Zarr is a Zarr-based format for time series recordings. A window is a time
range of a recording. A level is a precomputed, downsampled copy of a
recording. A minmax is the minimum and maximum of the samples in each pixel
column of a plot.

1. The format benchmark reads the raw samples of a window at a random position.
   It reads from EDF, NWB on HDF5, NWB on Zarr, and TS-Zarr files, each stored
   on local disk, behind an HTTP server, and in S3. It measures how the file
   format and the storage location change the latency of a raw read.
2. The resampling benchmark returns the minmax of a random window in two ways.
   A server reads the raw samples and resamples them on demand. A client reads
   the precomputed level directly from S3. It measures which way returns the
   window faster as the window grows.

The full benchmark design is described in the [benchmark plan](docs/plan.md).

## Install

You need Python 3.12 and, for the upload stage, the AWS CLI v2.

```bash
make venv
```

`make venv` installs the pinned libraries of `envs/requirements.lock` into `.venv`.
Every command below runs from the repository root.

## Example

This run writes the first hour of the R1 recording in every format, reads it from
local disk with windows of 1, 10, and 60 seconds, and plots Figures 1 and 2:

```bash
.venv/bin/python -m generate --recording R1-1h --root data --results results
.venv/bin/python -m format_benchmark --recordings R1-1h --stores local --local-root data \
    --windows 1,10,60 --max-repeats 10
.venv/bin/python -m plot --results results --out figures --recording R1-1h \
    --format-recording R1-1h
```

`make smoke` runs every stage on this machine with local servers in place of
AWS. It writes R1-1h and R2-1h, uploads them to a local S3 server (moto),
serves them over a local HTTP server, and runs both benchmarks. It then runs
the equality check, which confirms that the two resampling methods return the
same minmax, and makes the plots. It needs no AWS account and takes about 6
minutes on a laptop. Its output goes to `SMOKE_DIR`, or to a new temporary
directory when `SMOKE_DIR` is not set. The output stays after the run.

[The AWS runbook](docs/running.md) describes how to run the full benchmark
suite on AWS.

## Facts to know

- Recordings are `R1` to `R4`, and the 1 hour cuts `R1-1h`, `R2-1h`, and `R3-1h`.
  R4 lasts 10 minutes and has no EDF file. The format benchmark reads the cuts and
  R4, with windows of 1 second to 10 minutes. Every stage takes these names.
- A store root is a directory or an S3 prefix. It holds `<recording>/<file>`
  for each format and the 1-byte object `overhead/one-byte`. The file names are
  in `tszbench/schemas.py`.
- Window bounds are integer microseconds from sample 0. Times in the CSVs are
  seconds unless the column name ends in `_us`. Sizes are bytes.
- Every CSV opens with `# key: value` lines describing the date, commit,
  region, instance type, library versions, and the options of the run.
- A stopped stage can run again with the same options. `generate` needs
  `--overwrite` to replace files, `format_benchmark` skips the combinations it
  finished, and `resampling_benchmark.load.orchestrate --skip-done` skips the
  runs it finished.
- A combination is one recording, format, store, and window. Each combination
  keeps reading until the timed reads add up to `--read-budget-s` (60 seconds).
  It always makes at least `--min-reads` reads (1). `--max-repeats` caps the
  reads.
- NWB on Zarr runs on hdmf-zarr 0.14 over Zarr v3. One environment holds every
  library.

The stages write these files to the results directory:

| file | stage |
|---|---|
| `generation-<recording>.csv` | `python -m generate` |
| `upload-<recording>.csv` | `python -m upload s3` |
| `storage.csv` | `python -m upload storage` |
| `format-reads.csv`, `format-opens.csv` | `python -m format_benchmark` |
| `resampling-raw/<run>/` | `python -m resampling_benchmark.load.orchestrate` |
| `resampling-requests.csv`, `resampling-runs.csv` | `python -m resampling_benchmark.summarize` |
| `resampling-check.csv` | `python -m resampling_benchmark.check` |

`python -m plot` reads these files and writes the figures, the tables,
`methods.txt`, `captions.txt`, and `cost.txt`. Each figure reads one CSV.

## Build and test

```bash
make check   # ruff, mypy --strict, and pytest
make test    # pytest only
make smoke   # every stage end to end on local servers
make lock    # rewrite envs/requirements.lock for Linux x86_64
make lockcheck  # install the lock in a new venv and import every module
```

Tests that start a local HTTP or S3 server carry the `network` marker. Run
`.venv/bin/pytest -m "not network"` to skip them.

## License

Apache License 2.0. See [the license file](LICENSE).
