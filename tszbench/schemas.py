"""Columns of every CSV the benchmark writes, and the values of its keys.

Times are seconds unless a column name ends in `_us`. Sizes are bytes. Each CSV
opens with `# key: value` lines that `tszbench.csvmeta` writes.
"""

FORMATS = (
    "edf",
    "nwb-hdf5-time",
    "nwb-hdf5-channel",
    "nwb-zarr-time",
    "nwb-zarr-channel",
    "tszarr",
    "tszarr-raw",
)
"""Files written per recording. `-time` chunks time-major, `-channel` per channel.
`tszarr-raw` is a TS-Zarr bundle with raw samples and no levels. Only the 1 h cuts
and R4 have it, and it exists for the storage comparison."""

READ_FORMATS = tuple(name for name in FORMATS if name != "tszarr-raw")
"""Formats the format benchmark reads."""

FORMAT_ROWS = (*READ_FORMATS, "zarr-direct", "overhead")
"""Values of `format` in the format benchmark. `zarr-direct` reads the TS-Zarr raw arrays
with zarr-python and no reader. `overhead` reads a 1-byte object."""

STORES = ("local", "http", "s3")
SUBSET_CHANNELS = (4, 16, 64)
"""Block sizes of the subset reads. A subset read takes this many contiguous
channels."""

SUBSET_KINDS = tuple(f"channels{count}" for count in SUBSET_CHANNELS)
"""Values of `read_kind` for the subset reads, one per entry of `SUBSET_CHANNELS`."""

READ_KINDS = ("one", *SUBSET_KINDS, "all")
"""Values of `read_kind`. `one` reads one channel and `all` reads every channel."""
METHODS = ("server", "precomputed")

GENERATION = ("recording", "format", "seconds", "bytes", "objects")
"""One row per file written by `generate`."""

UPLOAD = (
    "recording",
    "format",
    "seconds",
    "bytes",
    "objects",
    "bytes_per_second",
)
"""One row per file uploaded by `upload` with `aws s3 cp`."""

STORAGE = ("recording", "format", "member", "bytes", "objects")
"""Bytes at rest. `member` is empty for a whole file. A `tszarr` bundle also has
one row per member: `raw`, `minmax`, `mean`, `var`, `count`, and `metadata`.
`tszarr-raw` has a whole-file row only."""

FORMAT_READS = (
    "recording",
    "store",
    "format",
    "read_kind",
    "window_s",
    "repeat",
    "channel",
    "start_us",
    "seconds",
    "bytes",
    "requests",
    "samples",
)
"""One row per timed read. `channel` is -1 for an all-channel read and the first
channel of the block for a subset read. `bytes` and
`requests` are what the store served for the read. `samples` counts samples
returned over every channel read."""

FORMAT_OPENS = (
    "recording",
    "store",
    "format",
    "read_kind",
    "window_s",
    "seconds",
    "bytes",
    "requests",
    "reads",
    "planned_reads",
)
"""One row per process: the untimed open that precedes the timed reads.
`reads` counts the timed reads the process made. `planned_reads` is the repeat
count of the combination. `reads` is below `planned_reads` when the reads used
up the read budget first."""

RESAMPLING_REQUESTS = (
    "run",
    "method",
    "recording",
    "window_s",
    "users",
    "generator",
    "user",
    "sequence",
    "start_us",
    "dispatch_s",
    "seconds",
    "bytes",
    "s3_requests",
    "s3_bytes",
    "status",
    "warmup",
    "level",
    "read_seconds",
    "reduce_seconds",
)
"""One row per request. `dispatch_s` counts from the start of the run. `seconds`
runs from dispatch to the decoded array. `bytes` is what the user received.
`status` is `ok`, `timeout`, or `error:<detail>`. `level` is the level read, 0
for raw. `read_seconds` and `reduce_seconds` split the time spent reading S3 and
reducing to columns, as the server or the precomputed read measured them."""

RESAMPLING_RUNS = (
    "run",
    "method",
    "recording",
    "window_s",
    "users",
    "generators",
    "users_per_generator",
    "requests",
    "failures",
    "failure_rate",
    "requests_per_second",
    "median_s",
    "p5_s",
    "p95_s",
    "median_bytes",
    "raw_bytes",
    "server_cpu_max",
    "generator_link_max",
    "repeated",
    "link_limited",
    "not_run",
)
"""One row per run, warm-up requests left out. `raw_bytes` is the float32 size
of every channel under the window. `server_cpu_max` is the peak of the
per-second CPU use in percent. `generator_link_max` is the peak per-second
receive rate of any load generator as a fraction of its link. `repeated` is
true for a run repeated with half the users per generator. `link_limited` is
true for a run whose peak receive rate passed half the generator link and that
no repeat replaced. `not_run` is true for a run that the orchestrator skipped
because a smaller user count of the same method, recording, and window failed.
Such a row holds the method, recording, window, and users, and no statistics."""

MONITOR = (
    "run",
    "host",
    "role",
    "time_s",
    "cpu_percent",
    "rx_bytes_per_s",
    "tx_bytes_per_s",
    "memory_percent",
)
"""One row per host per second during a resampling benchmark run.
`memory_percent` is the memory in use as a percent of the host memory."""

CHECK = (
    "recording",
    "window_s",
    "start_us",
    "end_us",
    "level",
    "bins_per_column",
    "rows",
    "equal",
)
"""One row per equality check of the two the resampling benchmark methods."""

FILE_NAMES = {
    "edf": "edf.edf",
    "nwb-hdf5-time": "nwb-hdf5-time.nwb",
    "nwb-hdf5-channel": "nwb-hdf5-channel.nwb",
    "nwb-zarr-time": "nwb-zarr-time.nwb.zarr",
    "nwb-zarr-channel": "nwb-zarr-channel.nwb.zarr",
    "tszarr": "tszarr.zarr",
    "tszarr-raw": "tszarr-raw.zarr",
}
"""Name of each file inside its recording directory, `<root>/<recording id>/`."""

NGINX_VERSION_FILE = "nginx-version.txt"
"""Path under the HTTP root of the `nginx -V` output that the file server saves
when it starts."""

ONE_BYTE_OBJECT = "overhead/one-byte"
"""Path under every store root of the 1-byte object that `overhead` reads."""


def relative_path(recording_id: str, file_format: str) -> str:
    """Return the path of a file under a store root."""
    return f"{recording_id}/{FILE_NAMES[file_format]}"


FORMAT_READS_FILE = "format-reads.csv"
"""Result file of the format benchmark, `FORMAT_READS` rows of every recording."""

FORMAT_OPENS_FILE = "format-opens.csv"
"""Result file of the timed opens of the format benchmark, `FORMAT_OPENS` rows."""

FORMAT_PARTS_DIRECTORY = "format-parts"
"""Directory under the results directory that holds the CSVs of each
the format benchmark process."""

RESAMPLING_RAW_DIRECTORY = "resampling-raw"
"""Directory under the results directory that holds one directory per
the resampling benchmark run."""

RESAMPLING_REQUESTS_FILE = "resampling-requests.csv"
"""Result file of the resampling benchmark, `RESAMPLING_REQUESTS` rows of every run."""

RESAMPLING_RUNS_FILE = "resampling-runs.csv"
"""Result file of the resampling benchmark, one `RESAMPLING_RUNS` row per run."""

CHECK_FILE = "resampling-check.csv"
"""Result file of the equality check, `CHECK` rows."""

STORAGE_FILE = "storage.csv"
"""Result file of the bytes at rest, `STORAGE` rows of every recording."""

REQUESTS_PATTERN = "requests-*.csv"
"""Glob of the request CSVs inside a resampling benchmark run directory."""

SERVER_MONITOR_FILE = "monitor-server.csv"
"""Monitor CSV of the resampling server inside a resampling benchmark run directory."""

GENERATOR_MONITOR_PATTERN = "monitor-generator-*.csv"
"""Glob of the load generator monitor CSVs inside a run directory."""

SERVER_METADATA_FILE = "server-metadata.json"
"""JSON file inside a run directory that holds the run metadata of the
resampling server, as `/health` reported it at the start of the run."""

IDLE_WAIT_FILE = "idle-wait-s.txt"
"""File in a server run's directory that holds the seconds spent waiting for
the server to go idle."""

COMPLETE_FILE = "complete"
"""Empty file in a run directory, written after every generator of the run
exits with status 0."""

LINK_LIMITED_FILE = "link-limited"
"""Empty file inside a run directory, written when the peak receive rate of a
generator passed half its link and no repeat with fewer users per generator
could run."""

NOT_RUN_FILE = "not-run"
"""File inside the directory of a run that the orchestrator skipped. It holds
the name of the run with fewer users whose median request failed."""


def generation_file(recording_id: str) -> str:
    """Return the name of the `GENERATION` result file of a recording."""
    return f"generation-{recording_id}.csv"


def upload_file(recording_id: str) -> str:
    """Return the name of the `UPLOAD` result file of a recording."""
    return f"upload-{recording_id}.csv"


def requests_file(generator: int) -> str:
    """Return the name of the request CSV of one load generator of a run."""
    return f"requests-{generator}.csv"


def generator_monitor_file(generator: int) -> str:
    """Return the name of the monitor CSV of one load generator of a run."""
    return f"monitor-generator-{generator}.csv"


def subset_size(read_kind: str) -> int | None:
    """Return the channels of a subset read kind, or None for `one` and `all`."""
    if read_kind in SUBSET_KINDS:
        return SUBSET_CHANNELS[SUBSET_KINDS.index(read_kind)]
    return None
