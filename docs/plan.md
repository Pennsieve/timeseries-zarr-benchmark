# TS-Zarr benchmark plan

Two benchmarks support the paper. The format benchmark reads raw samples at random
positions from EDF, NWB on HDF5, NWB on Zarr, and TS-Zarr. Each format sits on local disk,
behind an HTTP server, and in S3. The resampling benchmark serves the minmax of a window,
per pixel column, in two ways. A server computes it from raw samples on demand, or a
client reads the precomputed level straight from S3. Both run with 1 to 1000 simultaneous
users. A storage table and one cost paragraph come from the same runs. Nothing here times
a browser, a filter, or a montage. Appendix A of the manuscript bounds filtered and
montaged views from the level members analytically.

## The OME latency benchmark as the model

Moore et al. (2021) wrote two synthetic images in TIFF, HDF5, and Zarr and put each on
local disk, behind nginx, and in S3. They read 100 random chunks from each file in series.
They read a 1-byte file the same way as the per-request floor of each store. One figure
carries the result: a raincloud plot of seconds per chunk on a log axis. Rows are the
formats plus the 1-byte row, colors are the stores, and panels are the two images. A
second, computed figure shows chunk count against chunk size. Methods prose gives file
counts and one sentence on upload speed. The text quotes no numbers. It says "similar" and
"at least an order of magnitude faster". The code and the data generator are archived on
Zenodo under BSD-2.

The format benchmark is that design with time windows in place of chunks. Figure 1 here is
their Figure 1. The storage table replaces their chunk-count figure, and the upload
sentence is repeated for these formats. The resampling benchmark has no OME counterpart.

## Recordings

Four synthetic recordings, each generated from one committed seed.

| id | modality | channels | rate | duration | samples per channel | raw float32 | levels |
|---|---|---|---|---|---|---|---|
| R1 | single-channel ambulatory EEG | 1 | 256 Hz | 24 h | 22,118,400 | 88 MB | 7 |
| R2 | scalp EEG | 32 | 512 Hz | 24 h | 44,236,800 | 5.7 GB | 7 |
| R3 | intracranial EEG | 200 | 2000 Hz | 24 h | 172,800,000 | 138 GB | 8 |
| R4 | Neuropixels-like | 384 | 30 kHz | 10 min | 18,000,000 | 28 GB | 7 |

R3 is the worked example of the manuscript. Level counts follow the 1024-bin threshold of
the specification.

Each channel is Gaussian noise with a 1/f spectrum and a bump at 8 to 12 Hz, from a child
seed of the master seed. R4 carries a bump at 300 to 3000 Hz instead. RMS amplitude is
20 uV for R1 and R2, 50 uV for R3, and 100 uV for R4. Each recording has three planted
gaps on channel 0. R1, R2, and R3 hold 1 s at 600 s, 10 s at 1200 s, and 10 min at
1800 s. All three lie inside the first hour, so the 1 h cuts hold them. R4 holds 1 s at
60 s, 10 s at 120 s, and 1 min at 300 s. Samples are quantized to a 0.125 uV
grid, so the int16 EDF and the float32 copies hold the same values. EDF stores a gap
sample as the lowest digital value, -32768, and the range reader returns it as NaN.

The format benchmark reads the 1 h cuts R1-1h, R2-1h, and R3-1h, and R4. These four are
written in every format below. R1, R2, and R3 at full length are written as TS-Zarr only,
for the resampling benchmark.

Files written per recording:

- EDF: one file, every channel in each record. Records last 1 s for R1 and R2 and
  0.064 s for R3. A 1 s record of R3 would hold 800 KB, and the EDF specification
  recommends at most 61,440 B. R4 has no EDF file. EDF is not used at 30 kHz.
- NWB on HDF5, twice, written with pynwb: one ElectricalSeries of shape (samples,
  channels). One file chunks it time-major, about 1 MB with every channel. The other
  chunks it per channel, 8192 samples by 1 channel.
- NWB on Zarr, twice, with the same two chunk shapes, written with pynwb and hdmf-zarr.
- TS-Zarr: one bundle written by `write_bundle` from timeseries-zarr with
  `WriteOpts(zstd_level=3, write_var=True)`. The bundle holds raw, minmax, mean, var, and
  count at every level. Inner chunks hold 8192 elements, shards 16 MiB, and the level
  threshold is 1024 bins. The root `zarr.json` holds the consolidated metadata.

Every chunked file uses Zstd level 3 (through hdf5plugin for HDF5). EDF has no
compression.

Every format is written straight from the synthesizer. The TS-Zarr source is a
`MultiChannelSource` that synthesizes each block on request. Seeded streaming filters
shape the noise. Their state carries across blocks, so no seam appears at a block edge,
and no channel is held in memory. The writer runs in one process and writes to local
disk. Each bundle is therefore written locally and then uploaded. Each recording runs as
its own process, and R3-1h runs before R3. Methods quotes the generation seconds of
R3-1h and R4 per format, and of R3 in TS-Zarr.

## Software

Each format is read and written with its standard library. NWB on HDF5 uses pynwb and
h5py. NWB on Zarr uses pynwb and hdmf-zarr. TS-Zarr uses timeseries-zarr 0.1.0a10, the
Python reference reader and writer, for both benchmarks. EDF is the one exception. No EDF
library reads a window by byte range: pyedflib and MNE take a local path, and edfio and
edfplus load the whole signal. A small EDF range reader is therefore part of the package.

No resampling code is written for the benchmark. Both methods of the resampling benchmark
reduce to pixel columns with `NaNMinMaxDownsampler` from tsdownsample 0.1.5.1. The
resampling server is FastAPI under one uvicorn worker. Each request fans out over a
process pool with one process per vCPU. Each pool process holds its own counting
filesystem and its own open bundles.

One Python 3.12 environment holds every library, because hdmf-zarr 0.14 runs on
zarr 3.4. It holds zarr, timeseries-zarr, tsdownsample, pynwb, h5py with hdf5plugin,
hdmf-zarr, fsspec, and s3fs. Every CSV header prints the version of each library. The
paper's Methods names the reader once, as the Python reference reader (timeseries-zarr
0.1.0a10), and cites tsdownsample with its version.

## Machines and stores

Everything runs in one AWS region. Terraform creates and destroys it.

| role | instance | notes |
|---|---|---|
| file server | c6i.8xlarge | nginx; one gp3 volume holds every file |
| benchmark client | c6i.8xlarge | generation and the format benchmark; a gp3 volume holds the local copy |
| resampling server | c6i.8xlarge | answers the server requests of the resampling benchmark |
| load generators | c6i.8xlarge, 20 | simulate the users of the resampling benchmark; created only for it |
| bucket | S3, same region | private; read with the instance role |

A c6i.8xlarge has 32 vCPU, 64 GiB, a fixed 12.5 Gbps network, and a fixed 10 Gbps
(1250 MB/s) EBS bandwidth. The c6i.2xlarge and c6i.4xlarge have network baselines of
3.125 and 6.25 Gbps and reach 12.5 Gbps only in bursts. Each data volume is gp3 with
16,000 IOPS and 1000 MiB/s, so local and HTTP reads are capped at 1000 MiB/s, below the
instance figure. The instances reach S3 through a gateway endpoint in their VPC, so S3
traffic does not pass the internet gateway.

nginx serves byte ranges over HTTP/1.1 without TLS, gzip off. The bucket is read through
s3fs with signed requests, and the signing cost is part of every S3 read. Every CSV header
records the run date, commit, region, instance type, AMI, library versions, and the
options of the run.

## Format benchmark: random reads of raw samples

A read returns the raw samples of a window that starts at a random position. Two kinds of
read are timed: one channel over the window, and every channel over the window. Window
lengths are 1 s, 10 s, 1 min, and 10 min. The same seeded positions serve every format
and store, so each format reads the same windows. The benchmark reads R1-1h, R2-1h,
R3-1h, and R4. The time of a read depends on the window and the chunk layout. The length
of the file does not change it.

Each combination of recording, store, format, read kind, and window runs in a fresh
process. The process opens the file and fetches metadata before the timer starts: the
root `zarr.json`, the HDF5 superblock, or the EDF header. The timer runs from the read
call to the decoded array in memory. Repeats are 100 reads for windows up to 1 min and 20
at 10 min. A read budget of 60 s (`READ_BUDGET_S`) caps each combination. It stops after
the read that brings the sum of its timed read seconds to at least 60 s, once it has made
one read (`MIN_READS`). A read is never interrupted, so a combination whose first read
takes longer than the budget makes that one read. The open CSV records the reads made and
the reads planned. The page cache is dropped on the client and on the file server before
each process. A combination whose single read exceeds 8 GB is not measured. That excludes
the all-channel read of R4 at 10 min. Window starts are whole microseconds. Every format
returns the samples whose start lies in the window, so a window of whole seconds holds
exactly its length times the rate in samples per channel. Per read, the CSV records
seconds, bytes transferred, the number of requests, and the number of samples returned.
The open is timed once per process, for every format. Opening a TS-Zarr bundle fetches
`zarr.json` and `recording.json` and parses the metadata.

Readers, all through fsspec so that the store is the only thing that changes:

- EDF: the package's range reader. It parses the header, then fetches the bytes of the
  records under the window. Ranges closer than 1 MiB merge into one request, and a
  request holds at most 16 MiB. These limits are checked against the request counts of
  the zarr rows.
- NWB on HDF5: pynwb opens the file through h5py over an fsspec file object with
  read-ahead off and the HDF5 chunk cache off. A read slices `data` of the
  ElectricalSeries. An all-channel read is one selection, and HDF5 fetches the chunks it
  covers in series.
- NWB on Zarr: pynwb with hdmf-zarr opens the file over fsspec. A read slices `data` of
  the ElectricalSeries.
- TS-Zarr: `open_bundle` from timeseries-zarr opens the bundle. A one-channel read is
  `ContinuousChannel.read`. An all-channel read is `Bundle.stack` with `max_workers=32`.
- Overhead: a 1-byte object read through the same fsspec filesystem, the per-request
  floor of each store.

A wrapper around the fsspec filesystem counts requests and bytes for every row. It also
caps requests in flight at 32 for every format. Without the cap, 32 reader threads could
keep 320 requests in flight, because zarr allows 10 per read. The reader receives the
wrapped filesystem as a prebuilt `FsspecStore`. One supplementary row reads the TS-Zarr
raw arrays with zarr-python 3 and no reader, to show how much time the reader adds.

EDF transfers 2 bytes per sample and the float32 formats 4, before compression. Figure 2
shows median bytes per read below median seconds per read.

## Resampling benchmark: on-demand resampling against precomputed levels

A request asks for the minmax of every channel over a window, reduced to 2000 pixel
columns, at a random start. The response is a (channels, rows, 2) float32 array
with one row per column. A window that holds fewer than two entries per column (samples
for the server, bins for the precomputed read) returns one row per entry. Such a response
holds between 2000 and 3999 rows. That happens to both methods for R1 at 10 s, and to the
precomputed read for R1 at 1 min, 10 min, and 1 h, and for R2 and R3 at 24 h. Windows are
10 s, 1 min, 10 min, 1 h, and 24 h. R4 stops at 10 min, its length. Two methods answer
the same request from the same bundle in S3.

For each request the resampling server splits each channel into groups of whole columns
that span about one shard (4,194,304 samples). Each channel and group is one task on the
process pool. A task reads its part of `raw` from S3 with the reader, trims it to its
columns with `Samples.first`, reduces it with tsdownsample, and discards the samples.
Memory stays near one group per pool process. There is no cache. Every request reads S3.
The server records read seconds and reduce seconds apart for each request. When a client
disconnects, the server cancels the tasks of its request that have not started.

`NaNMinMaxDownsampler` returns two sample indices per column, in sample order. The server
takes the smaller value as the minimum and the larger as the maximum. A column that holds
a gap comes back NaN, which is what the level members hold. The plain `MinMaxDownsampler`
skips NaN and would draw through a gap. When a group holds the same whole number of
samples for every column, each tsdownsample column holds exactly those samples. The
generator plants NaN gaps only, because tsdownsample does not specify how it treats
infinities. tsdownsample runs on one thread per call. The pool processes already fill the
instance.

The precomputed read runs on the load generator itself, with the reader. `level_for` with
`bins=2000` picks the coarsest level that holds at least 2000 bins in the window. At every
window here that is also the coarsest level whose `period_us` is at most one column, the
rule in the manuscript. When no level holds 2000 bins, `level_for` returns level 0 and the
read uses `raw`. That happens for R1 and R2 at 10 s. A pool of 32 threads reads each
channel with `read` at that level and `members=()`, which fetches the `minmax` array
alone. tsdownsample then reduces the min column and the max column of the bins to 2000
columns. `Bundle.stack` is not used, because it returns one statistic and a minmax read
through it would fetch `minmax` twice. For R3 the level is 1 at 10 s, 2 at 1 min, 4 at 10
min, 5 at 1 h, and 8 at 24 h. The bins under the window then hold between 4 and 12 MB
before compression. Level bins are finer than columns, so that is one to four times the
bytes of the server's response. The read fetches whole inner chunks, so bytes per request
are recorded and shown.

Users are simulated in closed loop. Each user sends a request, waits for the full
response, decodes it, and sends the next one at a new random start. A load generator runs
at most 100 users. Each user sends one warm-up request, which is dropped, and 10 measured
requests. Users stop sending 5 min after the start of a run. A request that exceeds 180 s
or returns an error (including S3 503 SlowDown) is a failure. Medians and percentiles
count each failed request as 180 s, or as its own seconds when those are longer. At 12.5
Gbps, the 138 GB of float32 under a 24 h window of R3 takes 88 s to move. Before the
grid, one timed 1-user request of R3 at 24 h must finish in under 180 s. Before each
server run, the orchestrator waits until the server has no task in flight.

Per request the CSV records the method, window, start, user count, seconds from dispatch
to the decoded array, bytes received, requests sent to S3, the level read, and status. Per
run it records requests per second achieved and the failure rate. The server's CPU and
network throughput are sampled once per second during every run, as is each load
generator's network throughput. If a load generator exceeds half its link, the run is
repeated with half the users per generator on twice the generators, and the repeat
replaces it. A repeat that still passes half the link is marked link limited, as is a run
with too few generators for a repeat.

Once per window and recording, the two methods must return equal arrays, with NaN equal
to NaN. The check window gives every column the same whole number of level bins (samples
at level 0) and starts on a bin edge at a whole microsecond. It is centered on the
longest planted gap shorter than the window, so the gap channel takes part.

R3 runs 1, 10, 100, and 1000 users at every window. R4 runs 1 and 100 users, and R1 and
R2 run 1 user, at every window. The user counts of a method, recording, and window run in
ascending order. Once the median request of a run is a failure, the larger user counts of
that method, recording, and window are not run, and the results mark them not run.

## Storage and cost

After generation, one table reports per recording and format the bytes at rest, the
object count, and for TS-Zarr the bytes per member. The multipliers of the paper (raw plus
minmax at 5N/3, each further member at N/3) then stand beside measured values. One
sentence in Methods gives the median upload rate of each format with `aws s3 cp` at
default settings, as the OME paper did.

One paragraph prices one request of the resampling benchmark on R3 at each window from the
recorded numbers. A precomputed request costs its median S3 GET requests at the GET price
plus its median bytes at the data-out price. A server request costs the hourly instance
price divided by the requests per hour of its fastest run without failures, plus its own
S3 GETs at the GET price, plus its median response bytes at the data-out price. Both
methods pay the data-out price, because both send their bytes to a client outside AWS. The
paragraph states its prices, region, and date, then one sentence per window. The prices
are the us-east-1 list prices in `plot/prices.json`, checked again on the run date.

## Figures and tables

- Figure 1. Seconds per read at the 10 s all-channel window: a raincloud plot on a log
  axis. One row per format plus Overhead, one color per store, one panel per recording.
  The direct counterpart of OME Figure 1.
- Figure 2. Median seconds per read (upper rows) and median bytes per read (lower rows)
  against window length, log-log, one line per format, one panel per store and read
  kind, for R3-1h. A dotted line marks the median Overhead read of each store. The other
  recordings go to the supplement.
- Figure 3. The resampling benchmark on R3. One small plot per window. The x axis is the
  number of simultaneous users. The y axis is seconds per request, median with a p5 to p95
  band. One line for the server and one for the precomputed read. A cross at the median
  marks a run with failures, and an open ring marks a link-limited run. A user count that
  was not run has no point. A second row shows requests per second. R4 goes to the
  supplement.
- Figure 4. All four recordings on one plot. The x axis is raw bytes under the window. The
  y axis is median seconds per request. One line per method, drawn at 1 user and at 100
  users, with the failure cross and the link-limited ring of Figure 3. R1 and R2 have the
  1-user line only. This figure shows
  how each method's time depends on the raw bytes under the window.
- Table 1. The recordings table above.
- Table 2. Storage: bytes at rest, object count, and multiplier per recording and format.
- Table 3. Median server seconds divided by median precomputed-read seconds for each
  window and user count on R3. A cell reads "fail" where the server run had failures and
  "fail (precomputed)" where only the precomputed run had failures. A cell reads "not
  run" where a run was not run after a failure at fewer users. A cell ends with
  "(link)" where either run is link limited. The 10 s, 10 min, and 24 h rows are the
  page, overview, and whole-recording zooms of the manuscript's worked example.
- Table 4. Median seconds per read of the TS-Zarr reader and of zarr-python on the raw
  arrays of the same bundle (the zarr-direct row), and their ratio, per recording, store,
  read kind, and window.
- Methods sentences: the upload rate per format, the median seconds and requests of the
  timed open per format and store, the combinations that stopped at the read budget, the
  runs not run, the generation seconds of R3-1h, R3, and R4, the equality check, and the
  versions of timeseries-zarr and tsdownsample with the instance types.

Every figure reads one CSV and nothing else. Captions name the statistic, the number of
reads or requests, and the store. Captions of the format benchmark state the read budget,
and captions of the resampling benchmark state the seconds a failed request counts as.

## Reproduction package

A separate repository, mirroring the OME latency benchmark: `terraform/`, `envs/` (one
lock file for Linux x86_64, written by `make lock` and installed and imported by `make
lockcheck`), `tszbench/` (the code the stages share, with the synthesizer and the EDF
writer and range reader), `generate/`, `upload/`, `format_benchmark/`,
`resampling_benchmark/service/`, `resampling_benchmark/load/`, `plot/`, `smoke/`, and
`results/` with every CSV. Versions of every library, the nginx build, and the instance
types are pinned and printed in each CSV header. `make smoke` runs every stage on R1-1h
and R2-1h with a local S3 server and a local HTTP server. The 1 h cuts let a reviewer run
the format benchmark on a laptop against local disk and any S3 endpoint. The license is
Apache-2.0.
