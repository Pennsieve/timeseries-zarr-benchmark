# Running the benchmarks on AWS

This runbook creates the machines, runs every stage of the [benchmark plan](plan.md)
in its order, copies the results back, and destroys the machines. Each step
names the host it runs on. Commands on an instance run from
`~/timeseries-zarr-benchmark`, the clone that the Terraform user data makes.

The run takes about 8.5 h from the first apply to the destroy. Generation,
upload, and the pull to the file server take about 1 h. The format benchmark
takes 2.5 to 3.5 h. The calibration session takes about 2 h, and the resampling
benchmark about 2 h. Start each long
step inside `tmux` on the instance so that a dropped SSH connection does not
stop it. Every stage can run again with the same options and continues where it
stopped, as the README describes.

## Before you start

You need Terraform 1.9 or later, the AWS CLI v2 with credentials for the
account, and `jq`. The instances clone the repository from `repo_url` at
`repo_commit` and install `envs/requirements.lock`, so push the commit and the
lock file first.

`make lock` resolves the lock file for Linux x86_64 and Python 3.12 from any
host. Before you start, run `make lockcheck` on Linux x86_64. It installs the
lock into a new venv in a temporary directory and imports every module of every
stage. Without a Linux machine, run it in a `python:3.12` container with the
repository mounted.

## Create the machines

On your machine:

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`. Set `operator_cidr` to the address you connect from,
and `repo_url` and `repo_commit` to the pushed commit. The other variables and
their defaults are in `variables.tf`. Set `bucket_force_destroy = true` if
`terraform destroy` should delete the bucket with its objects.

The default `load_generator_count` is 20. A generator runs at most 100 server
users, or `USERS_PER_GENERATOR["precomputed"]` precomputed users. A generator
whose sustained CPU or receive rate passes half its capacity may limit the run.
The orchestrator then repeats the run with half the users per generator. A run
that stays above the limit, or that has too few generators for a repeat, stays
in the results. `resampling-runs.csv` marks it in the `client_limited` column
and names the resource in `client_limit`. The precomputed runs at 1000 users
need more than 20 generators, so they run on 20 and are marked client limited.

Generation and the format benchmark use no load generator. Create the other
machines first, then 2 load generators for the calibration session, and the 20
load generators before the resampling benchmark:

```bash
terraform init
terraform apply -var load_generator_count=0
```

The apply creates the file server, the benchmark client, the resampling server,
and the bucket. Each instance installs its packages and the repository after it
boots, and writes `/etc/tszbench/ready` when that setup ends. The client has 32
vCPU. A gateway endpoint keeps S3 traffic inside the VPC, off the internet
gateway.

The file server, the client, and the resampling server ignore later changes to
the image and the user data, so the second apply never replaces them or their
data volumes. To rebuild one of them on purpose, run `terraform apply
-replace=aws_instance.client` with its resource name. Pin the image so that the
load generators of a later apply run the same image as the other machines:

```bash
echo "ami_id = \"$(terraform output -raw ami_id)\"" >> terraform.tfvars
```

Keep `repo_commit` unchanged until the calibration session ends. The load
generators clone that commit when an apply creates them. The section on the
calibration results says when to change it.

## Read the outputs and the hosts file

On your machine, still in `terraform/`:

```bash
terraform output -raw ssh_private_key > ~/.ssh/tszbench
chmod 600 ~/.ssh/tszbench
terraform output -raw hosts_file > hosts.txt
CLIENT=$(terraform output -json public_ips | jq -r .client)
```

`hosts.txt` holds one `role private_ip` line per instance. The roles are
`file_server`, `client`, `resampling_server`, and `load_generator`. The client
reaches every other host over SSH, so copy the key and the hosts file to it:

```bash
scp -i ~/.ssh/tszbench ~/.ssh/tszbench ubuntu@${CLIENT}:.ssh/id_ed25519
scp -i ~/.ssh/tszbench hosts.txt ubuntu@${CLIENT}:hosts.txt
ssh -i ~/.ssh/tszbench ubuntu@${CLIENT}
```

Every step from here runs on the client unless it says otherwise. Set these
variables in each new shell:

```bash
cd ~/timeseries-zarr-benchmark
BUCKET=$(cat /etc/tszbench/bucket)
REGION=$(cat /etc/tszbench/region)
FILE_SERVER=$(awk '$1 == "file_server" {print $2}' ~/hosts.txt)
SERVER=$(awk '$1 == "resampling_server" {print $2}' ~/hosts.txt)
```

`hosts.txt` has no `load_generator` line yet. Wait until every host has
finished its setup:

```bash
for host in $(awk '{print $2}' ~/hosts.txt); do
    ssh -o StrictHostKeyChecking=accept-new $host test -f /etc/tszbench/ready \
        && echo "$host ready"
done
```

## Generate the recordings

The client writes the recordings to its data volume at `/data`. The format
benchmark reads the 1 h cuts and R4, so they are written in every format. They are
also written as a raw-only TS-Zarr bundle for the storage table. The resampling
benchmark reads R1, R2, and R3 at full length, which are written in TS-Zarr only.
The plan generates R3-1h before R3. Run one recording per process:

```bash
for recording in R1-1h R2-1h R3-1h R4; do
    .venv/bin/python -m generate --recording $recording --root /data \
        --results results --workers 32
done
for recording in R1 R2 R3; do
    .venv/bin/python -m generate --recording $recording --root /data \
        --results results --workers 32 --formats tszarr
done
```

`--workers` sets the threads that synthesize channels at once. The client has
32 vCPU. Each recording writes `results/generation-<recording>.csv`. The option
`--formats` adds a format to a recording already written, and the generation CSV
keeps the rows of its other formats.

## Upload to S3

Each upload times `aws s3 cp` of every format of one recording and writes
`results/upload-<recording>.csv`. The 1-byte object goes up with each recording.

```bash
for recording in R1-1h R2-1h R3-1h R4 R1 R2 R3; do
    .venv/bin/python -m upload s3 --root /data --bucket $BUCKET --region $REGION \
        --recording $recording
done
```

## Pull to the file server

The file server copies the recordings of the format benchmark and the 1-byte
object to its data volume, which nginx serves. It copies every format but the
raw-only TS-Zarr bundle:

```bash
ssh $FILE_SERVER "cd timeseries-zarr-benchmark && .venv/bin/python -m upload pull \
    --root /data --bucket $BUCKET --region $REGION \
    --recordings R1-1h,R2-1h,R3-1h,R4"
```

## Measure storage

```bash
.venv/bin/python -m upload storage --root /data \
    --recordings R1-1h,R2-1h,R3-1h,R4,R1,R2,R3
```

This writes `results/storage.csv`, which Table 2 reads. Table 2 has a row for
the raw-only TS-Zarr bundle. Measuring a recording again replaces its rows and
keeps the rows of the others.

## Run the format benchmark

The format benchmark runs every combination in its own process and drops the
page cache of the client and of the file server before each one. Each
combination stops after the read that brings the sum of its timed read seconds
to at least `--read-budget-s`, which is 60 s, once it has `--min-reads` reads,
which is 1. A read is never interrupted. The slowest single reads fetch HDF5
chunks from S3 one after the other and take several minutes to half an hour
each. Most combinations end in seconds.

Run the grid:

```bash
.venv/bin/python -m format_benchmark \
    --local-root /data --http-root http://$FILE_SERVER/ --s3-root $BUCKET \
    --drop-caches "sudo /usr/local/sbin/drop-caches" \
    --drop-caches "ssh $FILE_SERVER sudo /usr/local/sbin/drop-caches"
```

The defaults run R1-1h, R2-1h, R3-1h, and R4 on every store, format, read
kind, and window. The read kinds are `one`, `channels4`, `channels16`,
`channels64`, and `all`. The three `channels` kinds read a block of that many
channels at the 10 s and 10 min windows, and a recording with no more channels
than the block skips them. The R4 blocks at 10 min are the slowest
combinations of the grid. NWB HDF5 with channel chunks reads the 64 channels of
`channels64` from S3 one after the other, so one such read takes several hours.
Each combination writes its part under `results/format-parts/`. The merge at the end writes `results/format-reads.csv`
and `results/format-opens.csv` from every finished part, so a command that runs
one recording or store keeps the rows of the others. The same command run again
skips the finished combinations.

## Calibrate the service and the load generators

The calibration session sets two values before the grid. `TASK_THREADS` is the
number of tasks that each pool process of the service runs at once.
`USERS_PER_GENERATOR["precomputed"]` is the most precomputed users on one load
generator. Both sweeps run on R3, which the upload step already put in S3.

On your machine, in `terraform/`, create 2 load generators. The plan must add
2 `aws_instance.load_generator` and change nothing else:

```bash
terraform plan -var load_generator_count=2
terraform apply -var load_generator_count=2
terraform output -raw hosts_file > hosts.txt
CLIENT=$(terraform output -json public_ips | jq -r .client)
scp -i ~/.ssh/tszbench hosts.txt ubuntu@${CLIENT}:hosts.txt
```

On the client, run the setup check from the section on the outputs again. All 5
hosts must print `ready`.

### Sweep the task threads

The server method runs R3 at the 10 s and 10 min windows, with 10 and 100 users.
Each value of `--task-threads` is one pass. Restart the service between values,
because the option is read at startup. Each pass writes its runs to its own
directory:

```bash
for value in 1 4 8 16 32; do
    ssh $SERVER "pkill -f '[r]esampling_benchmark.service'"
    ssh $SERVER "cd timeseries-zarr-benchmark && nohup .venv/bin/python -m resampling_benchmark.service \
        --s3-root $BUCKET --task-threads $value > service.log 2>&1 < /dev/null &"
    until curl -sf -o /dev/null http://$SERVER:8000/health; do sleep 5; done
    .venv/bin/python -m resampling_benchmark.load.orchestrate --hosts ~/hosts.txt \
        --s3-root $BUCKET --methods server --recordings R3 --windows 10,600 \
        --users 10,100 --out results/calibration-raw/threads-$value
done
.venv/bin/python -m resampling_benchmark.calibrate server \
    --input results/calibration-raw/threads-*
```

The last command writes `results/calibration-server.csv`, one row per run, with
`task_threads` from the `server-metadata.json` of the run. Choose the smallest
value whose `requests_per_second` is within 5% of the best value at 100 users on
both windows, and whose `server_memory_max` stays below 75% in every run. At the
chosen value, check three facts:

- The 10 s window at 100 users shows `server_cpu_sustained` above 70, or an
  `s3_requests_per_second` that no longer rises with `--task-threads`.
- The 10 min window at 10 users has a `median_s` no higher than the one of
  `--task-threads 1`.
- The `server_memory_max` of the 10 min window at 100 users is below 75.

### Sweep the users per generator

The precomputed method runs R3 at the 10 s and 1 h windows with 1, 5, 10, 20,
and 40 users. All users of a run must sit on one generator, so the hosts file of
this sweep lists one load generator. Runs above `USERS_PER_GENERATOR["precomputed"]`
log the shortfall and get the `client-limited` file, which the calibration does
not read:

```bash
awk '$1 != "load_generator" || ++n == 1' ~/hosts.txt > ~/hosts-one-generator.txt
.venv/bin/python -m resampling_benchmark.load.orchestrate \
    --hosts ~/hosts-one-generator.txt --s3-root $BUCKET --methods precomputed \
    --recordings R3 --windows 10,3600 --users 1,5,10,20,40 \
    --out results/calibration-raw/generator
.venv/bin/python -m resampling_benchmark.calibrate generator \
    --input results/calibration-raw/generator
```

The last command writes `results/calibration-generator.csv`, one row per run.
Choose the most users whose `generator_cpu_sustained` stays below 0.5 and whose
`median_s` stays within 10% of the 1-user median, at both windows. At twice that
value, one of the two criteria must fail. If neither fails, the limit lies above
the twice value, so add larger user counts to the sweep.

### Set the chosen values

Set `TASK_THREADS` in `resampling_benchmark/service/engine.py` and the
`precomputed` entry of `USERS_PER_GENERATOR` in
`resampling_benchmark/load/orchestrate.py`. Commit and push both. Then set
`repo_commit` in `terraform/terraform.tfvars` to the new commit, and check it out
on the client and on the resampling server with `git fetch && git checkout
<commit>`. The 20 load generators clone that commit when they are created.

Stop the service and destroy the 2 load generators. On your machine, in
`terraform/`:

```bash
ssh $SERVER "pkill -f '[r]esampling_benchmark.service'"
terraform plan -var load_generator_count=0
terraform apply -var load_generator_count=0
```

The plan must destroy 2 `aws_instance.load_generator` and change nothing else.

## Create the load generators

On your machine, in `terraform/`, create the 20 load generators with the
default count. The plan must add 20 `aws_instance.load_generator` and change
nothing else. Stop if it replaces another instance:

```bash
terraform plan
terraform apply
terraform output -raw hosts_file > hosts.txt
CLIENT=$(terraform output -json public_ips | jq -r .client)
scp -i ~/.ssh/tszbench hosts.txt ubuntu@${CLIENT}:hosts.txt
```

On the client, run the setup check from the section on the outputs again. It
reads the new `~/hosts.txt`, and all 23 hosts must print `ready`.

## Run the resampling benchmark

Start the resampling service on the resampling server. It runs one pool process
per vCPU, and each process opens every recording at startup. One request fans
out over every process of the pool, and each process runs up to `TASK_THREADS`
tasks at once. The option `--task-threads` replaces that default for one start:

```bash
ssh $SERVER "cd timeseries-zarr-benchmark && nohup .venv/bin/python -m resampling_benchmark.service \
    --s3-root $BUCKET > service.log 2>&1 < /dev/null &"
```

Wait until the service answers, which happens after every process has opened the
recordings:

```bash
until curl -sf -o /dev/null http://$SERVER:8000/health; do sleep 5; done
```

Before the grid, time the largest request with one user. It must finish in
under 180 s, the request timeout of the load generators:

```bash
curl -s -o /dev/null --max-time 600 -w '%{http_code} %{time_total} s\n' \
    "http://$SERVER:8000/minmax?recording=R3&start_us=0&end_us=86400000000"
```

The output starts with `200`. If the time is 180 s or more, stop here. Every R3
and R4 run with long windows would time out.

The results of the 2026-11-02 run stay under `results/resampling-raw-2026-11-02/`
on your machine, and the plots leave them out. Before you copy the new results
back, move the old directory there with `mv results/resampling-raw
results/resampling-raw-2026-11-02`. Start the grid without `--skip-done`.

Then run the grid from the client:

```bash
.venv/bin/python -m resampling_benchmark.load.orchestrate --hosts ~/hosts.txt --s3-root $BUCKET
```

The orchestrator waits for the service, then starts the monitors and the load
generators of each run over SSH. Before each server run it waits until
`/health` reports no task in flight, and writes the wait to `idle-wait-s.txt`
in the run directory. A run stops sending requests 5 min after it starts. User
counts run in ascending order. When the median request of a run fails, the
orchestrator does not run the larger user counts of that method, recording, and
window. It writes a `not-run` file in each of their run directories, which names
the run that failed. It saves the run metadata that `/health` reports to
`server-metadata.json` there. It copies each run's CSVs to
`results/resampling-raw/<run>/` and writes a `complete` file there when every
generator exits with status 0. It samples the CPU and the receive rate of each load generator once per second.
When the highest 10 s mean of either passes half its capacity, it repeats the
run with half the users per generator. It writes a `client-limited` file, which
holds `cpu` or `link`, when a run stays above that limit or has fewer generators
than its users need. Add `--skip-done` when you
start it again after a stop. It skips the runs that have a `complete` file. It
decides again which runs are not run, from the runs of fewer users. It never
deletes a complete run. It moves a partial run that is now not run to
`results/resampling-raw-discarded/`.

## Summarize

```bash
.venv/bin/python -m resampling_benchmark.summarize
```

This writes `results/resampling-requests.csv` and
`results/resampling-runs.csv`. The runs file also holds the sustained CPU and
receive rate of the load generators, the client CPU-seconds per request, and the
S3 requests and bytes per second of each run. Medians and percentiles count each failed
request as 180 s, or as its own seconds when those are longer. Their headers
copy the server metadata as `server.<key>` lines. They also hold
`generator.instance`, `generator.ami`, `first_run_date`, and `last_run_date`
from the load generator CSVs.

## Check the methods agree

The service must still run:

```bash
.venv/bin/python -m resampling_benchmark.check --server-url http://$SERVER:8000 --s3-root $BUCKET
```

The check writes `results/resampling-check.csv` and exits with status 1 when a
window differs. Stop the service when it passes:

```bash
ssh $SERVER "pkill -f '[r]esampling_benchmark.service'"
```

## Plot

```bash
.venv/bin/python -m plot --results results --out figures
```

The figures, tables, `methods.txt`, `captions.txt`, and `cost.txt` go to
`figures/`. The prices of the cost paragraph are in `plot/prices.json`. Check
that its date and prices match the run before you plot.

On your machine, copy the results back:

```bash
scp -r -i ~/.ssh/tszbench ubuntu@${CLIENT}:timeseries-zarr-benchmark/results .
scp -r -i ~/.ssh/tszbench ubuntu@${CLIENT}:timeseries-zarr-benchmark/figures .
```

## Destroy the machines

On your machine, in `terraform/`:

```bash
terraform destroy
```

The bucket is deleted only when it is empty or `bucket_force_destroy` is true.
To empty it first, run `aws s3 rm s3://<bucket> --recursive` with the name that
`terraform output -raw bucket_name` returns.
