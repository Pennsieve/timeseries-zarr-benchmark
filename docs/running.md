# Running the benchmarks on AWS

This runbook creates the machines, runs every stage of the [benchmark plan](plan.md)
in its order, copies the results back, and destroys the machines. Each step
names the host it runs on. Commands on an instance run from
`~/timeseries-zarr-benchmark`, the clone that the Terraform user data makes.

Several steps run for hours. Start them inside `tmux` on the instance so that a
dropped SSH connection does not stop them. Every stage can run again with the
same options and continues where it stopped, as the README describes.

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

The default `load_generator_count` is 20. At 1000 users, 10 generators run 100
users each. A generator whose receive rate passes half its link may limit the
run. The orchestrator then repeats the run with half the users per generator,
which is 20 generators of 50 users, so the repeat also runs at 1000 users. A
repeat that still passes half the link stays in the results, and
`resampling-runs.csv` marks it in the `link_limited` column.

Generation and the format benchmark take days and use no load generator. Create
the other machines first, and the 20 load generators before the resampling
benchmark:

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
load generators of the second apply run the same image as the other machines:

```bash
echo "ami_id = \"$(terraform output -raw ami_id)\"" >> terraform.tfvars
```

Keep `repo_commit` unchanged until the resampling benchmark ends. The load
generators clone that commit when the second apply creates them.

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
scp -i ~/.ssh/tszbench ~/.ssh/tszbench ubuntu@$CLIENT:.ssh/id_ed25519
scp -i ~/.ssh/tszbench hosts.txt ubuntu@$CLIENT:hosts.txt
ssh -i ~/.ssh/tszbench ubuntu@$CLIENT
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

The client writes every format of each recording to its data volume at `/data`.
The plan generates R3-1h before R3. Run one recording per process:

```bash
for recording in R1 R2 R3-1h R3 R4; do
    .venv/bin/python -m generate --recording $recording --root /data \
        --results results --workers 32
done
```

`--workers` sets the threads that synthesize channels at once. The client has
32 vCPU. Each recording writes `results/generation-<recording>.csv`.

## Upload to S3

Each upload times `aws s3 cp` of every format of one recording and writes
`results/upload-<recording>.csv`. The 1-byte object goes up with each recording.

```bash
for recording in R1 R2 R3 R4; do
    .venv/bin/python -m upload s3 --root /data --bucket $BUCKET --region $REGION \
        --recording $recording
done
```

## Pull to the file server

The file server copies the bucket to its data volume, which nginx serves:

```bash
ssh $FILE_SERVER "cd timeseries-zarr-benchmark && .venv/bin/python -m upload pull \
    --root /data --bucket $BUCKET --region $REGION"
```

## Measure storage

```bash
.venv/bin/python -m upload storage --root /data --recordings R1,R2,R3,R4
```

This writes `results/storage.csv`, which Table 2 reads. Measuring a recording
again replaces its rows and keeps the rows of the others.

## Run the format benchmark

The format benchmark runs every combination in its own process and drops the
page cache of the client and of the file server before each one. Each
combination stops after the read that brings the sum of its timed read seconds
to at least `--read-budget-s`, which is 1 h, once it has `--min-reads` reads,
which is 3.

Before the full grid, time the slowest kind of read: R3 in the
`nwb-hdf5-channel` format, every channel, a 3600 s window, on the S3 store.
HDF5 fetches the 175,800 chunks of that read one after the other. One read is
enough, and `--results results-timing` keeps the timing run out of the real
results:

```bash
.venv/bin/python -m format_benchmark \
    --recordings R3 --stores s3 --formats nwb-hdf5-channel --read-kinds all \
    --windows 3600 --max-repeats 1 --results results-timing \
    --s3-root $BUCKET \
    --drop-caches "sudo /usr/local/sbin/drop-caches" \
    --drop-caches "ssh $FILE_SERVER sudo /usr/local/sbin/drop-caches"
```

Expect tens of minutes for the one read. Read the time from
`results-timing/format-reads.csv`. Use it to estimate the run time of the
slowest combinations before you start the grid. If one read takes longer than a
third of `--read-budget-s`, the combination reaches the budget before
`--min-reads`, so `--min-reads` sets its run time.

Then run the full grid:

```bash
.venv/bin/python -m format_benchmark \
    --local-root /data --http-root http://$FILE_SERVER/ --s3-root $BUCKET \
    --drop-caches "sudo /usr/local/sbin/drop-caches" \
    --drop-caches "ssh $FILE_SERVER sudo /usr/local/sbin/drop-caches"
```

The defaults run every recording, store, format, read kind, and window. Each
combination writes its part under `results/format-parts/`. The merge at the end
writes `results/format-reads.csv` and `results/format-opens.csv` from every
finished part, so a command that runs one recording or store keeps the rows of
the others. The same command run again skips the finished combinations.

## Create the load generators

On your machine, in `terraform/`, create the 20 load generators with the
default count. The plan must add 20 `aws_instance.load_generator` and change
nothing else. Stop if it replaces another instance:

```bash
terraform plan
terraform apply
terraform output -raw hosts_file > hosts.txt
CLIENT=$(terraform output -json public_ips | jq -r .client)
scp -i ~/.ssh/tszbench hosts.txt ubuntu@$CLIENT:hosts.txt
```

On the client, run the setup check from the section on the outputs again. It
reads the new `~/hosts.txt`, and all 23 hosts must print `ready`.

## Run the resampling benchmark

Start the resampling service on the resampling server. It runs one pool process
per vCPU, and each process opens every recording at startup. One request fans
out over every process of the pool:

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

Then run the grid from the client:

```bash
.venv/bin/python -m resampling_benchmark.load.orchestrate --hosts ~/hosts.txt --s3-root $BUCKET
```

The orchestrator waits for the service, then starts the monitors and the load
generators of each run over SSH. Before each server run it waits until
`/health` reports no task in flight, and writes the wait to `idle-wait-s.txt`
in the run directory. It saves the run metadata that `/health` reports to
`server-metadata.json` there. It copies each run's CSVs to
`results/resampling-raw/<run>/` and writes a `complete` file there when every
generator exits with status 0. It writes a `link-limited` file when a run stays
with a generator receive rate above half the link. Add `--skip-done` when you
start it again after a stop. It skips only the runs that have a `complete`
file.

## Summarize

```bash
.venv/bin/python -m resampling_benchmark.summarize
```

This writes `results/resampling-requests.csv` and
`results/resampling-runs.csv`. Medians and percentiles count each failed
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
scp -r -i ~/.ssh/tszbench ubuntu@$CLIENT:timeseries-zarr-benchmark/results .
scp -r -i ~/.ssh/tszbench ubuntu@$CLIENT:timeseries-zarr-benchmark/figures .
```

## Destroy the machines

On your machine, in `terraform/`:

```bash
terraform destroy
```

The bucket is deleted only when it is empty or `bucket_force_destroy` is true.
To empty it first, run `aws s3 rm s3://<bucket> --recursive` with the name that
`terraform output -raw bucket_name` returns.
