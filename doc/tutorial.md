# 1. Workload Generation
This section describes how to generate point, range, and join workloads consistent with the evaluation methodology of our project. To generate the workload, please use `utils/generate_query.py` .All workloads are derived from sorted uint64 key files (e.g., SOSD datasets) stored in binary format.

The generator is designed to (i) approximate realistic skewed access patterns, (ii) support reproducibility, and (iii) produce workloads directly consumable by the FALCON benchmarking pipeline.

## 1.1. Dataset Assumptions
All data files must satisfy :
```text
Type: uint64
Order: strictly sorted
Uniqueness: unique keys (recommended)
Storage format: raw binary file
```

## 1.2. Point Query Workloads

We provide a realistic mixture generator:
```python
generate_realistic_queries_from_data(keys, num_queries, seed)
```
#### Mixture Model

The workload is composed of three components:

|Component	|Ratio	|Distribution	|Purpose|
| ------------ | --------- | --------- | --------- |
|Hotspot|   40%	|Local Zipf	|Simulate temporal locality|
|Global Zipf|30%	|Zipf(α≈1.2)|	Simulate skewed popularity|
|Uniform|	30%	|Uniform	|Baseline randomness|


- **Hotspot Queries**
Select `num_hotpots = 5` regions. Each region covers 1% of the key space. Inside each region, sample with Zipf(1.5). This models clustered hot keys commonly observed in real systems.

- **Global Zipf Queries**
Zipf distribution over full key domain. Skew parameter α ≈ 1.2. This approximates long-tail access patterns.

- **Uniform Queries**
Uniform sampling without replacement. Prevents fully skew-dominated workload.

#### Generation Example
```python
queries = generate_realistic_queries_from_data(
    keys,
    num_queries=1000000,
    seed=42
)
queries.tofile("<dataset>.query.bin")
```

## 1.3. Range Query Workloads
We support two types of range query generators:

#### Synthetic Key-Space Ranges
```python
generate_range_queries(...)
```
Generates ranges based purely on key domain size.

|Parameter|	Meaning|
| -------- | --------- |
|keys|	key domain|
|start_dist|	uniform or normal|
|length_dist|	uniform or exponential|
|max_length|	maximum range length|
|exp_scale|	exponential scale|
|num_queries|	number of queries|
|seed|	random seed|

Example:
```python
queries = generate_range_queries(
    num_queries=4000000,
    key_space_size=8000000000,
    start_dist='uniform',
    length_dist='uniform',
    max_length=5000000
)
```

#### Data-Aware Range Queries (Recommended)
```python
generate_range_queries_from_data(...)
```
This version:
- Samples start positions in index space
- Samples range length in number of keys
- Converts index ranges back to key ranges

This ensures:
- Ranges correspond to actual key distribution
- No invalid gaps
- Stable range selectivity

|Parameter|	Meaning|
| -------- | --------- |
|keys|	key domain|
|max_length_keys|	Maximum range size (in key count)|
|start_dist|	uniform or normal|
|length_dist|	uniform or exponential|
|exp_scale|	exponential scale|
|num_queries|	number of queries|
|seed|	random seed|

Example:
```python
queries = generate_range_queries_from_data(
    keys,
    num_queries=1000000,
    start_dist='uniform',
    length_dist='uniform',
    max_length_keys=5000,
    seed=42
)

queries.tofile("<dataset>.range.bin")
```
Output format:
```python
uint64[Q][2]   # [lo_key, hi_key]
```

## 1.4. Unique Join Workloads
For join experiments, we require unique probe keys.
```python
sample_unique_mixture(...)
```
#### Characteristics
- Enforces unique queries
- Maintains mixture skew
- Uses oversampling to guarantee uniqueness

#### Components
- Hotspot
- Global Zipf
- Uniform fallback

#### Parameters
|Parameter|	Meaning|
| -------- | --------- |
|keys|	key domain|
|oversample|	candidate amplification factor|
|return_sorted|	sorted vs shuffled|
|k|	number of queries|
|hotpot_ratio|	hotspot ratio|
|zipf_ratio|	global zipf ratio|
|seed|	random seed|
|strict|	enforce uniqueness or fallback|

Example:
```python
queries = sample_unique_mixture(
    keys,
    k=1000000,
    hotpot_ratio=0.2,
    zipf_ratio=0.2,
    oversample=100,
    return_sorted=False
)
queries.tofile("<dataset>.1Mjoin.bin")
```

## 1.5. Join Partition Generation
For hybrid join execution, use:
```python
join_partition(...)
```
This function partitions sorted probe queries into segments, selecting per-partition execution strategy using a cost model.

#### Decision Model
For a window of size N:

```python
C_point = αN + λ_point * union_pages    # point cost
C_range = βK + η + λ_range * K  #range cost
```
Where:
- K = number of distinct pages covered
- union_pages = union of page intervals
- epsilon = PGM window size

A partition switches to range when:
```python
C_range <= (1 - γ) * C_point
```

Additional controls:

|Parameter|	Purpose|
| -------- | --------- |
|N_min|	Minimum partition size|
|K_max|	Hard cap on page span|
|gamma|	Hysteresis margin|

Example:
```python
join_partition(
    keys,
    queries,
    alpha=1.168e-06,
    beta=5.831e-06,
    eta=0.121,
    lambda_point=2.763e-05,
    lambda_range=4.714e-06,
    page_size=4096,
    key_size=8,
    epsilon=16,
    N_min=1000,
    K_max=8192,
    lengths_file="query.par",
    bitmap_file="query.bitmap"
)
```
Outputs:
- lengths: number of queries per partition
- bitmap: 0 = point, 1 = range

These files are directly consumed by join benchmarks.

# 2. I/O Cost Estimation
This section describes how we use our implemented functions to estimate I/O cost under a given memory budget. We only briefly summarize the underlying model and focus on how the estimation is carried out in practice. To estimate I/O cost, please use `utils/optimalEpsilon.py`. 

## 2.1. Overview
Our estimation framework predicts the average number of physical page I/Os per query under:
- total memory budget $M$,
- PGM error bound $\epsilon$, 
- page size $ps$,
- items per page $ipp$,
- dataset size $n$,
- query workload (point, range).

At a high level:

$$
Cost = (1-h)\times DAC
$$

where
- $h$ is the estimated buffer hit ratio,
- $DAC$ is the estimated data access cost computed from the learned index behavior. 

The hit ratio is estimated using Che’s approximation for LRU caching.

## 2.2. Point Query Cost Estimation
**1. Memory decomposition**
For each $\epsilon$, the function computes:
- index size:

$$
M_{index}=\frac{n\cdot seg_size}{2\epsilon}
$$
- Buffer size:

$$
M_{buffer}=M-M_{index}
$$
- Buffer capacity in pages:

$$
C=\frac{M_{buffer}}{ps}
$$

**2. Logical page accesses**
The expected data access cost (DAC) is computed via:
```python
expected_DAC(epsilon, ipp)
```
**3. Page popularity extraction**
We estimate page request distribution directly from the query trace using:
```python
estimate_page_counts_from_queryfile(...)
```
This function maps each query to a data position, expands it using a triangular kernel and aggregates counts at page granularity.

**4. Buffer hit ratio estimation**
We compute the hit ratio using:
```python
sample_ratio(C, total_pages, q)
```
This internally solves Che's characteristic time equation and returns the estimated hit ratio.

**5. Final cost**

$$
Cost = (1-h)\cdot \mathbb{E}[DAC]
$$

## 2.3. Range Query Cost Estimation
**1.** True range boundaries are mapped to data positions.

**2.** The expected Range DAC (RDAC) is computed via:
```python
get_RDAC(rlo, rhi, epsilon, ipp)
```
This function calculates the expected number of pages touched under bounded prediction error.

**3.** Expected per-page request counts are computed using:
```python
estimate_page_counts_from_range_queryfile(...)
```
This analytically distributes probability mass to:
- core pages (always touched),
- boundary pages (fractional probability).

**4.** The page distribution is normalized and passed to `sample_ratio(...)` to estimate the buffer hit ratio.

**5.** Final cost:
$$
Cost = (1-h)\cdot\mathbb{E}[RDAC]
$$


## 2.4. Epsilon Sweeping
For a user-given candidate set of $\epsilon$, `getExpectedCostPerEpsilon()` and `getExpectedRangeCostPerEpsilon()` interface iterate over feasible $\epsilon$ values and compute:
- index size,
- buffer capacity,
- hit ratio,
- expected I/O cost,

and output the cost-$\epsilon$ curve for parameter tuning. The results are stored in `<workload>.log` by default. The directory of log files should be configured first in `utils/optimalEpsilon.py`.

# 3. Benchmark


## 3.1. FALCON Benchmark Suite Quickstart

Before benchmarking, please make sure that the dataset and workload files are located in the configured directory. And use the command below to compile the benchmark binaries:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

Run:

```bash
./falcon_bench --help
./falcon_bench <subcommand> --help
```

#### Command Structure

```bash
./falcon_bench <subcommand> [options...]
```

Available subcommands:

* `epsilon`  : vary epsilon values
* `threads`  : vary thread count list
* `batch`    : vary batch size list
* `worker`   : vary producer/worker counts
* `rangeEps` : vary epsilon for range queries
* `rangeSig` : range query used to locate a single point (ordered workload)
* `join`     : hybrid join benchmark


#### Common Parameters

Even though each subcommand has its own config struct, most share these meanings:

* `--dataset <name>`: dataset basename (e.g., `books`, `fb`, `wiki_ts`)
* `--keys <N>`: number of keys to load / build index on
* `--memory <MB>`: memory budget for page buffer / cache (MB)
* `--policy <none|fifo|lru|lfu>`: cache policy (case-insensitive). Internally mapped to `CachePolicy::{NONE,FIFO,LRU,LFU}` via transformer. FIFO/LRU/LFU correspond to sharded cache implementations.   


#### Subcommand Reference + Examples

**1) `epsilon` — Vary epsilon values (point workload)**

**Options**

* `--dataset <name>` **required**
* `--keys <N>` **required**
* `--epsilon <e1> <e2> ...` **required** (list)
* `--memory <MB>` **required**
* `--policy <none|fifo|lru|lfu>` optional (but you probably want it)

**Example**

```bash
for memory in 10 20 40 60; do
    ./falcon_bench epsilon --dataset books_10M_uint64_unique \
                --keys 10000000 \
                --epsilon 2 4 6 8 10 12 14 16 18 20 24 32 48 64 128\
                --repeats 3 \
                --memory $memory \
                --policy LRU
done
```

**2) `threads` — Vary thread counts**

**Options**

* `--dataset <name>` **required**
* `--keys <N>` **required**
* `--threads <t1> <t2> ...` **required** (list)
* `--policy <none|fifo|lru|lfu>` optional
* `--memory <MB>` optional

**Example**

```bash
./falcon_bench threads --dataset books_200M_uint64_unique \
                --keys 200000000 \
                --threads 1 2 4 8 16 32 64 128 \
                --repeats 3 \
                --memory 0 \
                --policy NONE
```

**3) `batch` — Vary batch sizes**

**Options**

* `--dataset <name>` **required**
* `--keys <N>` **required**
* `--batch <b1> <b2> ...` **required** (list)
* `--memory <MB>` optional
* `--policy <none|fifo|lru|lfu>` optional

**Example**

```bash
./falcon_bench batch --dataset books_200M_uint64_unique \
                --keys 200000000 \
                --batch 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 \
                --memory 0 \
                --policy NONE
```

**4) `worker` — Vary workers & producers**

**Options**

* `--dataset <name>` **required**
* `--keys <N>` **required**
* `--workers <w1> <w2> ...` **required** (list)
* `--producers <p1> <p2> ...` **required** (list)
* `--memory <MB>` optional
* `--policy <none|fifo|lru|lfu>` optional

**Example**

```bash
./falcon_bench worker --dataset books_200M_uint64_unique \
                --keys 200000000 \
                --producers 1 2 4 8 16 32 64 128 \
                --workers 1 2 4 8 16 32 64 128 \
                --memory 0 \
                --policy NONE
```

**5) `rangeEps` — Vary epsilon for range queries**

**Options**

* `--dataset <name>` **required**
* `--keys <N>` **required**
* `--epsilon <e1> <e2> ...` **required** (list)
* `--memory <MB>` **required**
* `--policy <none|fifo|lru|lfu>` optional

**Example**

```bash
for memory in 10 20 40 60; do
    ./falcon_bench rangeEps --dataset books_10M_uint64_unique \
                --keys 10000000 \
                --epsilon 2 4 6 8 10 12 14 16 18 20 24 32 48 64 128 \
                --memory $memory \
                --policy LRU
done
```

**6) `rangeSig` — Range-to-single-point (ordered query file)**

This is intended for an **ordered** query workload; your help string says so.

**Options**

* `--dataset <name>` **required**
* `--keys <N>` **required**
* `--query <filename>` **required**
* `--memory <MB>` **required**
* `--policy <none|fifo|lru|lfu>` optional

**Example**

```bash
./falcon_bench rangeSig --dataset books_200M_uint64_unique \
            --keys 200000000 \
            --query books_200M_uint64_unique.1Mtable5.bin \
            --memory 0 \
            --policy NONE
```

**7) `join` — Hybrid join benchmark**

**Options**

* `--dataset <name>` **required**
* `--keys <N>` **required**
* `--query <filename>` **required**
* `--memory <MB>` **required**
* `--policy <none|fifo|lru|lfu>` optional

**Example**

```bash
./falcon_bench join --dataset books_200M_uint64_unique \
            --keys 200000000 \
            --query books_200M_uint64_unique.1Mtable5 \
            --memory 0 \
            --policy NONE
```

## 3.2. Benchmark for Baselines

To evaluate the performance of different disk-based index baselines, we provide a unified benchmarking script `/experiments/run/examples/baseline_bench.sh`. This script automatically runs all baseline experiments used in our evaluation, including both point queries and range queries for PGM-disk, AULID and B+Tree.

The script sequentially invokes the corresponding benchmark programs:

- pgm_disk_test

- pgm_disk_range_test

- aulid_test

- aulid_range_test

- bplustree_test

- bplustree_range_test

Each program measures multi-threaded query performance under different thread counts and repeats the experiment multiple times to obtain stable results. The outputs are automatically written to CSV files for later analysis.

Users can simply execute the provided script to run all baseline experiments:

```bash
bash baseline_bench.sh
```

The dataset name, number of keys, maximum thread exponent, and repeat count can be configured at the beginning of the script.


## 3.3 Page Fetch Strategy Test
This benchmark evaluates the performance difference between two **page fetch strategies** when using a PGM-index to locate keys on disk.

The benchmark measures how different strategies fetch candidate pages predicted by the PGM index and evaluates the resulting **query throughput (QPS)** under multi-threaded workloads.

#### Program Usage

```bash
pgm_fetch_test \
    --data <dataset_file> \
    --queries <query_file> \
    [--strategy all|one] \
    [--threads N] \
    [--io psync|libaio|uring] \
    [--direct 0|1]
```

#### Parameters

|Option	|Description|
| -- | -- |
|--data	|dataset file (required)|
|--queries	|query workload file|
|--strategy	|page fetch strategy (all or one)|
|--threads	|number of worker threads|
|--io	|I/O backend (psync, libaio, uring)|
|--direct	|enable O_DIRECT (1 = on)|

#### Automated Experiment Script

We provide a script that runs multiple experiments automatically in `experiments/run/examples/strategy_test.sh`. The script varies different parameters and runs the benchmark multiple times. Then the script computes mean QPS and standard deviation for each parameter combination.

**Script Usage**

Run:

```bash
bash strategy_test.sh
```

Script configuration:

```bash
BIN="./page_fetch_test"

DATA=".../books_200M_uint64_unique"
QUERIES=".../books_200M_uint64_unique.1Kquery.bin"

STRATEGIES=("all" "one")
THREADS=(1 4 16 64 256)
REPEAT=10
...
```