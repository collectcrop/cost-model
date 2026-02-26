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

