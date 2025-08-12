import numpy as np
import random
DATASETS_DIRECTORY = "/mnt/home/zwshi/Datasets/SOSD/"

def generate_realistic_queries_from_data(keys, num_queries=100000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    n = len(keys)
    queries = []

    # 1. hotpot query
    hotpot_ratio = 0.4
    hotpot_queries = int(num_queries * hotpot_ratio)
    num_hotpots = 5
    hotpot_size = int(0.01 * n)  # each hotpot 1% key
    for _ in range(num_hotpots):
        base = random.randint(0, n - hotpot_size)
        hotpot_indices = np.random.zipf(1.5, hotpot_queries // num_hotpots)
        hotpot_indices = np.clip(hotpot_indices, 0, hotpot_size - 1)
        queries.extend(keys[base + hotpot_indices])

    # 2. zipf query
    zipf_ratio = 0.3
    zipf_queries = int(num_queries * zipf_ratio)
    zipf_indices = np.random.zipf(1.2, zipf_queries)
    zipf_indices = np.clip(zipf_indices, 0, n - 1)
    queries.extend(keys[zipf_indices])

    # 3. Uniform query
    uniform_ratio = 0.3
    uniform_queries = int(num_queries * uniform_ratio)
    queries.extend(np.random.choice(keys, size=uniform_queries, replace=False))

    queries = np.array(queries[:num_queries], dtype=np.uint64)
    return queries

def generate_range_queries(num_queries, key_space_size,
                           start_dist='uniform',
                           length_dist='exponential',
                           max_length=1000000,
                           exp_scale=100):
    """
    generate range query sample
    - num_queries: query numbers
    - key_space_size: keys range
    - start_dist: distribution type of starting point, support 'uniform' or 'normal'
    - length_dist: distribution type of len, support 'uniform' or 'exponential'
    - max_length: len limit
    - exp_scale: exponential distribution factor
    """

    # generate starting point
    if start_dist == 'uniform':
        starts = np.random.randint(0, key_space_size, size=num_queries)
    elif start_dist == 'normal':
        mean = key_space_size // 2
        std = key_space_size // 6
        starts = np.random.normal(loc=mean, scale=std, size=num_queries).astype(int)
        starts = np.clip(starts, 0, key_space_size-1)
    else:
        raise ValueError("Unsupported start_dist")

    # generate length
    if length_dist == 'uniform':
        lengths = np.random.randint(1, max_length+1, size=num_queries)
    elif length_dist == 'exponential':
        lengths = np.random.exponential(scale=exp_scale, size=num_queries).astype(int) + 1
        lengths = np.clip(lengths, 1, max_length)
    else:
        raise ValueError("Unsupported length_dist")

    # calc end point, below key_space_size - 1
    ends = starts + lengths - 1
    ends = np.clip(ends, 0, key_space_size-1)

    # return [(lo, hi), ...] list
    queries = list(zip(starts, ends))
    queries = np.array(queries[:num_queries], dtype=np.uint64)
    return queries
def main():
    num_queries = 10000000
    sizeList = [1e7,2e7,3e7,5e7,7e7,9e7,1e8]
    datasets = ["fb","books","osm_cellids","wiki_ts"]
    for dataset in datasets:
        for size in sizeList:
            print(f"[*] Generate queries for {dataset}_{int(size/1e6)}M_uint64_unique")
            raw = np.fromfile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique", dtype=np.uint64)
            keys = raw[1:]
            print(f"[*] Loaded {len(keys)} keys.")
            queries = generate_realistic_queries_from_data(keys,num_queries)
            queries.tofile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.query.bin")
            print(f"[+] save queries to {DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.query.bin successfully!")
    
    # num_queries = 1000000
    # queries = generate_range_queries(num_queries, 8000000000,
    #                              start_dist='uniform',
    #                              length_dist='uniform',
    #                              max_length=5000000,
    #                              exp_scale=10
    #                              )
    # queries.tofile(f"{DATASETS_DIRECTORY}range_query_fb_uu.bin")
    # print(f"[+] save queries to {DATASETS_DIRECTORY}range_query.bin successfully!")
if __name__ == '__main__':
    main()