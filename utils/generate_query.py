import numpy as np
import random
import math
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

def generate_join_table_from_data(keys, num_queries=100000, seed=42, num_segments=5, active_segments=2):
    """
    生成带空page gap的 join query:
      - 将key space划分成num_segments段
      - 只在active_segments个segment里生成query
      - 其他segment完全空白，制造page gap
    """

    np.random.seed(seed)
    random.seed(seed)

    n = len(keys)
    seg_size = n // num_segments

    queries = []

    # 随机挑选几个segment作为"热点段"
    active_idx = random.sample(range(num_segments), active_segments)

    per_segment_queries = num_queries // active_segments

    for seg in active_idx:
        start = seg * seg_size
        end = min((seg + 1) * seg_size, n)
        segment_keys = keys[start:end]

        # 在segment内再制造热点 + 稀疏
        hotspot_ratio = 0.6
        hotspot_queries = int(per_segment_queries * hotspot_ratio)
        uniform_queries = per_segment_queries - hotspot_queries

        # hotspot: 取该segment内的小范围
        hotspot_size = max(1, int(0.05 * len(segment_keys)))
        base = random.randint(0, len(segment_keys) - hotspot_size)
        hot_indices = np.random.zipf(1.5, hotspot_queries)
        hot_indices = np.clip(hot_indices, 0, hotspot_size - 1)
        queries.extend(segment_keys[base + hot_indices])

        # uniform: 均匀取整个segment
        take = min(uniform_queries, len(segment_keys))
        print('uniform:',take)
        queries.extend(np.random.choice(segment_keys, size=take, replace=False))

    # 转成 numpy array
    queries = np.array(queries, dtype=np.uint64)

    # 升序
    join_keys = np.sort(queries)
    return join_keys

def join_partition(keys, queries, page_size=4096, key_size=8,
                   lengths_file="", bitmap_file="", delta=37, H=0, epsilon=8):
    """
    Join-Partition Algorithm
    input:
      - keys: 大表升序 key (np.uint64)
      - queries: join 表升序 key (np.uint64)
      - page_size: SSD page 大小 (默认 4096B)
      - key_size: key 大小 (默认 8B)
    output:
      - lengths: 每个 partition 的 query 数
      - bitmap: 每个 partition 的策略 (0=point, 1=range)
    """

    ipp = page_size // key_size
    num_pages = int(np.ceil(len(keys) / ipp))

    # 将 key 映射到 page_id
    key_to_page = np.arange(len(keys)) // ipp

    # 用二分搜索找到 query 的位置 -> 对应 page id
    query_idx = np.searchsorted(keys, queries)
    query_idx = np.clip(query_idx, 0, len(keys) - 1)  # 防越界
    query_pages = key_to_page[query_idx]

    # 统计每个 page 被命中的次数
    pageCount = np.bincount(query_pages, minlength=num_pages)
    # print("Page Count:", pageCount)
    # 划分连续的非零区间
    partitions = []
    start = None
    for i, cnt in enumerate(pageCount):
        if cnt > 0 and start is None:
            start = i
        elif cnt == 0 and start is not None:
            partitions.append((start, i - 1))
            start = None
    if start is not None:
        partitions.append((start, num_pages - 1))
    print("Partitions:", partitions)
    
    totalPagesRef = 0  
    # 生成结果
    lengths = []
    bitmap = []
    for (l, r) in partitions:
        N = pageCount[l:r+1].sum()
        K = r - l + 1
        totalPagesRef += K
        print("N:",N,"K:",K)
        mu = N / K
        lengths.append(N)
        tau = (ipp+delta+H/K)/(math.log(epsilon,2)+H+delta)
        print("τ:",tau)
        print("μ:",mu)
        if mu >= tau:
            bitmap.append(1)  # range
        else:
            bitmap.append(0)  # point
    print("Total Pages (Ref):", totalPagesRef)
    print("Total Queries:", sum(lengths))
    # save to file
    np.array(lengths, dtype=np.int64).tofile(lengths_file)
    np.array(bitmap, dtype=np.int8).tofile(bitmap_file)

    return lengths, bitmap


def main():
    
    sizeList = [1e7,2e7,3e7,5e7,7e7,9e7,1e8]
    datasets = ["fb","books","osm_cellids","wiki_ts"]
    """ point """
    # num_queries = 10000000
    # for dataset in datasets:
    #     for size in sizeList:
    #         print(f"[*] Generate queries for {dataset}_{int(size/1e6)}M_uint64_unique")
    #         raw = np.fromfile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique", dtype=np.uint64)
    #         keys = raw
    #         print(f"[*] Loaded {len(keys)} keys.")
    #         queries = generate_realistic_queries_from_data(keys,num_queries)
    #         queries.tofile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.query.bin")
    #         print(f"[+] save queries to {DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.query.bin successfully!")
    
    """ range """
    # num_queries = 1000000
    # queries = generate_range_queries(num_queries, 8000000000,
    #                              start_dist='uniform',
    #                              length_dist='uniform',
    #                              max_length=5000000,
    #                              exp_scale=10
    #                              )
    # queries.tofile(f"{DATASETS_DIRECTORY}range_query_fb_uu.bin")
    # print(f"[+] save queries to {DATASETS_DIRECTORY}range_query.bin successfully!")
    
    """ join """
    # datasets = ["books"]
    # num_queries = 1000000
    # for dataset in datasets:
    #     for size in sizeList:
    #         print(f"[*] Generate queries for {dataset}_{int(size/1e6)}M_uint64_unique")
    #         raw = np.fromfile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique", dtype=np.uint64)
    #         keys = raw
    #         print(f"[*] Loaded {len(keys)} keys.")
    #         queries = generate_join_table_from_data(keys,num_queries,num_segments=1,active_segments=1)
    #         queries.tofile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.{int(num_queries/1e6)}Mtable1.bin")
    #         print(f"[+] save queries to {DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.{int(num_queries/1e6)}Mtable1.bin successfully!")
    
    # datasets = ["books"]
    # num_queries = 100000
    # for dataset in datasets:
    #     for size in sizeList:
    #         print(f"[*] Generate queries for {dataset}_{int(size/1e6)}M_uint64_unique")
    #         raw = np.fromfile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique", dtype=np.uint64)
    #         keys = raw
    #         print(f"[*] Loaded {len(keys)} keys.")
    #         queries = generate_join_table_from_data(keys,num_queries,num_segments=10,active_segments=5)
    #         queries.tofile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.{int(num_queries/1e3)}Ktable2.bin")
    #         print(f"[+] save queries to {DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.{int(num_queries/1e3)}Ktable2.bin successfully!")

    
    """ partition join"""
    page_size = 4096
    H = 4
    delta = 400
    epsilon = 8
    queryfile = "books_20M_uint64_unique.100Ktable2.bin"
    dataset = "books_20M_uint64_unique"
    raw = np.fromfile(f"{DATASETS_DIRECTORY}{dataset}", dtype=np.uint64)
    keys = raw
    queries = np.fromfile(f"{DATASETS_DIRECTORY}{queryfile}", dtype=np.uint64)
    lengths_file=f"{DATASETS_DIRECTORY}{queryfile}.par".replace(".bin","")
    bitmap_file=f"{DATASETS_DIRECTORY}{queryfile}.bitmap".replace(".bin","")
    join_partition(keys,queries,page_size,lengths_file=lengths_file,bitmap_file=bitmap_file,H=H,delta=delta,epsilon=epsilon)
            
if __name__ == '__main__':
    main()