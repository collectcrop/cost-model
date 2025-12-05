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
    np.random.shuffle(queries)
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

def generate_range_queries_from_data(keys, num_queries,
                                     start_dist='uniform',
                                     length_dist='exponential',
                                     max_length_keys=100000,  # 以“key 个数”为单位
                                     exp_scale=100,
                                     seed=42):
    np.random.seed(seed)
    keys = np.asarray(keys, dtype=np.uint64)
    n = len(keys)
    if n == 0:
        raise ValueError("keys is empty")

    # ---- 1. 生成起点下标 start_idx ----
    if start_dist == 'uniform':
        # 均匀落在整个 key 数组的下标空间
        start_idx = np.random.randint(0, n, size=num_queries)
    elif start_dist == 'normal':
        # 以中间为均值的高斯分布, 再截断到合法下标
        mean = n // 2
        std = n // 6
        start_idx = np.random.normal(loc=mean, scale=std, size=num_queries).astype(int)
        start_idx = np.clip(start_idx, 0, n - 1)
    else:
        raise ValueError(f"Unsupported start_dist: {start_dist}")

    # ---- 2. 生成长度 (以 key 个数为单位) ----
    if length_dist == 'uniform':
        lengths = np.random.randint(1, max_length_keys + 1, size=num_queries)
    elif length_dist == 'exponential':
        lengths = np.random.exponential(scale=exp_scale, size=num_queries).astype(int) + 1
        lengths = np.clip(lengths, 1, max_length_keys)
    else:
        raise ValueError(f"Unsupported length_dist: {length_dist}")

    # ---- 3. 根据起点和长度计算终点下标 ----
    end_idx = start_idx + lengths - 1
    end_idx = np.clip(end_idx, 0, n - 1)

    # 为保险起见保证 lo_idx <= hi_idx
    lo_idx = np.minimum(start_idx, end_idx)
    hi_idx = np.maximum(start_idx, end_idx)

    # ---- 4. 映射回真实 key 值 ----
    lo_keys = keys[lo_idx]
    hi_keys = keys[hi_idx]

    queries = np.stack([lo_keys, hi_keys], axis=1).astype(np.uint64)
    return queries

def generate_join_table_from_data(keys, num_queries=100000, seed=42,
                                  num_segments=5, active_segments=2, skew=0.5):
    """
    生成带空page gap的 join query（简版）：
      - 将 key space 划分为 num_segments 段
      - 仅在 active_segments 个段里生成 query
      - 用一个超参数 skew 控制各活跃段的“密度差异”（配额不均匀程度）
        * skew < 1: 更集中（某几个段非常密）
        * skew = 1: 中等不均
        * skew > 1: 更平均
    """
    np.random.seed(seed)
    random.seed(seed)

    n = len(keys)
    seg_size = n // num_segments
    queries = []

    # 选出若干活跃段
    active_idx = random.sample(range(num_segments), active_segments)

    # 用 Dirichlet(skew) 得到活跃段权重，再用多项分布分配查询总量
    w = np.random.dirichlet(alpha=[skew] * active_segments)
    quotas = np.random.multinomial(num_queries, w)

    for j, seg in enumerate(active_idx):
        start = seg * seg_size
        end = min((seg + 1) * seg_size, n)
        segment_keys = np.asarray(keys[start:end])  # 确保是 ndarray

        this_q = int(quotas[j])
        if this_q <= 0 or segment_keys.size == 0:
            continue

        # 段内仍保持“热点 + 均匀”两部分（可按需改比例）
        hotspot_ratio = 0.6
        hotspot_queries = int(this_q * hotspot_ratio)
        uniform_queries = this_q - hotspot_queries

        # hotspot: 取 segment 内的小范围 + Zipf 采样
        hotspot_size = max(1, int(0.05 * len(segment_keys)))
        base = random.randint(0, max(0, len(segment_keys) - hotspot_size))
        hot_indices = np.random.zipf(1.5, hotspot_queries)
        hot_indices = np.clip(hot_indices - 1, 0, hotspot_size - 1)  # zipf 从1起
        queries.extend(segment_keys[base + hot_indices])

        # uniform: 段内均匀采样；为保证配额，使用 replace=True
        if uniform_queries > 0:
            queries.extend(np.random.choice(segment_keys, size=uniform_queries, replace=True))

    # 返回升序
    join_keys = np.sort(np.array(queries, dtype=np.uint64))
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
        tau = (ipp+delta+H/K)/(math.log(2*epsilon,2)+H+delta)
        print("τ:",tau)
        print("μ:",mu)
        if mu >= tau:
            bitmap.append(1)  # range
        else:
            bitmap.append(0)  # point
    print("Total Pages (Ref):", totalPagesRef)
    print("Total Queries:", sum(lengths))
    # show numbers of range and point
    range_queries = sum(N for N, b in zip(lengths, bitmap) if b == 1)
    point_queries = sum(N for N, b in zip(lengths, bitmap) if b == 0)
    print("Range queries:", range_queries)
    print("Point queries:", point_queries)
    
    # save to file
    np.array(lengths, dtype=np.int64).tofile(lengths_file)
    np.array(bitmap, dtype=np.int8).tofile(bitmap_file)

    return lengths, bitmap


def main():
    
    # sizeList = [1e7,2e7,3e7,5e7,7e7,9e7,1e8,2e8]
    # datasets = ["fb","books","osm_cellids","wiki_ts"]
    # """ point """
    # num_queries = 1000000
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
    # num_queries = 4000000
    # queries = generate_range_queries(num_queries, 8000000000,
    #                              start_dist='uniform',
    #                              length_dist='uniform',
    #                              max_length=5000000,
    #                              exp_scale=10
    #                              )
    # queries.tofile(f"{DATASETS_DIRECTORY}range_query_{int(num_queries/1e6)}M_uu.bin")
    # print(f"[+] save queries to {DATASETS_DIRECTORY}range_query_{int(num_queries/1e6)}M_uu.bin successfully!")
    
    # datasets = ["fb","books","osm_cellids","wiki_ts"]
    # size = 2e8
    # num_queries = 500000
    # for dataset in datasets:
    #     raw = np.fromfile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique",
    #                     dtype=np.uint64)
    #     keys = raw        # SOSD 已排序 key

    #     queries = generate_range_queries_from_data(
    #         keys,
    #         num_queries,
    #         start_dist='uniform',      # 或 'normal'
    #         length_dist='uniform', # 或 'uniform'
    #         max_length_keys=5000,      # 这里是“key 个数”的最大跨度
    #         exp_scale=100,
    #         seed=42
    #     )

    #     # 保存为 [lo0,hi0, lo1,hi1, ...] 的 uint64 流
    #     out_path = f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.range.bin"
    #     queries.tofile(out_path)
    #     print(f"[+] save range queries to {out_path} successfully!")
    
    """ join """
    sizeList = [2e8]
    datasets = ["books"]
    num_queries = 1000000
    for dataset in datasets:
        for size in sizeList:
            print(f"[*] Generate queries for {dataset}_{int(size/1e6)}M_uint64_unique")
            raw = np.fromfile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique", dtype=np.uint64)
            keys = raw
            print(f"[*] Loaded {len(keys)} keys.")
            queries = generate_join_table_from_data(keys,num_queries,num_segments=20,active_segments=5,skew=1)
            queries.tofile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.{int(num_queries/1e6)}Mtable.bin")
            print(f"[+] save queries to {DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.{int(num_queries/1e6)}Mtable.bin successfully!")
    
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
    epsilon = 16
    queryfile = "books_200M_uint64_unique.1Mtable.bin"
    dataset = "books_200M_uint64_unique"
    raw = np.fromfile(f"{DATASETS_DIRECTORY}{dataset}", dtype=np.uint64)
    keys = raw
    queries = np.fromfile(f"{DATASETS_DIRECTORY}{queryfile}", dtype=np.uint64)
    lengths_file=f"{DATASETS_DIRECTORY}{queryfile}.par".replace(".bin","")
    bitmap_file=f"{DATASETS_DIRECTORY}{queryfile}.bitmap".replace(".bin","")
    join_partition(keys,queries,page_size,lengths_file=lengths_file,bitmap_file=bitmap_file,H=H,delta=delta,epsilon=epsilon)
            
if __name__ == '__main__':
    main()