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

# def generate_join_table_from_data(keys, num_queries=100000, seed=42,
#                                   num_segments=5, active_segments=2, skew=0.5):
#     """
#     生成带空page gap的 join query:
#       - 将 key space 划分为 num_segments 段
#       - 仅在 active_segments 个段里生成 query
#       - 用一个超参数 skew 控制各活跃段的“密度差异”（配额不均匀程度）
#         * skew < 1: 更集中（某几个段非常密）
#         * skew = 1: 中等不均
#         * skew > 1: 更平均
#     """
#     np.random.seed(seed)
#     random.seed(seed)

#     n = len(keys)
#     seg_size = n // num_segments
#     queries = []

#     # 选出若干活跃段
#     active_idx = random.sample(range(num_segments), active_segments)

#     # 用 Dirichlet(skew) 得到活跃段权重，再用多项分布分配查询总量
#     w = np.random.dirichlet(alpha=[skew] * active_segments)
#     quotas = np.random.multinomial(num_queries, w)

#     for j, seg in enumerate(active_idx):
#         start = seg * seg_size
#         end = min((seg + 1) * seg_size, n)
#         segment_keys = np.asarray(keys[start:end])  # 确保是 ndarray

#         this_q = int(quotas[j])
#         if this_q <= 0 or segment_keys.size == 0:
#             continue

#         # 段内仍保持“热点 + 均匀”两部分（可按需改比例）
#         hotspot_ratio = 0.6
#         hotspot_queries = int(this_q * hotspot_ratio)
#         uniform_queries = this_q - hotspot_queries

#         # hotspot: 取 segment 内的小范围 + Zipf 采样
#         hotspot_size = max(1, int(0.05 * len(segment_keys)))
#         base = random.randint(0, max(0, len(segment_keys) - hotspot_size))
#         hot_indices = np.random.zipf(1.5, hotspot_queries)
#         hot_indices = np.clip(hot_indices - 1, 0, hotspot_size - 1)  # zipf 从1起
#         queries.extend(segment_keys[base + hot_indices])

#         # uniform: 段内均匀采样；为保证配额，使用 replace=True
#         if uniform_queries > 0:
#             queries.extend(np.random.choice(segment_keys, size=uniform_queries, replace=True))

#     # 返回升序
#     join_keys = np.sort(np.array(queries, dtype=np.uint64))
#     return join_keys
def sample_unique_mixture(
    keys, k, seed=42,
    hotpot_ratio=0.4, zipf_ratio=0.3, uniform_ratio=0.3,
    num_hotpots=5, hotpot_frac=0.01,
    hotpot_zipf_a=1.5, zipf_a=1.2,
    oversample=20,                # 关键：一次性过采样倍数（建议 10~30）
    min_candidates=1_000_000,     # 防止 k 很小时候采样太少
    return_sorted=True,
    strict=True,                  # strict=True: 若一次性去重后仍不足 k，则报错；False: 用均匀无放回补齐
):
    """
    方案3：一次性生成 oversample*k 个候选 -> 按顺序去重 -> 取前 k 个唯一 key。
    相比“反复采样补齐”，对目标分布的破坏更小。

    keys: np.uint64 sorted array recommended
    """
    keys = np.asarray(keys, dtype=np.uint64)
    n = len(keys)
    if k > n:
        raise ValueError(f"k={k} > n={n}, cannot sample unique keys")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    m = max(int(k * oversample), min_candidates)

    # ---- 生成 m 个候选（一次性）----
    cand_parts = []

    # 1) hotspot
    m_hot = int(m * hotpot_ratio)
    if m_hot > 0:
        hotpot_size = max(1, int(hotpot_frac * n))
        # 把 m_hot 均分到多个 hotspot
        per = int(np.ceil(m_hot / num_hotpots))
        for _ in range(num_hotpots):
            base = random.randint(0, max(0, n - hotpot_size))
            idx = rng.zipf(hotpot_zipf_a, size=per) - 1
            idx = np.clip(idx, 0, hotpot_size - 1)
            cand_parts.append(keys[base + idx])

    # 2) zipf over full space
    m_zipf = int(m * zipf_ratio)
    if m_zipf > 0:
        idx = rng.zipf(zipf_a, size=m_zipf) - 1
        idx = np.clip(idx, 0, n - 1)
        cand_parts.append(keys[idx])

    # 3) uniform
    m_uni = m - m_hot - m_zipf
    if m_uni > 0:
        idx = rng.integers(0, n, size=m_uni, endpoint=False)
        cand_parts.append(keys[idx])

    if not cand_parts:
        raise ValueError("No candidates generated; check ratios")

    cand = np.concatenate(cand_parts).astype(np.uint64, copy=False)

    # 打乱候选顺序，使“按顺序去重”不过度偏向某一类候选
    rng.shuffle(cand)

    # ---- 单次按顺序去重，取前 k 个 ----
    chosen = np.empty(k, dtype=np.uint64)
    seen = set()
    cnt = 0
    for x in cand:
        xi = int(x)
        if xi in seen:
            continue
        seen.add(xi)
        chosen[cnt] = x
        cnt += 1
        if cnt >= k:
            break

    if cnt < k:
        if strict:
            raise RuntimeError(
                f"Not enough unique keys in one-shot oversample: got {cnt}, need {k}. "
                f"Try larger oversample (e.g., 30/50) or reduce hotspot_ratio/Zipf skew."
            )
        # fallback：用均匀无放回补齐（会引入少量分布偏差，但比多轮补齐小）
        remain = k - cnt
        # 从 keys 中挑剩余没选过的（注意：keys 很大时，这一步可能慢；只在极少数情况下触发）
        # 更高效的做法是随机抽 idx 并检查 seen，直到补齐。
        extra = []
        while len(extra) < remain:
            idx = int(rng.integers(0, n))
            v = int(keys[idx])
            if v not in seen:
                seen.add(v)
                extra.append(keys[idx])
        chosen[cnt:] = np.array(extra, dtype=np.uint64)
        cnt = k

    if return_sorted:
        chosen.sort()
    else:
        rng.shuffle(chosen)

    return chosen

    
# def join_partition(keys, queries, page_size=4096, key_size=8,
#                    lengths_file="", bitmap_file="", delta=37, H=0, epsilon=8):
#     """
#     Join-Partition Algorithm
#     input:
#       - keys: 大表升序 key (np.uint64)
#       - queries: join 表升序 key (np.uint64)
#       - page_size: SSD page 大小 (默认 4096B)
#       - key_size: key 大小 (默认 8B)
#     output:
#       - lengths: 每个 partition 的 query 数
#       - bitmap: 每个 partition 的策略 (0=point, 1=range)
#     """

#     ipp = page_size // key_size
#     num_pages = int(np.ceil(len(keys) / ipp))

#     # 将 key 映射到 page_id
#     key_to_page = np.arange(len(keys)) // ipp

#     # 用二分搜索找到 query 的位置 -> 对应 page id
#     query_idx = np.searchsorted(keys, queries)
#     query_idx = np.clip(query_idx, 0, len(keys) - 1)  # 防越界
#     query_pages = key_to_page[query_idx]

#     # 统计每个 page 被命中的次数
#     pageCount = np.bincount(query_pages, minlength=num_pages)
#     # print("Page Count:", pageCount)
#     # 划分连续的非零区间
#     partitions = []
#     start = None
#     for i, cnt in enumerate(pageCount):
#         if cnt > 0 and start is None:
#             start = i
#         elif cnt == 0 and start is not None:
#             partitions.append((start, i - 1))
#             start = None
#     if start is not None:
#         partitions.append((start, num_pages - 1))
#     print("Partitions:", partitions)
    
#     totalPagesRef = 0  
#     # 生成结果
#     lengths = []
#     bitmap = []
#     for (l, r) in partitions:
#         N = pageCount[l:r+1].sum()
#         K = r - l + 1
#         totalPagesRef += K
#         print("N:",N,"K:",K)
#         mu = N / K
#         lengths.append(N)
#         tau = (ipp+delta+H/K)/(math.log(2*epsilon,2)+H+delta)
#         print("τ:",tau)
#         print("μ:",mu)
#         if mu >= tau:
#             bitmap.append(1)  # range
#         else:
#             bitmap.append(0)  # point
#     print("Total Pages (Ref):", totalPagesRef)
#     print("Total Queries:", sum(lengths))
#     # show numbers of range and point
#     range_queries = sum(N for N, b in zip(lengths, bitmap) if b == 1)
#     point_queries = sum(N for N, b in zip(lengths, bitmap) if b == 0)
#     print("Range queries:", range_queries)
#     print("Point queries:", point_queries)
    
#     # save to file
#     np.array(lengths, dtype=np.int64).tofile(lengths_file)
#     np.array(bitmap, dtype=np.int8).tofile(bitmap_file)

#     return lengths, bitmap

def join_partition(
    keys: np.ndarray,
    queries: np.ndarray,
    alpha: float,
    beta: float,
    eta: float,
    lambda_point: float,
    lambda_range: float = None,
    page_size: int = 4096,
    key_size: int = 8,
    epsilon: int = 16,
    N_min: int = 4096,
    K_max: int = 1024,
    gamma: float = 0.05,      # hysteresis margin: require range <= (1-gamma)*point
    phi: float = 0.0,         # absolute gain threshold in seconds (optional)
    cooldown: int = 0,        # optional: after a range cut, force next partition to accumulate >=cooldown queries
    lengths_file: str = "",
    bitmap_file: str = "",
):
    """
    Partition by immediate threshold crossing:
      - cut immediately when range becomes clearly better than point (after N_min)
      - or cut when K reaches K_max
    bitmap: 0=point, 1=range
    lengths: #queries per partition
    """

    if lambda_range is None:
        lambda_range = lambda_point

    ipp = page_size // key_size
    assert ipp > 0

    if np.any(queries[1:] < queries[:-1]):
        queries = np.sort(queries)

    Q = len(queries)
    if Q == 0:
        if lengths_file:
            np.array([], dtype=np.int64).tofile(lengths_file)
        if bitmap_file:
            np.array([], dtype=np.int8).tofile(bitmap_file)
        return [], []

    # rank proxy (you can swap to model-predicted pos if available)
    idx = np.searchsorted(keys, queries, side="left")
    idx = np.clip(idx, 0, len(keys) - 1).astype(np.int64)

    lo = np.maximum(0, idx - int(epsilon))
    hi = np.minimum(len(keys) - 1, idx + int(epsilon))
    l_page = (lo // ipp).astype(np.int64)
    r_page = (hi // ipp).astype(np.int64)

    lengths, bitmap = [], []

    i = 0
    cool_left = 0  # cooldown counter

    while i < Q:
        N = 0
        pmin = int(l_page[i])
        pmax = int(r_page[i])

        # union length of page intervals for point (distinct pages)
        union_len = 0
        curL = None
        curR = None

        j = i
        cut_reason = None

        while j < Q:
            # extend window
            N += 1
            lj = int(l_page[j]); rj = int(r_page[j])

            if lj < pmin: pmin = lj
            if rj > pmax: pmax = rj
            K = pmax - pmin + 1

            # union merge
            if curL is None:
                curL, curR = lj, rj
                union_len = (curR - curL + 1)
            else:
                if lj > curR:
                    union_len += (rj - lj + 1)
                    curL, curR = lj, rj
                else:
                    if rj > curR:
                        union_len += (rj - curR)
                        curR = rj

            # enforce cooldown: do not allow early range cut right after a range partition
            eligible = (N >= N_min) and (cool_left <= 0)

            # hard cut by K_max once eligible (or always, if you prefer)
            if eligible and K >= K_max:
                cut_reason = "Kmax"
                break

            if eligible:
                cost_point = alpha * N + lambda_point * union_len
                cost_range = beta * K + eta + lambda_range * K
                gain = cost_point - cost_range

                # immediate trigger
                if (gain > phi) and (cost_range <= (1.0 - gamma) * cost_point):
                    cut_reason = "threshold"
                    break

            j += 1

        # decide partition end
        if j >= Q:
            j = Q - 1

        part_len = j - i + 1

        # decide strategy for this partition (evaluate at final window [i..j])
        # recompute final K and union_len is already for [i..j] in the loop
        # if loop ended because j == Q-1 without running eligible checks, we still decide here.
        # For correctness, re-evaluate with final K and union_len.
        # (K depends on pmin/pmax which are final values in loop.)
        K_final = pmax - pmin + 1
        cost_point_final = alpha * part_len + lambda_point * union_len
        cost_range_final = beta * K_final + eta + lambda_range * K_final
        gain_final = cost_point_final - cost_range_final

        use_range = (part_len >= N_min) and (gain_final > phi) and (cost_range_final <= (1.0 - gamma) * cost_point_final)

        bitmap.append(1 if use_range else 0)
        lengths.append(part_len)

        # update cooldown
        if use_range and cooldown > 0:
            cool_left = cooldown
        else:
            cool_left = max(0, cool_left - part_len)

        i = j + 1

    if lengths_file:
        np.array(lengths, dtype=np.int64).tofile(lengths_file)
    if bitmap_file:
        np.array(bitmap, dtype=np.int8).tofile(bitmap_file)
    print("[+] save partitions to", lengths_file, bitmap_file)
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
    # sizeList = [2e8]
    # datasets = ["books"]
    # num_queries = 1000000
    # for dataset in datasets:
    #     for size in sizeList:
    #         print(f"[*] Generate queries for {dataset}_{int(size/1e6)}M_uint64_unique")
    #         raw = np.fromfile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique", dtype=np.uint64)
    #         keys = raw
    #         print(f"[*] Loaded {len(keys)} keys.")
    #         # queries = sample_unique_mixture(keys,num_queries)
    #         queries = sample_unique_mixture(keys,num_queries,hotpot_ratio=0,zipf_ratio=0)
    #         print(f"[*] Loaded {len(queries)} queries.")
    #         queries.tofile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.{int(num_queries/1e6)}Mtable2.bin")
    #         print(f"[+] save queries to {DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.{int(num_queries/1e6)}Mtable2.bin successfully!")
    
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
    epsilon = 16
    queryfile = "books_200M_uint64_unique.1Mtable2.bin"
    dataset = "books_200M_uint64_unique"
    raw = np.fromfile(f"{DATASETS_DIRECTORY}{dataset}", dtype=np.uint64)
    keys = raw
    queries = np.fromfile(f"{DATASETS_DIRECTORY}{queryfile}", dtype=np.uint64)
    lengths_file=f"{DATASETS_DIRECTORY}{queryfile}.par".replace(".bin","")
    bitmap_file=f"{DATASETS_DIRECTORY}{queryfile}.bitmap".replace(".bin","")
    join_partition(keys,queries,alpha=1.168e-06,beta=5.831e-06,eta=0.121,lambda_point=2.763e-05,lambda_range=4.714e-06,
                   page_size=page_size,key_size=8,epsilon=epsilon,N_min=1000,K_max=8192
                   ,lengths_file=lengths_file,bitmap_file=bitmap_file)
            
if __name__ == '__main__':
    main()